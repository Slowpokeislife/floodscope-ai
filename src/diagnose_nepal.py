"""FloodScope AI: inspección de compatibilidad de Sen1Floods11 y Nepal (SIN inferencia).

Lee ZIP directamente, genera gráficos pequeños y reportes; no modifica datos.
"""
from __future__ import annotations
import argparse
import csv
import io
import json
import re
from collections import Counter
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from PIL import Image

TILE_RE = re.compile(r"^tiles_VV/tile_(\d+)_(\d+)\.png$", re.I)


def read_png(z, name):
    with z.open(name) as stream:
        with Image.open(io.BytesIO(stream.read())) as im:
            return np.asarray(im.copy())


def rgb_preview(vv, vh, mode='course'):
    # Valores de PNG uint8 tratados como intensidades normalizadas [0,1].
    v = vv.astype(np.float32) / 255.0
    h = vh.astype(np.float32) / 255.0
    if mode == 'course':
        r = np.where(v > 0.7, 0.0, v)
    else:
        r = v
    ratio = np.divide(v, h, out=np.zeros_like(v), where=h > 0)
    b = np.clip(ratio, 0, 1)
    rgb = np.stack([r, h, b], axis=-1)
    # VV=255/VH=0 son marcadores sospechosos según las estadísticas,
    # pero no se declaran no-data confirmado.
    suspect = (vv == 255) | (vh == 0)
    return rgb, suspect


def sample_stats(rgb, mask=None):
    x = rgb.reshape(-1, 3)
    if mask is not None:
        good = ~mask.reshape(-1)
        x = x[good]
    if len(x) == 0:
        return None
    return [{'mean': float(x[:,i].mean()), 'std': float(x[:,i].std()),
             'p01': float(np.quantile(x[:,i], 0.01)),
             'p50': float(np.quantile(x[:,i], 0.5)),
             'p99': float(np.quantile(x[:,i], 0.99))} for i in range(3)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--flood', default='data/flood_data.zip')
    p.add_argument('--nepal', default='data/UNOSAT_case_study.zip')
    p.add_argument('--outdir', default='results/nepal_diagnostics')
    p.add_argument('--train-samples', type=int, default=24)
    p.add_argument('--nepal-samples', type=int, default=80)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    report = {'warning': 'Diagnóstico visual/estadístico; NO demuestra compatibilidad SAR ni permite mapas de inundación fiables.',
              'input_v4': 'dataset.py: RGB uint8 -> float32 [0,1], CHW; sin ImageNet mean/std ni BGR.',
              'nepal_composite': 'R=VV con valores >0.7 reemplazados por 0, G=VH, B=clip(VV/VH,0,1); lectura PNG escalada /255; división segura.',
              'suspect_pixels': 'VV==255 OR VH==0 es regla exploratoria; NO es máscara no-data verificada.'}
    with ZipFile(args.flood) as z:
        images = sorted(n for n in z.namelist() if n.startswith('images/all_images/') and n.lower().endswith('.png'))
        masks = sorted(n for n in z.namelist() if n.startswith('masks/all_masks/') and n.lower().endswith('.png'))
        if not images: raise ValueError('No se encontraron imágenes de entrenamiento')
        index = rng.choice(len(images), size=min(args.train_samples,len(images)), replace=False)
        tr = []
        for i in index:
            rgb = read_png(z, images[int(i)])
            if rgb.ndim != 3 or rgb.shape[2] != 3: raise ValueError(f'RGB inválido: {images[int(i)]}')
            tr.append((images[int(i)], sample_stats(rgb.astype(np.float32)/255)))
        values = Counter(); unusual = []
        for f in masks:
            m = read_png(z, f)
            vals, counts = np.unique(m, return_counts=True)
            for v, n in zip(vals,counts): values[int(v)] += int(n)
            if not set(map(int, vals)).issubset({0,255}): unusual.append(f)
        report['training'] = {'image_count': len(images), 'mask_count':len(masks),
                              'sampled_rgb':len(tr), 'mask_unique_pixel_counts': dict(sorted(values.items())),
                              'mask_files_with_nonbinary_values': unusual[:30],
                              'mask_nonbinary_file_count': len(unusual)}

    with ZipFile(args.nepal) as z:
        names = set(z.namelist())
        vv_names = sorted([n for n in names if TILE_RE.match(n)])
        pairs = [(v, 'tiles_VH/' + v.split('/')[-1]) for v in vv_names if 'tiles_VH/'+v.split('/')[-1] in names]
        if not pairs: raise ValueError('No hay pares VV/VH en Nepal')
        ids = rng.choice(len(pairs), size=min(args.nepal_samples,len(pairs)), replace=False)
        nr = []; all_flags=[]; examples=[]
        for k in ids:
            fn,vhfn=pairs[int(k)]
            vv=read_png(z,fn); vh=read_png(z,vhfn)
            if vv.shape != vh.shape: raise ValueError(f'Dimensiones distintas: {fn}')
            if vv.ndim!=2 or vh.ndim!=2: raise ValueError(f'Se esperaban bandas L: {fn}')
            rgb, suspect=rgb_preview(vv,vh)
            row = {'tile':fn.split('/')[-1], 'suspect_fraction':float(suspect.mean()),
                   'vv_255_fraction':float((vv==255).mean()), 'vh_zero_fraction':float((vh==0).mean()),
                   'vv_std':float(vv.std()), 'vh_std':float(vh.std()),
                   'channels_all': sample_stats(rgb), 'channels_non_suspect':sample_stats(rgb,suspect)}
            nr.append(row); all_flags.append(row['suspect_fraction'])
            if .02 < row['suspect_fraction'] < .90 and row['vv_std'] > 3 and row['vh_std'] > 3: examples.append((fn,vhfn,row))
        report['nepal']={'pairs':len(pairs),'sampled_pairs':len(nr),
                         'mean_suspect_fraction_sample':float(np.mean(all_flags)),
                         'all_suspect_tiles_in_sample':sum(r['suspect_fraction']>=0.999 for r in nr),
                         'valid_examples':len(examples)}
        with (out/'nepal_tile_stats.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=['tile','suspect_fraction','vv_255_fraction','vh_zero_fraction','vv_std','vh_std'])
            w.writeheader();w.writerows({k:r[k] for k in w.fieldnames} for r in nr)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from PIL import Image as PILImage
        if examples:
            fig,axes=plt.subplots(min(4,len(examples)),5,figsize=(15,3*min(4,len(examples))),squeeze=False)
            for rowidx,(fn,vhfn,rec) in enumerate(examples[:4]):
                vv=read_png(z,fn);vh=read_png(z,vhfn)
                rgb,suspect=rgb_preview(vv,vh)
                views=[(vv,'VV PNG','gray'),(vh,'VH PNG','gray'),(rgb,'RGB compuesto',None),(suspect,'Píxeles sospechosos','magma'),(rgb_preview(vv,vh,'raw')[0],'RGB sin filtro VV',None)]
                for j,(arr,ttl,cmap) in enumerate(views):
                    axes[rowidx,j].imshow(arr,cmap=cmap,vmin=0,vmax=255 if j in (0,1) else None)
                    axes[rowidx,j].set_title(ttl+'\n'+rec['tile'] if j==0 else ttl,fontsize=9)
                    axes[rowidx,j].axis('off')
            fig.tight_layout();fig.savefig(out/'nepal_examples.png',dpi=130);plt.close(fig)
        else:
            report['nepal']['warning']='Ninguno de los mosaicos muestreados cumple el criterio de ejemplos; aumentar --nepal-samples.'
        # Previsualización de cobertura por índice de mosaicos, no coordenadas geográficas.
        coords=[(int(TILE_RE.match(f).group(1)),int(TILE_RE.match(f).group(2))) for f,_ in pairs]
        maxy,maxx=max(y for y,x in coords),max(x for y,x in coords)
        grid=np.zeros((maxy+1,maxx+1),dtype=np.uint8)
        for y,x in coords: grid[y,x]=1
        plt.figure(figsize=(12,7));plt.imshow(grid,cmap='Greys',interpolation='nearest');plt.title('Cobertura de mosaicos VV/VH (índices, NO geografía)');plt.xlabel('índice x');plt.ylabel('índice y');plt.tight_layout();plt.savefig(out/'tile_coverage.png',dpi=140);plt.close()
        report['nepal']['tile_index_bounds']={'max_x':maxx,'max_y':maxy}
        report['nepal']['baseline_maps']=[n for n in sorted(names) if n.startswith('baseline_maps/')]
    # Gráfico de distribuciones medias por muestra. No interpreta canales físicamente.
    fig, axes=plt.subplots(1,3,figsize=(12,4))
    for ch in range(3):
        t=[stats[ch]['mean'] for _,stats in tr]
        n=[r['channels_all'][ch]['mean'] for r in nr if r['channels_all']]
        axes[ch].hist(t,bins=12,alpha=.55,label='Sen1Floods11 RGB')
        axes[ch].hist(n,bins=12,alpha=.55,label='Nepal compuesto')
        axes[ch].set_title(f'Canal {ch} / media por mosaico')
        axes[ch].set_xlabel('Intensidad [0,1]')
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'channel_comparison.png',dpi=140);plt.close(fig)
    with (out/'report.json').open('w',encoding='utf-8') as f:json.dump(report,f,indent=2,ensure_ascii=False)
    lines=['# Diagnóstico de compatibilidad Nepal vs entrenamiento','',
           f'- RGB entrenamiento: {len(images)} imágenes; {len(masks)} máscaras revisadas para valores.',
           f'- Máscaras con valores distintos de 0/255: {len(unusual)}.',
           f'- Nepal: {len(pairs)} pares VV/VH, {len(nr)} muestreados.',
           f'- Fracción media de píxeles sospechosos: {np.mean(all_flags):.2%} (regla exploratoria).',
           '- La red V4 espera RGB con valores [0,1], sin swap BGR ni normalización [-1,1].',
           '- La receta de RGB del curso se usa solo para visualizar y comparar, no como equivalencia validada.',
           '- Las imágenes PNG no permiten inferir directamente georreferenciación ni población afectada.',
           '', '## Archivos','- report.json','- nepal_tile_stats.csv','- channel_comparison.png','- tile_coverage.png','- nepal_examples.png (si hay ejemplos)']
    (out/'RESUMEN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('\n'.join(lines));print('\nGuardado en',out)

if __name__=='__main__':main()
