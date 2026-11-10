"""FloodScope AI / Etapa 5: auditoría de ZIPs sin extraerlos.

Dependencias: numpy, Pillow (ya instaladas en FloodScope AI).
Uso: python -m src.audit_data --flood data/flood_data.zip --nepal data/UNOSAT_case_study.zip
"""
import argparse
import csv
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
META_EXTS = {'.json', '.geojson', '.tfw', '.prj', '.wld', '.xml', '.vrt', '.shp', '.dbf', '.shx', '.gpkg', '.kml', '.csv', '.txt'}


def classify(name):
    p = Path(name.lower())
    part = p.parts
    ext = p.suffix
    if ext not in IMAGE_EXTS:
        return 'metadata' if ext in META_EXTS else 'other'
    if 'all_masks' in part or 'masks' in part or 'mask' in p.stem:
        return 'mask'
    if 'tiles_vv' in part or re.search(r'(^|[_-])vv([_-]|$)', p.stem):
        return 'vv'
    if 'tiles_vh' in part or re.search(r'(^|[_-])vh([_-]|$)', p.stem):
        return 'vh'
    if 'all_images' in part or 'images' in part:
        return 'training_image'
    return 'other_image'


def inspect_image(z, name, sample_arrays=True):
    with z.open(name) as f:
        b = f.read()
    with Image.open(io.BytesIO(b)) as im:
        width, height = im.size
        fmt = im.format
        mode = im.mode
        a = np.asarray(im)
    data = {'file': name, 'width': width, 'height': height, 'format': fmt,
            'mode': mode, 'shape': list(a.shape), 'dtype': str(a.dtype)}
    if not sample_arrays:
        return data
    if a.ndim == 2:
        a = a[..., None]
    elif a.ndim != 3:
        data['warning'] = 'Unusual image dimensions'; return data
    chans = []
    for i in range(min(4, a.shape[-1])):
        x = a[..., i].astype(np.float64)
        valid = np.isfinite(x)
        y = x[valid]
        if y.size:
            channel = {'channel': i, 'min': float(y.min()), 'max': float(y.max()),
                       'mean': float(y.mean()), 'std': float(y.std()),
                       'zero_fraction': float(np.mean(y == 0)),
                       'nan_or_inf_count': int(np.size(x) - y.size),
                       'q01': float(np.quantile(y, .01)), 'q50': float(np.median(y)),
                       'q99': float(np.quantile(y, .99))}
            if len(np.unique(y)) <= 20:
                channel['unique_values'] = [float(v) for v in np.unique(y)]
            chans.append(channel)
    data['channels'] = chans
    return data


def choose_evenly(items, count):
    if count <= 0 or not items:
        return []
    if len(items) <= count:
        return items
    return [items[int(i)] for i in np.linspace(0, len(items) - 1, count)]


def audit_archive(path, n, label):
    if not path.is_file():
        return {'status': 'missing', 'path': str(path), 'message': 'Archivo no encontrado'}
    result = {'status': 'ok', 'path': str(path), 'size_bytes': path.stat().st_size}
    with zipfile.ZipFile(path) as z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        by_category = defaultdict(list)
        for i in infos:
            by_category[classify(i.filename)].append(i.filename)
        result['total_files'] = len(infos)
        result['categories'] = {k: len(v) for k, v in sorted(by_category.items())}
        result['top_level_names'] = sorted({i.filename.split('/')[0] for i in infos})[:30]
        metadata = sorted(by_category.get('metadata', []))
        result['metadata_files'] = metadata[:70]
        result['metadata_truncated'] = max(0, len(metadata) - 70)
        result['sample_records'] = {}
        for group in ('training_image', 'mask', 'vv', 'vh', 'other_image'):
            files = sorted(by_category.get(group, []))
            selection = choose_evenly(files, n)
            records = []
            for name in selection:
                try:
                    records.append(inspect_image(z, name))
                except Exception as exc:
                    records.append({'file': name, 'error': f'{type(exc).__name__}: {exc}'})
            result['sample_records'][group] = records
        if label == 'flood':
            imgs = {Path(p).stem: p for p in by_category.get('training_image', [])}
            masks = {Path(p).stem: p for p in by_category.get('mask', [])}
            pairs = sorted(set(imgs) & set(masks))
            result['paired_samples'] = len(pairs)
            result['unpaired_images'] = sorted(set(imgs) - set(masks))[:20]
            result['unpaired_masks'] = sorted(set(masks) - set(imgs))[:20]
            result['regions'] = dict(sorted(Counter(s.rsplit('_', 1)[0] for s in pairs).items()))
            masks_used = choose_evenly(pairs, n)
            mask_stats = []
            for stem in masks_used:
                try:
                    rec = inspect_image(z, masks[stem])
                    with z.open(masks[stem]) as f:
                        with Image.open(f) as img:
                            a = np.asarray(img.convert('L'))
                    unique, counts = np.unique(a, return_counts=True)
                    mask_stats.append({'sample': stem, 'original_mode': rec['mode'],
                                       'pixel_values': {str(int(k)): int(v) for k, v in zip(unique, counts)},
                                       'flood_fraction_255': float(np.mean(a == 255)),
                                       'non_0_255_count': int(np.sum((a != 0) & (a != 255)))})
                except Exception as exc:
                    mask_stats.append({'sample': stem, 'error': str(exc)})
            result['mask_stats'] = mask_stats
            result['sample_pairs'] = pairs[:12]
        if label == 'nepal':
            vv = by_category.get('vv', [])
            vh = by_category.get('vh', [])
            vv_files = {Path(x).name.lower(): x for x in vv}
            vh_files = {Path(x).name.lower(): x for x in vh}
            common = sorted(set(vv_files) & set(vh_files))
            result['vv_vh_same_filename_pairs'] = len(common)
            result['vv_only_examples'] = sorted(set(vv_files) - set(vh_files))[:10]
            result['vh_only_examples'] = sorted(set(vh_files) - set(vv_files))[:10]
            result['tile_name_examples'] = common[:12]
            result['tile_coordinate_extent'] = None
            coords = [tuple(map(int, m.groups())) for item in common
                      if (m := re.fullmatch(r'tile_(\d+)_(\d+)\.[a-z0-9]+', item))]
            if coords:
                ys, xs = zip(*coords)
                result['tile_coordinate_extent'] = {'min_y': min(ys), 'max_y': max(ys),
                                                    'min_x': min(xs), 'max_x': max(xs),
                                                    'coordinate_filename_matches': len(coords)}
    return result


def load_splits(path):
    if not path.exists():
        return {'status': 'missing', 'path': str(path)}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        summary = {'status': 'ok', 'path': str(path), 'groups': {}}
        for k, value in data.items():
            if isinstance(value, list):
                summary['groups'][k] = {'samples': len(value), 'regions': sorted({v.rsplit('_', 1)[0] for v in value if isinstance(v, str)})}
        return summary
    except Exception as exc:
        return {'status': 'error', 'detail': str(exc)}


def build_warnings(flood, nepal, splits):
    warnings = []
    if flood.get('status') != 'ok':
        warnings.append('No se ha podido auditar flood_data.zip.')
    else:
        if any(x.get('non_0_255_count', 0) > 0 for x in flood.get('mask_stats', [])):
            warnings.append('Hay máscaras con valores distintos a 0 y 255: inspeccionar antes de entrenar o evaluar.')
        warnings.append('PIL convert(L) y y=(pixel == 255) pueden ocultar valores especiales; revisar el formato original.')
    if nepal.get('status') != 'ok':
        warnings.append('No se ha podido auditar UNOSAT_case_study.zip; indicar su ruta o nombre real.')
    else:
        if not nepal.get('vv_vh_same_filename_pairs'):
            warnings.append('No se detectaron pares VV/VH con idéntico nombre de archivo; comprobar rutas y nomenclatura.')
        warnings.append('El notebook UNOSAT usa compuesto R=VV, G=VH, B=VV/VH, con R>0.7 a cero; NO confirma la preparación de los PNG de entrenamiento.')
        warnings.append('No calcular superficies ni población impactada hasta confirmar georreferenciación y dimensiones de píxel.')
    if splits.get('status') != 'ok':
        warnings.append('No se pudo verificar results/splits.json.')
    return warnings


def main():
    ap = argparse.ArgumentParser(description='Auditoría de imágenes, máscaras, tiles y metadatos FloodScope AI')
    ap.add_argument('--flood', default='data/flood_data.zip')
    ap.add_argument('--nepal', default='data/UNOSAT_case_study.zip')
    ap.add_argument('--splits', default='results/splits.json')
    ap.add_argument('--outdir', default='results/data_audit')
    ap.add_argument('--samples-per-category', type=int, default=12)
    args = ap.parse_args()
    n = max(1, min(args.samples_per_category, 50))
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    print('Auditing ZIPs (no extraction, no model training)...', flush=True)
    flood = audit_archive(Path(args.flood), n, 'flood')
    nepal = audit_archive(Path(args.nepal), n, 'nepal')
    splits = load_splits(Path(args.splits))
    report = {'flood': flood, 'nepal': nepal, 'splits': splits,
              'source_note': 'UNOSAT original notebook: RGB red=clip(VV,0,1), values red>0.7 set to 0; green=clip(VH,0,1); blue=clip(VV/VH,0,1); inference additionally normalizes and swaps RGB to BGR. Verify numerical handling of division by zero before implementation.',
              'limitations': ['Images are sampled, not all image pixels in the archive.',
                              'Archive metadata listing does not establish georeferencing.',
                              'RGB training PNG source and pixel preprocessing remain unverified.'],
              'warnings': build_warnings(flood, nepal, splits)}
    (out / 'audit_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    rows = []
    for archive, body in (('flood', flood), ('nepal', nepal)):
        for group, records in body.get('sample_records', {}).items():
            for rec in records:
                for channel in rec.get('channels', []):
                    rows.append({'archive': archive, 'group': group, 'file': rec.get('file'),
                                 'width': rec.get('width'), 'height': rec.get('height'),
                                 'mode': rec.get('mode'), 'dtype': rec.get('dtype'), **channel})
    with (out / 'sample_channel_stats.csv').open('w', newline='', encoding='utf-8-sig') as f:
        cols = ['archive','group','file','width','height','mode','dtype','channel',
                'min','max','mean','std','zero_fraction','nan_or_inf_count','q01','q50','q99']
        writer = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        '# FloodScope AI — auditoría de datos', '',
        '## Resumen',
        f'- flood_data.zip: {flood.get("status")}; archivos={flood.get("total_files", "?")}; pares imagen/máscara={flood.get("paired_samples", "?")}',
        f'- UNOSAT_case_study.zip: {nepal.get("status")}; archivos={nepal.get("total_files", "?")}; pares VV/VH por nombre={nepal.get("vv_vh_same_filename_pairs", "?")}',
        f'- splits.json: {splits.get("status")}', '',
        '## Tipos de archivo detectados',
        f'- flood: {json.dumps(flood.get("categories", {}), ensure_ascii=False)}',
        f'- nepal: {json.dumps(nepal.get("categories", {}), ensure_ascii=False)}', '',
        '## Regiones y divisiones',
        f'- Regiones del dataset: {json.dumps(flood.get("regions", {}), ensure_ascii=False)}',
        f'- Splits: {json.dumps(splits.get("groups", {}), ensure_ascii=False)}', '',
        '## Valores de máscaras muestreadas',
    ]
    for row in flood.get('mask_stats', []):
        lines.append(f'- {row.get("sample")}: valores={row.get("pixel_values", {})}; flood_fraction={row.get("flood_fraction_255", "?")}')
    lines.extend(['', '## Nepal: archivos de metadatos (primeros 70)'])
    lines += [f'- {name}' for name in nepal.get('metadata_files', [])]
    lines.extend(['', '## Advertencias y próximos pasos'])
    lines += [f'- {note}' for note in report['warnings']]
    lines.extend(['', 'Este reporte contiene nombres de archivo y estadísticas, no imágenes ni información de usuarios.',
                  'La inspección es exploratoria; no reutilizar test para ajustar hiperparámetros.'])
    (out / 'RESUMEN.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\nFLOOD:', flood.get('status'), '| groups:', flood.get('categories', {}), '| pairs:', flood.get('paired_samples'))
    print('NEPAL:', nepal.get('status'), '| groups:', nepal.get('categories', {}), '| VV/VH pairs:', nepal.get('vv_vh_same_filename_pairs'))
    print('SPLITS:', splits.get('status'))
    for warning in report['warnings']:
        print('NOTE:', warning)
    print('\nSaved:', out / 'RESUMEN.md')
    print('Saved:', out / 'audit_report.json')
    print('Saved:', out / 'sample_channel_stats.csv')


if __name__ == '__main__':
    main()
