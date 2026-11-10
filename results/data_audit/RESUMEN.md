# FloodScope AI — auditoría de datos

## Resumen
- flood_data.zip: ok; archivos=1788; pares imagen/máscara=446
- UNOSAT_case_study.zip: ok; archivos=4152; pares VV/VH por nombre=2073
- splits.json: ok

## Tipos de archivo detectados
- flood: {"mask": 446, "metadata": 896, "training_image": 446}
- nepal: {"other_image": 6, "vh": 2073, "vv": 2073}

## Regiones y divisiones
- Regiones del dataset: {"Bolivia": 15, "Ghana": 53, "India": 68, "Mekong": 30, "Nigeria": 18, "Pakistan": 28, "Paraguay": 67, "Somalia": 26, "Spain": 30, "Sri-Lanka": 42, "USA": 69}
- Splits: {"train": {"samples": 292, "regions": ["Bolivia", "Ghana", "Nigeria", "Pakistan", "Paraguay", "Sri-Lanka", "USA"]}, "val": {"samples": 98, "regions": ["India", "Spain"]}, "test": {"samples": 56, "regions": ["Mekong", "Somalia"]}}

## Valores de máscaras muestreadas
- Bolivia_103757: valores={'0': 226782, '255': 35362}; flood_fraction=0.13489532470703125
- Ghana_24858: valores={'0': 261660, '255': 484}; flood_fraction=0.0018463134765625
- India_177319: valores={'0': 260549, '255': 1595}; flood_fraction=0.006084442138671875
- India_774689: valores={'0': 243289, '255': 18855}; flood_fraction=0.07192611694335938
- Mekong_774566: valores={'0': 185101, '255': 77043}; flood_fraction=0.2938957214355469
- Pakistan_667363: valores={'0': 260925, '255': 1219}; flood_fraction=0.004650115966796875
- Paraguay_339807: valores={'0': 204101, '255': 58043}; flood_fraction=0.22141647338867188
- Somalia_167787: valores={'0': 261919, '255': 225}; flood_fraction=0.000858306884765625
- Spain_716716: valores={'0': 259500, '255': 2644}; flood_fraction=0.0100860595703125
- Sri-Lanka_649970: valores={'0': 244801, '255': 17343}; flood_fraction=0.06615829467773438
- USA_354981: valores={'0': 260420, '255': 1724}; flood_fraction=0.0065765380859375
- USA_994009: valores={'0': 260247, '255': 1897}; flood_fraction=0.007236480712890625

## Nepal: archivos de metadatos (primeros 70)

## Advertencias y próximos pasos
- PIL convert(L) y y=(pixel == 255) pueden ocultar valores especiales; revisar el formato original.
- El notebook UNOSAT usa compuesto R=VV, G=VH, B=VV/VH, con R>0.7 a cero; NO confirma la preparación de los PNG de entrenamiento.
- No calcular superficies ni población impactada hasta confirmar georreferenciación y dimensiones de píxel.

Este reporte contiene nombres de archivo y estadísticas, no imágenes ni información de usuarios.
La inspección es exploratoria; no reutilizar test para ajustar hiperparámetros.
