# Diagnóstico de compatibilidad Nepal vs entrenamiento

- RGB entrenamiento: 446 imágenes; 446 máscaras revisadas para valores.
- Máscaras con valores distintos de 0/255: 0.
- Nepal: 2073 pares VV/VH, 80 muestreados.
- Fracción media de píxeles sospechosos: 35.79% (regla exploratoria).
- La red V4 espera RGB con valores [0,1], sin swap BGR ni normalización [-1,1].
- La receta de RGB del curso se usa solo para visualizar y comparar, no como equivalencia validada.
- Las imágenes PNG no permiten inferir directamente georreferenciación ni población afectada.

## Archivos
- report.json
- nepal_tile_stats.csv
- channel_comparison.png
- tile_coverage.png
- nepal_examples.png (si hay ejemplos)
