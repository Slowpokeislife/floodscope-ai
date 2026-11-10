# FloodScope AI — Dashboard (estilo Dopper)

## Contenido
- `app/dashboard.py` — cuatro vistas: Overview, Model Explorer, Geographic Evaluation, Nepal Domain Shift
- `app/dopper_theme.css` — tema crema, navy, azul eléctrico y tarjetas de 20 px
- `app/assets/` — tres imágenes de diagnóstico de Nepal (creadas con tus datos)
- `requirements_dashboard.txt` — dependencias del dashboard

## Instalación (Windows / PowerShell)

1. Extrae este ZIP.
2. Copia la carpeta `app/` dentro de tu proyecto `FloodScope-AI/`. Si ya tienes `app/streamlit_app.py`, **déjalo intacto**: solo agrega `dashboard.py`, `dopper_theme.css` y `assets/`.
3. Copia `requirements_dashboard.txt` a la raíz de `FloodScope-AI/`.
4. Abre PowerShell en la raíz de tu proyecto, activa `.venv` y ejecuta:

```powershell
pip install -r requirements_dashboard.txt
streamlit run app/dashboard.py
```

5. La app se abrirá en `http://localhost:8501` (o en el puerto que indique Streamlit).

## Archivos que debes mantener en tu proyecto

```text
FloodScope-AI/
├── app/
│   ├── dashboard.py
│   ├── dopper_theme.css
│   └── assets/
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── inference.py
│   └── model_v4.py
├── data/
│   └── flood_data.zip
├── checkpoints/
│   ├── v2_512_best.pt
│   └── v4_deeplab_best.pt
└── results/
    ├── splits.json
    ├── v2_512_thresholds.json
    ├── v4_deeplab_thresholds.json
    ├── v2_512_test.json
    └── v4_deeplab_test.json
```

El dashboard muestra Overview, Geographic Evaluation y Nepal aunque falten los checkpoints. Para usar la inferencia interactiva necesitas `flood_data.zip`, `src/` y uno o ambos checkpoints.

### Importante sobre las métricas
- En Overview y Geographic Evaluation se muestran resultados históricos de referencia cuando falta un JSON local. Se señalan como tales en la interfaz.
- El Model Explorer calcula **métricas por imagen**, que no son comparables directamente con las métricas agregadas sobre todos los píxeles.
- En el conjunto de prueba, los umbrales están fijados según validación. No cambies configuraciones buscando mejorar el test.
- En Nepal **no se publican mapas de inundación ni estimaciones de población**: no existe ground truth y la representación de las entradas no está validada.

### Compatibilidad
- Python 3.11–3.13, con PyTorch compatible con tu GPU.
- Reutiliza la arquitectura definida en `src/model.py` (U-Net) y `src/model_v4.py` (DeepLabV3+).
- No incluye datasets, pesos ni el material del curso NVIDIA.
- Las fuentes Google Fonts son opcionales; hay fallbacks locales.

### Si aparece un error
- `No module named segmentation_models_pytorch` → `pip install segmentation-models-pytorch`
- `No module named src` → Ejecuta Streamlit desde la raíz `FloodScope-AI/`, no desde `app/`.
- `CUDA out of memory` → cierra otras aplicaciones GPU; puedes cambiar `device = "cuda" ...` a `device = "cpu"` dentro de `model_explorer()` para inferencia CPU.
