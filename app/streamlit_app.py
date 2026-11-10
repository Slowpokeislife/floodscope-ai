from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
import sys

import numpy as np
from PIL import Image
import streamlit as st
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.inference import load_checkpoint, predict, overlay
from src.dataset import discover, IMG_PREFIX, MASK_PREFIX

st.set_page_config(page_title='FloodScope AI', page_icon='🌊', layout='wide')
st.title('FloodScope AI')
st.caption('Satellite flood segmentation · Research demo · Not for emergency decision-making')

@st.cache_resource
def cached_model(checkpoint_file):
    return load_checkpoint(checkpoint_file, device='cpu')

mode = st.radio('Image source', ['Labeled dataset', 'Upload image'], horizontal=True)
if mode == 'Labeled dataset':
    zip_path = Path('data/flood_data.zip')
    if not zip_path.exists():
        st.error('Copy flood_data.zip to data/flood_data.zip first.')
        st.stop()
    ids = discover(zip_path)
    sample_id = st.selectbox('Select a labeled sample', ids)
    with ZipFile(zip_path) as z:
        with z.open(IMG_PREFIX + sample_id + '.png') as f:
            image = Image.open(BytesIO(f.read())).convert('RGB')
        with z.open(MASK_PREFIX + sample_id + '.png') as f:
            gt = Image.open(BytesIO(f.read())).convert('L')
    gt_mask = (np.asarray(gt) == 255).astype(np.uint8)
else:
    upload = st.file_uploader('Upload a flood-dataset-style RGB PNG/JPG', type=['png', 'jpg', 'jpeg'])
    if upload is None:
        st.stop()
    image = Image.open(upload).convert('RGB')
    gt_mask = None

checkpoint = Path('checkpoints/best_model.pt')
if not checkpoint.exists():
    st.warning('No trained model found. Train first with: python -m src.train --epochs 15')
    cols = st.columns(2 if gt_mask is not None else 1)
    cols[0].image(image, caption='Input', use_container_width=True)
    if gt_mask is not None:
        cols[1].image((gt_mask * 255).astype('uint8'), caption='Ground truth', use_container_width=True)
    st.stop()

threshold = st.slider('Flood threshold', 0.05, 0.95, 0.50, 0.05)
model, size = cached_model(str(checkpoint))
mask, probs = predict(model, image, size, threshold=threshold)
cols = st.columns(3)
cols[0].image(image, caption='Input', use_container_width=True)
cols[1].image((mask * 255).astype('uint8'), caption='Predicted mask', use_container_width=True)
cols[2].image(overlay(image, mask), caption='Overlay', use_container_width=True)
st.metric('Predicted water pixels', f'{mask.mean() * 100:.1f}%')
st.caption('This is a pixel fraction, NOT affected area in km² or population exposed.')
if gt_mask is not None:
    from src.metrics import summarize_counts
    tp = int(((mask == 1) & (gt_mask == 1)).sum())
    fp = int(((mask == 1) & (gt_mask == 0)).sum())
    fn = int(((mask == 0) & (gt_mask == 1)).sum())
    tn = int(((mask == 0) & (gt_mask == 0)).sum())
    m = summarize_counts(tp, fp, fn, tn)
    st.write(f"**Dice:** {m['dice']:.3f} · **IoU:** {m['iou']:.3f}")
    st.image((gt_mask * 255).astype('uint8'), caption='Ground truth')
output = BytesIO()
Image.fromarray((mask * 255).astype('uint8')).save(output, format='PNG')
st.download_button('Export predicted mask', output.getvalue(), file_name='flood_mask.png', mime='image/png')
