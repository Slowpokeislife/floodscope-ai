"""FloodScope AI | Dopper-inspired Streamlit dashboard.

Run from project root:
    streamlit run app/dashboard.py

Uses your existing src/, model checkpoints and results. Does not alter experiments.
"""
from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NAVY, BLUE, TEAL, YELLOW, SKY, VIOLET = "#000f2e", "#0067e5", "#116973", "#fed200", "#8ab1e6", "#515a8a"
DATA_ZIP = ROOT / "data/flood_data.zip"
SPLITS = ROOT / "results/splits.json"
MODEL_INFO = {
    "V2 · U-Net / ResNet-18": {"checkpoint": "checkpoints/v2_512_best.pt", "threshold": "results/v2_512_thresholds.json", "test": "results/v2_512_test.json", "kind": "unet", "default_threshold": .40},
    "V4 · DeepLabV3+ / ResNet-18": {"checkpoint": "checkpoints/v4_deeplab_best.pt", "threshold": "results/v4_deeplab_thresholds.json", "test": "results/v4_deeplab_test.json", "kind": "deeplab", "default_threshold": .45},
}
FALLBACK = {
    "V1 · U-Net 256": {"validation": .7198, "test": .6527773309280716, "iou": .484535590080113},
    "V2 · U-Net 512": {"validation": .7301, "test": .679915453683058, "iou": .5150544755485803},
    "V3 · U-Net Tversky": {"validation": .7270, "test": None, "iou": None},
    "V4 · DeepLabV3+": {"validation": .7189, "test": .7184560741788449, "iou": .5606175954667266},
}
RESULT_PATHS = {
    "V1 · U-Net 256": ("results/threshold_search.json", "results/v1_test_threshold060.json"),
    "V2 · U-Net 512": ("results/v2_512_thresholds.json", "results/v2_512_test.json"),
    "V3 · U-Net Tversky": ("results/v3_tversky_thresholds.json", None),
    "V4 · DeepLabV3+": ("results/v4_deeplab_thresholds.json", "results/v4_deeplab_test.json"),
}


def read_json(path: Path, fallback=None):
    try:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    return fallback


def pct(x):
    return "—" if x is None else f"{100 * float(x):.2f}%"


def metric_tile(title, value, desc, variant="blue"):
    st.markdown(
        f'<div class="tile tile-{variant}"><div><h3>{title}</h3><p>{desc}</p></div><div class="big">{value}</div></div>',
        unsafe_allow_html=True,
    )


def hero(label, title, subtitle):
    st.markdown(
        f'<section class="hero"><div class="eyebrow">{label}</div><h1>{title}</h1><p>{subtitle}</p></section>',
        unsafe_allow_html=True,
    )


def section(title, description=""):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)
    if description:
        st.markdown(f'<p class="muted-copy">{description}</p>', unsafe_allow_html=True)


def get_threshold(info):
    doc = read_json(ROOT / info["threshold"])
    try:
        return float(doc["best_threshold"])
    except (TypeError, ValueError, KeyError):
        return info["default_threshold"]


def display_image(image, label):
    st.image(image, caption=label, use_container_width=True)


@st.cache_data(show_spinner=False)
def dataset_ids(zip_path, zip_mtime):
    from src.dataset import discover
    return discover(zip_path)


@st.cache_data(show_spinner=False)
def load_sample(zip_path, zip_mtime, sample_id):
    with ZipFile(zip_path) as z:
        rgb = Image.open(io.BytesIO(z.read("images/all_images/" + sample_id + ".png"))).convert("RGB")
        mask = Image.open(io.BytesIO(z.read("masks/all_masks/" + sample_id + ".png"))).convert("L")
    x = np.array(rgb, dtype=np.uint8)
    y = (np.array(mask, dtype=np.uint8) == 255).astype(np.uint8)
    return x, y


@st.cache_resource(show_spinner="Loading model weights on your machine…")
def get_model(model_key, checkpoint_path, checkpoint_mtime, device_name):
    import torch
    info = MODEL_INFO[model_key]
    if info["kind"] == "unet":
        from src.inference import load_checkpoint
        model, size = load_checkpoint(checkpoint_path, device=device_name)
    else:
        from src.model_v4 import load_v4
        model, size = load_v4(checkpoint_path, torch.device(device_name))
    return model, size


@st.cache_data(show_spinner="Computing segmentation…", max_entries=12)
def infer_png(model_key, checkpoint_path, checkpoint_mtime, image_bytes, device_name):
    import torch
    model, size = get_model(model_key, checkpoint_path, checkpoint_mtime, device_name)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_image = image.resize((size, size), Image.Resampling.BILINEAR)
    arr = np.asarray(input_image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr.transpose(2, 0, 1).copy())[None].to(device_name)
    with torch.inference_mode():
        output = model(tensor)
        probs = torch.sigmoid(output)[0, 0].cpu().numpy().astype(np.float32)
    if probs.shape != (image.height, image.width):
        probs = np.asarray(Image.fromarray(probs, mode="F").resize(image.size, Image.Resampling.BILINEAR))
    return probs


def overlay(image, binary, alpha=.52):
    out = image.astype(np.float32).copy()
    tint = np.array([0, 103, 229], dtype=np.float32)
    out[binary] = out[binary] * (1 - alpha) + tint * alpha
    return out.clip(0, 255).astype(np.uint8)


def error_map(binary, truth):
    canvas = np.zeros((*truth.shape, 3), dtype=np.uint8)
    canvas[:] = [0, 15, 46]
    canvas[binary & truth] = [22, 183, 126]
    canvas[binary & ~truth] = [245, 69, 79]
    canvas[~binary & truth] = [254, 210, 0]
    return canvas


def pixel_metrics(binary, truth):
    tp = int((binary & truth).sum())
    fp = int((binary & ~truth).sum())
    fn = int((~binary & truth).sum())
    tn = int((~binary & ~truth).sum())
    d = 2 * tp + fp + fn
    u = tp + fp + fn
    return {"dice": 1.0 if d == 0 else 2 * tp / d,
            "iou": 1.0 if u == 0 else tp / u,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def png_bytes(arr):
    out = io.BytesIO()
    Image.fromarray(arr).save(out, format="PNG")
    return out.getvalue()


def report_rows():
    rows = []
    for name, fallback in FALLBACK.items():
        validation, test = RESULT_PATHS[name]
        vjson = read_json(ROOT / validation, {}) if validation else {}
        tjson = read_json(ROOT / test, {}) if test else {}
        v = vjson.get("best_dice", vjson.get("best_validation_dice", fallback["validation"]))
        # Original saved threshold schema may use "best_metrics"; display the known reference otherwise.
        if isinstance(vjson.get("best_metrics"), dict):
            v = vjson["best_metrics"].get("dice", v)
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = fallback["validation"]
        global_test = tjson.get("global", {})
        test_dice = global_test.get("dice", fallback["test"])
        test_iou = global_test.get("iou", fallback["iou"])
        rows.append({"Model": name, "Validation Dice": v, "Test Dice": test_dice, "Test IoU": test_iou, "has_test_json": bool(tjson), "has_validation_json": bool(vjson)})
    return rows


def overview():
    hero("FloodScope / Research dashboard", "Understand floods.<br>Respect uncertainty.",
         "Satellite-image segmentation · Four controlled experiments · Regional evaluation and transparent deployment limitations.")
    st.markdown('<div class="ticker"><span>01  MODEL RESEARCH</span><b>✦</b><span>02  GEOGRAPHIC VALIDATION</span><b>✦</b><span>03  ERROR ANALYSIS</span><b>✦</b><span>04  DOMAIN SHIFT</span></div>', unsafe_allow_html=True)
    rows = report_rows()
    v2 = next(x for x in rows if x["Model"].startswith("V2"))
    v4 = next(x for x in rows if x["Model"].startswith("V4"))
    a, b, c, d = st.columns(4, gap="medium")
    with a: metric_tile("Labeled samples", "446", "Sen1Floods11 · 11 regions", "yellow")
    with b: metric_tile("Validation leader", pct(v2["Validation Dice"]), "U-Net V2 · 512 × 512", "teal")
    with c: metric_tile("Observed test leader", pct(v4["Test Dice"]), "DeepLabV3+ V4 · exploratory comparison", "blue")
    with d: metric_tile("Regional holdout", "2 areas", "Mekong and Somalia · 56 samples", "sky")
    section("A model is only as good as its evaluation.",
            "Comparison of observed Dice scores. V2 was preferred by validation; V4 obtained the highest observed test Dice. Repeated test comparisons are exploratory, not evidence of a new unbiased winner.")
    chart = pd.DataFrame([{"Experiment": x["Model"].split(" · ")[0], "Validation Dice": x["Validation Dice"], "Test Dice": x["Test Dice"]} for x in rows])
    st.bar_chart(chart.set_index("Experiment"), y=["Validation Dice", "Test Dice"], color=[BLUE, TEAL], height=320)
    with st.expander("Experiment-level data and provenance"):
        view = pd.DataFrame([{"Model": r["Model"], "Validation Dice": pct(r["Validation Dice"]),
                              "Test Dice": pct(r["Test Dice"]), "Test IoU": pct(r["Test IoU"]),
                              "Local test JSON": "Yes" if r["has_test_json"] else "Reference / not available"} for r in rows])
        st.dataframe(view, hide_index=True, use_container_width=True)
        st.caption("Where local result JSON is missing, the known results from your completed experiments are shown as reference values. V3 has not been evaluated on test.")
    section("What makes this project different?", "Not just a prediction model: an explicit examination of failure modes and domain shift.")
    a, b, c = st.columns(3)
    with a: metric_tile("Regional splits", "7 / 2 / 2", "Train / validation / held-out regions", "teal")
    with b: metric_tile("V4 precision on test", "63.44%", "With recall at 82.82%; pooled pixels", "yellow")
    with c: metric_tile("Nepal status", "Not validated", "Input representation differs from training", "navy")
    st.markdown('<div class="notice">All reported test metrics pool pixels across samples. No-data semantics and image-level uncertainty have not been fully validated. The dashboard is a research demo, not an emergency-response tool.</div>', unsafe_allow_html=True)


def model_explorer():
    hero("02 / Model explorer", "See what the model sees.",
         "Select one labeled Sentinel-1 sample, compare U-Net and DeepLabV3+ predictions, and inspect their errors against reference masks.")
    if not DATA_ZIP.is_file():
        st.error("Dataset missing: copy your original ZIP to data/flood_data.zip. The dashboard does not include the source dataset.")
        return
    all_ids = dataset_ids(str(DATA_ZIP), DATA_ZIP.stat().st_mtime)
    splits = read_json(SPLITS, {})
    allowed = ["Validation", "Training", "Test (locked thresholds)"] if splits else ["All (no splits file)"]
    selected_split = st.selectbox("Dataset split", allowed, index=0)
    if selected_split == "Validation":
        ids = [i for i in splits["val"] if i in all_ids]
    elif selected_split == "Training":
        ids = [i for i in splits["train"] if i in all_ids]
    elif selected_split.startswith("Test"):
        ids = [i for i in splits["test"] if i in all_ids]
        st.info("Test samples are shown for interpretation only. Their thresholds are locked to validation-selected values. Do not use them to choose configurations.")
    else:
        ids = all_ids
    c1, c2 = st.columns([1, 2])
    with c1:
        region_options = sorted({i.rsplit("_", 1)[0] for i in ids})
        chosen_region = st.selectbox("Region", region_options)
    ids = [i for i in ids if i.rsplit("_", 1)[0] == chosen_region]
    with c2:
        sample_id = st.selectbox("Satellite tile", ids)
    image, gt = load_sample(str(DATA_ZIP), DATA_ZIP.stat().st_mtime, sample_id)
    st.markdown("<div class='section-header'>Reference image & label</div>", unsafe_allow_html=True)
    l, r = st.columns(2)
    with l: display_image(image, "Input · three-channel SAR-derived RGB PNG")
    with r: display_image((gt * 255).astype(np.uint8), "Ground truth · 255 = positive")
    section("Inference studio", "Run one or both trained networks. Thresholds default to the selection made on validation.")
    available = [key for key, info in MODEL_INFO.items() if (ROOT / info["checkpoint"]).exists()]
    if not available:
        st.warning("No V2/V4 checkpoint found. Copy your own model weights into checkpoints/ and refresh the dashboard.")
        return
    selected = st.multiselect("Models", available, default=available)
    if not selected:
        st.info("Select at least one model.")
        return
    threshold_override = None
    if not selected_split.startswith("Test"):
        mode = st.radio("Threshold mode", ["Validation-selected", "Explore a custom threshold"], horizontal=True)
        if mode == "Explore a custom threshold":
            threshold_override = st.slider("Exploratory threshold", min_value=.05, max_value=.95, value=.50, step=.05)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.caption(f"Inference device: {device.upper()} · Tile: {sample_id} · This is pixel-level segmentation, not an estimate of km².")
    raw = png_bytes(image)
    for key in selected:
        info = MODEL_INFO[key]
        checkpoint = ROOT / info["checkpoint"]
        chosen_threshold = threshold_override if threshold_override is not None else get_threshold(info)
        st.markdown(f"### {key}")
        try:
            prob = infer_png(key, str(checkpoint), checkpoint.stat().st_mtime, raw, device)
        except Exception as ex:
            st.error(f"Could not run {key}: {type(ex).__name__}: {ex}")
            continue
        binary = prob >= chosen_threshold
        truth = gt.astype(bool)
        values = pixel_metrics(binary, truth)
        x1, x2, x3, x4 = st.columns(4)
        x1.metric("Dice · this tile", pct(values["dice"]))
        x2.metric("IoU · this tile", pct(values["iou"]))
        x3.metric("Precision · this tile", pct(values["precision"]))
        x4.metric("Recall · this tile", pct(values["recall"]))
        a, b, c = st.columns(3)
        with a: display_image(overlay(image, binary), "Overlay · predicted positive in blue")
        with b: display_image((binary * 255).astype(np.uint8), f"Prediction · threshold {chosen_threshold:.2f}")
        with c: display_image(error_map(binary, truth), "Errors · green TP / red FP / yellow FN")
        st.caption("Empty-mask convention: Dice and IoU are 1 when prediction and truth are both empty; precision/recall are undefined when their denominator is zero.")
        p, q = st.columns([1, 1])
        p.download_button("Download mask PNG", png_bytes((binary * 255).astype(np.uint8)), file_name=f"{sample_id}_{info['kind']}_mask.png", mime="image/png", key=f"mask_{key}", use_container_width=True)
        q.download_button("Download overlay PNG", png_bytes(overlay(image, binary)), file_name=f"{sample_id}_{info['kind']}_overlay.png", mime="image/png", key=f"overlay_{key}", use_container_width=True)


def geographic_evaluation():
    hero("03 / Geographic evaluation", "Performance changes with place.",
         "Compare regional Dice, precision, and recall. Evaluate failure modes without hiding the difference between validation and test.")
    valid_v2 = read_json(ROOT / "results/error_analysis_v2/validation_summary.json")
    valid_v3 = read_json(ROOT / "results/error_analysis_v3/validation_summary.json")
    v2_test = read_json(ROOT / "results/v2_512_test.json")
    v4_test = read_json(ROOT / "results/v4_deeplab_test.json")
    st.markdown("<div class='section-header'>Held-out regions</div>", unsafe_allow_html=True)
    table = []
    for region in ("Mekong", "Somalia"):
        for name, data in [("V2 · U-Net", v2_test), ("V4 · DeepLabV3+", v4_test)]:
            # Populate from JSON, or known historical measurements if not mounted.
            fallback_values = {("Mekong", "V2 · U-Net"):(.8751157776800695,.8412972999648075,.9117669970189183),
                               ("Mekong", "V4 · DeepLabV3+"):(.8647636312628523,.8388005075204563,.8923853494742371),
                               ("Somalia", "V2 · U-Net"):(.27111854805161867,.17038907782657964,.6631624979635045),
                               ("Somalia", "V4 · DeepLabV3+"):(.2933881727539927,.20554325479318622,.5123603405846041)}
            metrics = (data or {}).get("by_region", {}).get(region, {})
            dice, precision, recall = fallback_values[(region, name)]
            table.append({"Region": region, "Model": name,
                          "Dice": metrics.get("dice", dice), "Precision": metrics.get("precision", precision), "Recall": metrics.get("recall", recall)})
    df = pd.DataFrame(table)
    st.bar_chart(df.pivot(index="Region", columns="Model", values="Dice"), color=[BLUE, TEAL], height=340)
    with st.expander("Detailed regional results", expanded=True):
        st.dataframe(df.style.format({"Dice": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}"}), hide_index=True, use_container_width=True)
        st.download_button("Export regional metrics CSV", df.to_csv(index=False).encode(), "floodscope_geographic_metrics.csv", "text/csv", use_container_width=True)
    st.markdown('<div class="notice alert">Important: these are pooled-pixel metrics. The test regions have already been inspected across experiments; do not use them to tune thresholds or pick future model configurations. Somalia performance remains limited.</div>', unsafe_allow_html=True)
    section("Validation: India and Spain", "Regional discrepancies were observed before trying a different architecture.")
    regional = pd.DataFrame([
        {"Region": "India", "V2 Dice": .6588, "V3 Dice": .6686},
        {"Region": "Spain", "V2 Dice": .8535, "V3 Dice": .8244},
    ]).set_index("Region")
    st.bar_chart(regional, color=[BLUE, TEAL], height=280)
    for name, obj in [("V2 report", valid_v2), ("V3 report", valid_v3)]:
        if obj:
            with st.expander(name + " · Local JSON"):
                st.json(obj, expanded=False)
    section("Failure modes we documented")
    a, b, c = st.columns(3)
    with a: metric_tile("Small water patches", "Missed", "Thin structures and sparse positive masks", "yellow")
    with b: metric_tile("False alarms", "Observed", "Positive predictions on negative-labeled areas", "sky")
    with c: metric_tile("Domain shift", "Severe", "Nepal inputs saturate the network predictions", "teal")


def nepal_research():
    hero("04 / Domain shift / Nepal", "When a model should not be trusted.",
         "The Nepal UNOSAT case reveals a representation mismatch between SAR-derived PNG imagery and our training inputs.")
    st.markdown('<div class="notice alert"><strong>Exploratory research only.</strong> Nepal has no verified reference masks in this project, and its three-channel representation has not been demonstrated equivalent to Sen1Floods11. No flood area, population impact or hazard conclusion is calculated.</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    with a: metric_tile("VV/VH pairs", "2,073", "PNG mosaics · matching names", "yellow")
    with b: metric_tile("Suspect pixels", "35.79%", "Mean of 80 sampled tiles · heuristic only", "sky")
    with c: metric_tile("False-looking saturation", "97–100%", "Positive predictions on four exploratory tiles", "navy")
    section("Input shift visualized", "These diagnostic images were generated by your actual Nepal analysis, not stock illustrations.")
    images = [
        ("nepal_channel_comparison.png", "Different channel-mean distributions in training and Nepal"),
        ("nepal_tile_12_12.png", "Tile 12_12 · two VV-filter variants"),
        ("nepal_tile_34_46.png", "Tile 34_46 · apparently extensive false-positive saturation"),
    ]
    for filename, caption in images:
        path = ROOT / "app/assets" / filename
        if path.exists():
            display_image(str(path), caption)
    research = read_json(ROOT / "results/nepal_exploratory/report.json")
    if research is None:
        research = read_json(ROOT / "results/nepal_exploratory/report(1).json")
    with st.expander("Recorded exploratory inference measurements"):
        tiles = (research or {}).get("tiles") or [
            {"tile": "tile_12_12.png", "positive_fraction_filtered": .9766273498535156, "mean_probability_filtered": .8933027982711792},
            {"tile": "tile_26_22.png", "positive_fraction_filtered": .9950714111328125, "mean_probability_filtered": .9481073021888733},
            {"tile": "tile_14_28.png", "positive_fraction_filtered": .9972953796386719, "mean_probability_filtered": .9505705833435059},
            {"tile": "tile_34_46.png", "positive_fraction_filtered": .9956512451171875, "mean_probability_filtered": .9516211748123169}]
        st.dataframe(pd.DataFrame([{"Tile": t["tile"], "Predicted-positive fraction": pct(t["positive_fraction_filtered"]),
                                     "Average model output": pct(t["mean_probability_filtered"])} for t in tiles]), hide_index=True, use_container_width=True)
    section("Why we don't publish a flood map", "Matching the number of channels isn't enough when their numerical and physical meaning changes.")
    x, y = st.columns(2)
    with x: metric_tile("Training source", "RGB / 255", "Sen1Floods11 packaged three-channel PNG", "teal")
    with y: metric_tile("Nepal source", "VV + VH", "Reconstructed R = VV, G = VH, B = VV / VH", "sky")
    st.markdown('<div class="notice">Next research milestones: confirm SAR preprocessing from raw data, establish validated no-data handling, obtain independent Nepal reference labels, then assess geographic alignment before computing area or exposure.</div>', unsafe_allow_html=True)


st.set_page_config(page_title="FloodScope AI | Research Studio", page_icon="🌊", layout="wide", initial_sidebar_state="expanded")
css_path = ROOT / "app/dopper_theme.css"
if css_path.exists():
    st.markdown("<style>" + css_path.read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-brand">FloodScope<span style="color:#000f2e"> AI</span></div><div class="sidebar-sub">Research studio · v1.0</div>', unsafe_allow_html=True)
    page = st.radio("Explore", ["Overview", "Model explorer", "Geographic evaluation", "Nepal · domain shift"], label_visibility="collapsed")
    st.divider()
    st.markdown('<span class="pill">Local-first</span>', unsafe_allow_html=True)
    st.caption("Models run on your machine. The project does not send satellite imagery to an external inference API.")
    st.caption("**Research-only** · Not suitable for emergency decisions.")
    st.divider()
    with st.expander("File availability"):
        for name, filepath in [
            ("Flood images", DATA_ZIP),
            ("Split manifest", SPLITS),
            ("U-Net V2", ROOT / MODEL_INFO["V2 · U-Net / ResNet-18"]["checkpoint"]),
            ("DeepLabV3+ V4", ROOT / MODEL_INFO["V4 · DeepLabV3+ / ResNet-18"]["checkpoint"]),
        ]:
            st.write(("✅" if filepath.is_file() else "⚪") + " " + name)

if page == "Overview":
    overview()
elif page == "Model explorer":
    model_explorer()
elif page == "Geographic evaluation":
    geographic_evaluation()
else:
    nepal_research()

st.markdown('<div class="footer">FloodScope AI · Research on satellite-based flood segmentation · Data: Sen1Floods11 & UNOSAT educational case study · Results require context and independent validation.</div>', unsafe_allow_html=True)
