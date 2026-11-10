"""ACIT4620 Group 1 | Frozen EfficientNet-B0 + 16-rule ANFIS demo.

Run locally with: python -m streamlit run app.py
No training, metrics recomputation, file-system image scans, or test-set access.
Only an image explicitly uploaded to this UI is decoded for inference.
"""
from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
from PIL import Image
import streamlit as st

from src.inference import (
    CLASS_DESCRIPTIONS,
    CLASSES,
    load_bundle,
    normalized_rgb_from_tensor,
    prepare_image,
    validate_assets,
)
from src.chart_utils import contribution_plot_rows
from src.app_utils import (
    export_json,
    image_digest,
    prediction_table_rows,
    read_demo_image,
    rule_table_rows,
)

APP_ROOT = Path(__file__).resolve().parent
ASSET_ROOT = APP_ROOT / "model_assets"

st.set_page_config(
    page_title="CNN–ANFIS | Skin Lesion Research",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { max-width: 1180px; padding-top: 1.9rem; }
    div[data-testid="stMetricValue"] { font-size: 1.55rem; }
    .notice {border-left: 4px solid #0d9488; padding: 0.65rem 0.9rem;
             background: rgba(13, 148, 136, 0.08); border-radius: 6px;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading original frozen CNN and ANFIS models…")
def get_model_bundle(asset_path: str):
    """Shared, read-only loaded model. Source file hashes rechecked per analysis."""
    return load_bundle(asset_path, device="cpu")

with st.sidebar:
    st.title("🔬 ACIT4620")
    st.caption("Group 1 · Hybrid AI")
    st.divider()
    st.subheader("Model")
    st.write("EfficientNet-B0 + 16-rule first-order Sugeno ANFIS")
    st.caption("Original checkpoint: ANFIS R16 · seed 52")
    st.caption("Frozen 1,280-D features → train-only scaling → 8-D projection → 7 outputs")
    st.divider()
    st.subheader("Display")
    compare_cnn = st.checkbox("Show CNN-only comparison", value=True)
    show_cam = st.checkbox("Show ANFIS Grad-CAM", value=True)
    show_cnn_cam = st.checkbox("Also show CNN-only Grad-CAM", value=False)
    show_fuzzy = st.checkbox("Show fuzzy-rule breakdown", value=True)
    st.divider()
    st.caption("Validation-only ANFIS macro-F1: 0.5676. Locked-test results pending group sign-off.")
    st.caption("No image is saved to disk by this interface. Uploaded images are processed in memory.")

st.title("Skin lesion image analysis")
st.caption("Frozen CNN–ANFIS · Seven-class HAM10000 research demonstration")
st.markdown('<div class="notice"><strong>Research use only.</strong> This is not a medical device, a diagnosis, or a melanoma screening tool. Do not use predictions for clinical decisions. Use only authorized, non-identifying validation/demo images; do not upload sealed test images before protocol sign-off.</div>', unsafe_allow_html=True)
st.write("")

missing_assets = [name for name in ("e2a50_best_efficientnet_b0.pth", "anfis_r16_seed52.pth", "train_feature_scaler.npz")
                  if not (ASSET_ROOT / name).is_file()]
if missing_assets:
    st.error("Missing original model assets: " + ", ".join(missing_assets))
    st.code("powershell -ExecutionPolicy Bypass -File .\\setup_model_assets.ps1 -SourceDir '<Notebook11>\\data\\model_assets'", language="powershell")
    st.stop()

upload = st.file_uploader("Upload a dermoscopic research image", type=["jpg", "jpeg", "png"],
                          help="JPG/PNG, up to 12 MiB. Use known validation or authorized demo images only.")
if upload is None:
    st.info("Choose a dermoscopic image to begin. Predictions are computed only when you select Analyze image.")
    st.stop()

raw_bytes = upload.getvalue()
try:
    pil_image = read_demo_image(raw_bytes)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

sha = image_digest(raw_bytes)
if st.session_state.get("current_image_sha") != sha:
    st.session_state["current_image_sha"] = sha
    st.session_state.pop("current_prediction", None)
    # Keep uploaded pixels in Streamlit's uploader only, not in custom session state.

left, right = st.columns([1, 1.25], gap="large")
with left:
    st.subheader("Uploaded image")
    st.image(pil_image, caption=f"Original uploaded image · {pil_image.width}×{pil_image.height} pixels",
             use_container_width=True)
    analyze = st.button("Analyze image", type="primary", use_container_width=True)
    st.caption("The model uses the training-time resize (short edge 256), 224×224 center crop, and ImageNet normalization.")

if analyze:
    try:
        with st.spinner("Running the verified CNN–ANFIS pipeline…"):
            # Check original bytes even if the model bundle is already cached.
            verified_hashes = validate_assets(ASSET_ROOT)
            model = get_model_bundle(str(ASSET_ROOT))
            prepared = prepare_image(pil_image)
            prediction = model.predict_tensor(prepared)
            st.session_state["current_prediction"] = prediction
            st.session_state["current_asset_hashes"] = verified_hashes
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
        st.error(f"Prediction stopped: {exc}")
        st.stop()

pred = st.session_state.get("current_prediction")
with right:
    st.subheader("CNN–ANFIS result")
    if pred is None:
        st.info("Select **Analyze image** to generate seven-class probabilities and explanations.")
    else:
        a, b = st.columns([1.4, 1])
        with a:
            st.metric("Predicted label", pred["predicted_class_description"])
        with b:
            st.metric("Top softmax score", f"{pred['predicted_probability']:.1%}")
        st.caption("Softmax outputs are uncalibrated model scores, **not** melanoma/cancer probabilities or clinical risk estimates.")
        if compare_cnn:
            cnn_code = pred["cnn_predicted_class"]
            st.write(f"CNN-only prediction: **{CLASS_DESCRIPTIONS[cnn_code]}** (`{cnn_code}`)")
            if cnn_code != pred["predicted_class"]:
                st.warning("The CNN-only and CNN–ANFIS predictions differ. Neither prediction is medically validated.")

if pred is None:
    st.stop()

st.divider()
st.subheader("Seven-class model scores")
frame = pd.DataFrame(prediction_table_rows(pred, CLASS_DESCRIPTIONS))
score_columns = ["CNN–ANFIS"] + (["CNN only"] if compare_cnn else [])

# IMPORTANT: st.bar_chart stacks multiple series by default. That produces an
# impossible ~188% bar for this example (99.31% ANFIS + 89.12% CNN).
# Display the two separate, NON-STACKED scores side by side, with a fixed 0–100% axis.
long_scores = frame.melt(
    id_vars=["Class"], value_vars=score_columns,
    var_name="Model", value_name="Score_percent",
)
score_plot = (
    alt.Chart(long_scores)
    .mark_bar()
    .encode(
        x=alt.X(
            "Score_percent:Q",
            title="Model softmax score (%)",
            scale=alt.Scale(domain=[0, 100], nice=False),
            axis=alt.Axis(values=[0, 20, 40, 60, 80, 100]),
        ),
        y=alt.Y("Class:N", title=None, sort=frame["Class"].tolist()),
        yOffset=alt.YOffset("Model:N"),
        color=alt.Color("Model:N", title="Model"),
        tooltip=[
            alt.Tooltip("Class:N", title="Class"),
            alt.Tooltip("Model:N", title="Model"),
            alt.Tooltip("Score_percent:Q", title="Score (%)", format=".4f"),
        ],
    )
    .properties(height=340)
)
st.altair_chart(score_plot, use_container_width=True)
st.caption("Scores for each model are plotted separately; they are not added together. Scores are uncalibrated.")

with st.expander("View exact scores in a table", expanded=False):
    visible = ["Class", "Code", "CNN–ANFIS"] + (["CNN only"] if compare_cnn else [])
    table_scores = frame[visible].copy()
    for column in score_columns:
        table_scores[column] = table_scores[column].map(lambda value: f"{value:.4f}%")
    # st.table renders a static HTML table instead of st.dataframe's browser
    # canvas/grid, which can appear as solid black on certain configurations.
    st.table(table_scores.set_index("Class"))

if show_cam or show_cnn_cam:
    st.divider()
    st.subheader("Grad-CAM attribution")
    st.caption("Color indicates which features influence a selected model logit in the center-cropped image. This is **not** lesion segmentation or causal/clinical evidence.")
    try:
        model = get_model_bundle(str(ASSET_ROOT))
        # Image tensor is recreated in RAM; no raw image is persisted to disk.
        x = prepare_image(pil_image)
        view_np = normalized_rgb_from_tensor(x)
        view_image = Image.fromarray((view_np * 255).round().astype("uint8"))
        columns = st.columns(1 + int(show_cam) + int(show_cnn_cam))
        with columns[0]:
            st.image(view_image, caption="Model-view image · 224×224 crop", use_container_width=True)
        current = 1
        if show_cam:
            with st.spinner("Computing ANFIS Grad-CAM…"):
                heat = model.gradcam(x, pred["predicted_class_index"], head="anfis")
                overlay = model.make_overlay(x, heat)
            with columns[current]:
                st.image(overlay, caption=f"CNN–ANFIS CAM · {pred['predicted_class']}", use_container_width=True)
            current += 1
        if show_cnn_cam:
            with st.spinner("Computing CNN-only Grad-CAM…"):
                cnn_class_index = CLASSES.index(pred["cnn_predicted_class"])
                heat_cnn = model.gradcam(x, cnn_class_index, head="cnn")
                overlay_cnn = model.make_overlay(x, heat_cnn)
            with columns[current]:
                st.image(overlay_cnn, caption=f"CNN-only CAM · {pred['cnn_predicted_class']}", use_container_width=True)
    except (ValueError, RuntimeError, OSError) as exc:
        st.warning(f"Could not display Grad-CAM: {exc}")

if show_fuzzy:
    st.divider()
    st.subheader("Inside the 16 fuzzy rules")
    st.caption("Gaussian premise firing strengths are normalized to sum to 1. Signed contributions sum to the predicted class **logit**, not to a probability.")
    fuzzy = pred["fuzzy"]
    k1, k2, k3 = st.columns(3)
    k1.metric("Number of rules", "16")
    k2.metric("Highest firing rule", f"Rule {fuzzy['highest_firing_rule_index_1_based']}")
    k3.metric("Largest |logit contribution|", f"Rule {fuzzy['highest_absolute_logit_contribution_rule_index_1_based']}")
    rules = pd.DataFrame(rule_table_rows(pred)).set_index("Rule")
    lcol, rcol = st.columns(2, gap="large")
    with lcol:
        st.markdown("**Rule firing strength**")
        st.bar_chart(rules[["Firing strength"]], height=330)
    with rcol:
        st.markdown("**Signed contribution to predicted logit**")
        # st.bar_chart can round every numeric tick to "0" in some Streamlit
        # versions. Draw a true signed, zero-anchored axis using Altair.
        # Rule contributions are logits, never probabilities.
        signed_rows = contribution_plot_rows(rule_table_rows(pred))
        signed_frame = pd.DataFrame(signed_rows)
        ordered_rules = [f"Rule {i}" for i in range(1, 17)]
        signed_bars = (
            alt.Chart(signed_frame)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Contribution:Q",
                    title="Signed contribution to predicted-class logit",
                    scale=alt.Scale(zero=True),
                    axis=alt.Axis(format=".2f", tickCount=6, grid=True),
                ),
                y=alt.Y("Rule:N", title=None, sort=ordered_rules),
                color=alt.Color("Direction:N", title="Contribution sign"),
                tooltip=[
                    alt.Tooltip("Rule:N"),
                    alt.Tooltip("Contribution:Q", format="+.8g"),
                ],
            )
            .properties(height=340)
        )
        st.altair_chart(signed_bars, use_container_width=True)
        st.caption("Zero-anchored signed logit contributions; negative values remain negative.")
    # A dominant rule is an image-specific observation, not proof of globally
    # collapsed/under-utilized rules. Display it transparently for research.
    largest_firing = max(fuzzy["normalized_rule_firing"])
    if largest_firing > 0.99:
        st.info(
            f"For this image, Rule {fuzzy['highest_firing_rule_index_1_based']} "
            f"accounts for {largest_firing:.5%} of normalized firing strength. "
            "Other rules contribute very little to this image's prediction. "
            "This alone does not establish how frequently each rule is used across the dataset."
        )
    with st.expander("All rule values and projected features", expanded=False):
        visible_rules = rules.reset_index().copy()
        visible_rules["Firing strength"] = visible_rules["Firing strength"].map(
            lambda value: f"{value:.7g}"
        )
        visible_rules["Predicted-logit contribution"] = visible_rules[
            "Predicted-logit contribution"
        ].map(lambda value: f"{value:+.7g}")
        st.table(visible_rules.set_index("Rule"))
        st.write("Eight learned projected features:")
        st.code(", ".join(f"{float(v):+.4f}" for v in fuzzy["latent_projection"]))
        st.caption(f"Sum of signed rule contributions (predicted logit): {fuzzy['predicted_logit_reconstruction']:.6f}")

st.divider()
st.subheader("Reproducibility")
st.caption("Download model scores and frozen checkpoint fingerprints. The original uploaded image is not included.")
st.download_button(
    "Download prediction JSON",
    data=export_json(pred, sha, st.session_state.get("current_asset_hashes", {})),
    file_name=f"ACIT4620_prediction_{sha[:12]}.json",
    mime="application/json",
)
st.caption("Image inputs and predictions from this demonstration are not part of the locked-test benchmark. Model selection remains frozen; the unverified artifact-annotation subgroup analysis is not claimed here.")
