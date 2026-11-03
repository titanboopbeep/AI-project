import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Mars Surface Atlas", page_icon="M", layout="wide", initial_sidebar_state="expanded")

FEATURES = [f"n{i}" for i in range(1000)]
LABEL = "category"

# A restrained research-console visual system: warm mineral background, charcoal type,
# and a muted oxidised-red accent. All functionality remains native Streamlit.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root { --ink:#202625; --muted:#727b76; --paper:#f5f4ef; --panel:#fffefa; --line:#e4e4dc; --rust:#a64f3d; --rust-dark:#843b2e; --sage:#65776b; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:var(--paper); color:var(--ink); }
[data-testid="stHeader"] { background:rgba(245,244,239,.92); }
.block-container { max-width:1260px; padding-top:2.2rem; padding-bottom:4rem; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; color:var(--ink) !important; letter-spacing:-.035em; }
h1 { font-size:2.65rem !important; font-weight:600 !important; line-height:1.05 !important; }
h2 { font-size:1.55rem !important; font-weight:600 !important; }
h3 { font-size:1.12rem !important; font-weight:600 !important; }
p, label, .stMarkdown { color:var(--ink); }
.hero { display:flex; justify-content:space-between; align-items:flex-end; gap:24px; padding:8px 0 26px; border-bottom:1px solid var(--line); margin-bottom:28px; }
.eyebrow { color:var(--rust); text-transform:uppercase; letter-spacing:.18em; font-size:.69rem; font-weight:700; margin-bottom:13px; }
.hero-title { font-family:'Space Grotesk',sans-serif; font-size:clamp(2.3rem,5vw,3.7rem); letter-spacing:-.065em; line-height:.98; font-weight:600; color:var(--ink); }
.hero-sub { color:var(--muted); font-size:.98rem; margin-top:13px; max-width:590px; line-height:1.6; }
.hero-mark { border:1px solid #d8d8ce; background:#eeece4; border-radius:50%; width:86px; height:86px; display:flex; align-items:center; justify-content:center; color:var(--rust); font-family:'Space Grotesk'; font-size:1.55rem; flex:0 0 auto; }
.section-kicker { color:var(--rust); text-transform:uppercase; letter-spacing:.15em; font-size:.67rem; font-weight:700; margin:4px 0 8px; }
.panel { background:var(--panel); border:1px solid var(--line); border-radius:15px; padding:20px 22px; }
.panel-label { color:var(--muted); text-transform:uppercase; letter-spacing:.12em; font-size:.66rem; font-weight:700; }
.panel-value { color:var(--ink); font-family:'Space Grotesk'; font-size:1.8rem; font-weight:600; margin-top:5px; }
.panel-note { color:var(--muted); font-size:.82rem; margin-top:3px; }
div[data-testid="stTabs"] > div[role="tablist"] { gap:8px; border-bottom:1px solid var(--line); }
button[data-baseweb="tab"] { background:transparent !important; border:1px solid transparent !important; border-radius:9px 9px 0 0 !important; color:var(--muted) !important; padding:12px 17px !important; font-weight:600 !important; }
button[data-baseweb="tab"][aria-selected="true"] { color:var(--rust-dark) !important; border-bottom:2px solid var(--rust) !important; }
.stButton > button, .stDownloadButton > button { border-radius:9px; border:1px solid var(--rust) !important; background:var(--rust) !important; color:white !important; font-weight:600; padding:.62rem 1.05rem; transition:all .18s ease; }
.stButton > button:hover, .stDownloadButton > button:hover { background:var(--rust-dark) !important; border-color:var(--rust-dark) !important; }
.stFileUploader section { background:#fbfaf6; border:1px dashed #c9c9bd; border-radius:12px; }
.stFileUploader section:hover { border-color:var(--rust); }
[data-testid="stMetric"] { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:14px 16px; }
[data-testid="stMetricLabel"] { color:var(--muted) !important; }
[data-testid="stMetricValue"] { font-family:'Space Grotesk'; color:var(--ink) !important; }
[data-testid="stSidebar"] { background:#eeeee7; border-right:1px solid var(--line); }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { font-size:1.05rem !important; }
.sidebar-brand { font-family:'Space Grotesk'; font-size:1.2rem; font-weight:700; letter-spacing:-.04em; padding:6px 0 18px; border-bottom:1px solid #d9d9d0; margin-bottom:18px; }
.sidebar-step { display:flex; gap:11px; margin:14px 0; align-items:flex-start; }
.step-no { width:24px; height:24px; border:1px solid #c9c9bd; border-radius:50%; display:flex; justify-content:center; align-items:center; color:var(--rust); font-size:.72rem; flex:0 0 auto; }
.step-copy { color:var(--muted); font-size:.82rem; line-height:1.45; }
div[data-testid="stAlert"] { border-radius:10px; }
.stDataFrame { border:1px solid var(--line); border-radius:10px; overflow:hidden; }
hr { border-color:var(--line) !important; }
footer { visibility:hidden; }
@media(max-width:700px) { .hero { align-items:flex-start; } .hero-mark { width:58px; height:58px; font-size:1.1rem; } .hero-title { font-size:2.35rem; } .block-container { padding-left:1rem; padding-right:1rem; } }
</style>
<div class="hero">
  <div>
    <div class="eyebrow">Planetary surface intelligence · Research prototype</div>
    <div class="hero-title">Mars Surface<br>Atlas<span style="color:#a64f3d">.</span></div>
    <div class="hero-sub">A visual classification workspace for distinguishing rocky and smooth terrain in Mars imagery.</div>
  </div>
  <div class="hero-mark">MSA</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-brand">MARS / SURFACE ATLAS</div>', unsafe_allow_html=True)
    st.markdown("**WORKFLOW**")
    for number, title, detail in [
        ("01", "Prepare data", "Upload labelled Orange embeddings."),
        ("02", "Train model", "Evaluate and save the classifier."),
        ("03", "Classify imagery", "Upload JPG or PNG surface images."),
    ]:
        st.markdown(f'<div class="sidebar-step"><div class="step-no">{number}</div><div class="step-copy"><strong style="color:#202625">{title}</strong><br>{detail}</div></div>', unsafe_allow_html=True)
    st.divider()
    st.caption("MODEL STACK")
    st.markdown("SqueezeNet 1.1  ·  Random Forest")
    st.caption("Prototype interface · Not a validated scientific instrument")

@st.cache_resource
def load_squeezenet():
    import torch
    from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights
    weights = SqueezeNet1_1_Weights.DEFAULT
    model = squeezenet1_1(weights=weights)
    model.eval()
    return model, weights.transforms(), torch


def embed_images(files):
    model, preprocess, torch = load_squeezenet()
    vectors, names, errors = [], [], []
    for f in files:
        try:
            f.seek(0)
            img = Image.open(f).convert("RGB")
            tensor = preprocess(img).unsqueeze(0)
            with torch.inference_mode():
                vec = model(tensor).squeeze(0).cpu().numpy().astype(np.float32)
            if vec.shape[0] != 1000:
                raise ValueError(f"Expected 1000 features, got {vec.shape[0]}")
            vectors.append(vec)
            names.append(f.name)
        except Exception as e:
            errors.append(f"{f.name}: {e}")
    return vectors, names, errors


def read_training_csv(file):
    df = pd.read_csv(file)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError("Missing embedding columns: " + ", ".join(missing[:10]))
    if LABEL not in df.columns:
        raise ValueError("Training CSV must contain a 'category' column with rocky/smooth labels.")
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce")
    y = df[LABEL].astype(str).str.strip().str.lower()
    valid = X.notna().all(axis=1) & np.isfinite(X.to_numpy()).all(axis=1) & y.isin(["rocky", "smooth"])
    return X.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True), len(df) - int(valid.sum())

train_tab, predict_tab, guide_tab = st.tabs(["01  ·  Train & evaluate", "02  ·  Image classification", "03  ·  Method & notes"])

with train_tab:
    st.markdown('<div class="section-kicker">Model development</div>', unsafe_allow_html=True)
    st.header("Train the classifier")
    st.write("Import a labelled Orange Image Embedding export. The app uses the 1,000 SqueezeNet feature columns to train a Random Forest model.")
    file = st.file_uploader("Labelled embeddings · CSV", type=["csv"], key="train_csv")
    if file:
        try:
            X, y, skipped = read_training_csv(file)
            a, b, c = st.columns(3)
            a.metric("Usable samples", len(y))
            b.metric("Rocky samples", int((y == "rocky").sum()))
            c.metric("Smooth samples", int((y == "smooth").sum()))
            if skipped:
                st.warning(f"{skipped} rows excluded because of invalid labels or embedding values.")
            if y.nunique() != 2 or y.value_counts().min() < 2:
                st.error("At least two valid examples per class are required.")
            elif st.button("Train Random Forest", type="primary"):
                Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
                with st.spinner("Training and evaluating the classifier…"):
                    evaluation = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1)
                    evaluation.fit(Xtr, ytr)
                    pred = evaluation.predict(Xte)
                    final = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced", n_jobs=-1)
                    final.fit(X, y)
                st.session_state["bundle"] = {"model": final, "feature_columns": FEATURES, "classes": list(final.classes_), "label_column": LABEL}
                st.session_state["metrics"] = {"accuracy": accuracy_score(yte, pred), "report": classification_report(yte, pred, zero_division=0), "matrix": confusion_matrix(yte, pred, labels=["rocky", "smooth"])}
                st.success("Training complete. The downloadable model has been fitted on all valid labelled samples.")
            if "metrics" in st.session_state:
                m = st.session_state["metrics"]
                st.markdown("#### Hold-out evaluation")
                st.metric("Test accuracy", f"{m['accuracy']:.1%}")
                st.caption("Stratified 25% hold-out split · Fixed random seed 42. A single split is an initial check, not proof of generalisation.")
                left, right = st.columns([1, 1])
                with left:
                    st.markdown("**Classification report**")
                    st.code(m["report"], language="text")
                with right:
                    st.markdown("**Confusion matrix**")
                    st.dataframe(pd.DataFrame(m["matrix"], index=["Actual · rocky", "Actual · smooth"], columns=["Predicted · rocky", "Predicted · smooth"]), use_container_width=True)
                buf = io.BytesIO()
                joblib.dump(st.session_state["bundle"], buf)
                st.download_button("Download trained model", buf.getvalue(), "mars_orange_random_forest.joblib", "application/octet-stream")
        except Exception as e:
            st.error(f"Training data error: {e}")

with predict_tab:
    st.markdown('<div class="section-kicker">Image analysis</div>', unsafe_allow_html=True)
    st.header("Classify Mars imagery")
    st.write("Load a trained classifier, add one or more surface images, and run the prediction pipeline.")
    model_file = st.file_uploader("Trained model · JOBLIB", type=["joblib"], key="model_file")
    bundle = st.session_state.get("bundle")
    if model_file:
        try:
            model_file.seek(0)
            bundle = joblib.load(model_file)
            st.session_state["uploaded_bundle"] = bundle
        except Exception as e:
            st.error(f"Could not load model: {e}")
            bundle = None
    elif st.session_state.get("uploaded_bundle") is not None:
        bundle = st.session_state["uploaded_bundle"]
    if bundle is None:
        st.info("Train a classifier in the first tab or upload an existing trusted .joblib model.")
    else:
        st.markdown('<div class="panel"><div class="panel-label">Classifier status</div><div class="panel-value" style="font-size:1.25rem">Model loaded</div><div class="panel-note">Ready to process JPG, JPEG and PNG images.</div></div>', unsafe_allow_html=True)
        images = st.file_uploader("Mars surface images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="images")
        if images:
            st.caption(f"{len(images)} image(s) selected · Preview limited to the first four files.")
            cols = st.columns(min(4, len(images)))
            for i, f in enumerate(images[:4]):
                with cols[i]:
                    f.seek(0)
                    st.image(f, caption=f.name, use_container_width=True)
                    f.seek(0)
            if st.button("Run classification", type="primary"):
                try:
                    with st.spinner("Loading SqueezeNet and extracting image features…"):
                        vectors, names, errors = embed_images(images)
                    for err in errors:
                        st.warning(err)
                    if not vectors:
                        st.error("No images could be processed.")
                    else:
                        X = pd.DataFrame(vectors, columns=bundle["feature_columns"])
                        predictions = bundle["model"].predict(X)
                        probs = bundle["model"].predict_proba(X)
                        results = pd.DataFrame({"Image": names, "Prediction": [str(p).upper() for p in predictions], "Model score (%)": np.round(probs.max(axis=1) * 100, 2)})
                        st.markdown("#### Classification summary")
                        a, b, c = st.columns(3)
                        a.metric("Processed", len(results))
                        b.metric("Rocky", int((results.Prediction == "ROCKY").sum()))
                        c.metric("Smooth", int((results.Prediction == "SMOOTH").sum()))
                        st.dataframe(results, use_container_width=True, hide_index=True)
                        st.download_button("Export results · CSV", results.to_csv(index=False).encode(), "mars_image_predictions.csv", "text/csv")
                        st.caption("Model score is the classifier's probability output. It is not a calibrated measure or guarantee of correctness.")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

with guide_tab:
    st.markdown('<div class="section-kicker">Technical notes</div>', unsafe_allow_html=True)
    st.header("Model pipeline & limitations")
    st.markdown("""
    **Feature extraction**  
    This application currently uses Torchvision SqueezeNet 1.1 with ImageNet-pretrained weights and the standard Torchvision inference transform. Its 1,000-value output is passed to the Random Forest classifier.

    **Training data**  
    The labelled training CSV is expected to contain `n0` through `n999` and a `category` column containing `rocky` or `smooth`. The training embeddings are currently prepared using Orange Data Mining.

    **Compatibility requirement**  
    Orange's Image Embedding widget and this Torchvision implementation must use equivalent model weights, preprocessing and output representation. Matching feature count alone does not establish compatibility. Compare the feature vectors for identical images before relying on direct-image predictions.

    **Evaluation limitation**  
    A single train/test split can overestimate performance, especially with a small dataset or visually similar images. Use a separate test set collected from different images or rover locations before making scientific claims.
    """)
    st.warning("This is a research prototype for surface-image classification, not a validated planetary-science instrument.")

st.markdown("<div style='border-top:1px solid #e4e4dc;margin-top:42px;padding-top:16px;color:#8a918b;font-size:.75rem;display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap'><span>MARS SURFACE ATLAS · CLASSIFICATION WORKSPACE</span><span>Research prototype</span></div>", unsafe_allow_html=True)
