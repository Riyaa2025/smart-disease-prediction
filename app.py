import html
import joblib
from pathlib import Path

import numpy as np
import streamlit as st

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model_bundle.pkl"

# -------------------- STYLING --------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

    :root {
        --ink: #1b2430;
        --muted: #667085;
        --line: #e4e7ec;
        --bg: #f6f7f9;
        --accent: #1d4e89;
        --accent-dark: #163d6d;
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
        color: var(--ink);
    }
    .stApp { background: var(--bg); }

    /* hide streamlit chrome */
    #MainMenu, footer, header [data-testid="stToolbar"],
    .stDeployButton { display: none !important; }
    header[data-testid="stHeader"] { background: transparent; }

    .block-container {
        max-width: 1100px;
        padding-top: 3.5rem;
        padding-bottom: 3rem;
    }

    /* page title */
    .page-title {
        font-size: 1.65rem;
        font-weight: 600;
        letter-spacing: -0.01em;
        margin: 0;
        color: var(--ink);
    }
    .page-sub {
        color: var(--muted);
        font-size: 0.95rem;
        margin: 0.35rem 0 0;
    }
    .title-wrap {
        padding-bottom: 1.1rem;
        border-bottom: 1px solid var(--line);
        margin-bottom: 1.8rem;
    }

    /* section headings */
    .section-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--muted);
        margin-bottom: 0.6rem;
    }

    /* panels */
    .panel {
        background: #fff;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1.1rem 1.25rem;
        min-height: 3.2rem;
    }
    .panel ul { margin: 0; padding-left: 1.1rem; }
    .panel li { margin: 0.25rem 0; font-size: 0.95rem; }
    .panel .empty { color: var(--muted); font-size: 0.92rem; }

    /* result */
    .result {
        background: #fff;
        border: 1px solid var(--line);
        border-left: 4px solid var(--accent);
        border-radius: 8px;
        padding: 1.1rem 1.4rem;
    }
    .result .k { color: var(--muted); font-size: 0.82rem; }
    .result .v { font-size: 1.5rem; font-weight: 600; margin: 0.15rem 0 0; }

    .row { margin: 0.9rem 0 0; }
    .row .top {
        display: flex; justify-content: space-between;
        font-size: 0.92rem; margin-bottom: 0.3rem;
    }
    .row .pct { color: var(--muted); font-variant-numeric: tabular-nums; }
    .bar { height: 6px; background: #e9edf2; border-radius: 3px; overflow: hidden; }
    .bar > span { display: block; height: 100%; background: var(--accent); }

    .note {
        color: var(--muted);
        font-size: 0.85rem;
        margin-top: 1.4rem;
    }

    /* buttons */
    div.stButton > button {
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        font-size: 0.92rem;
        box-shadow: none;
    }
    div.stButton > button[kind="primary"] {
        background: var(--accent);
        border: 1px solid var(--accent);
        color: #fff;
    }
    div.stButton > button[kind="primary"]:hover {
        background: var(--accent-dark);
        border-color: var(--accent-dark);
        color: #fff;
    }
    div.stButton > button[kind="secondary"] {
        background: #fff;
        border: 1px solid #cfd4dc;
        color: var(--ink);
    }
    div.stButton > button[kind="secondary"]:hover {
        border-color: var(--accent);
        color: var(--accent);
    }

    /* multiselect */
    [data-baseweb="select"] > div { border-radius: 6px; background: #fff; }
    [data-baseweb="tag"] {
        background: #e8eef6 !important;
        color: var(--accent) !important;
        border-radius: 4px !important;
    }
    [data-baseweb="tag"] span { color: var(--accent) !important; }

    /* sidebar */
    section[data-testid="stSidebar"] {
        background: #fff;
        border-right: 1px solid var(--line);
    }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        font-size: 1rem; font-weight: 600;
    }
    .meta { margin: 0 0 0.9rem; }
    .meta .k { color: var(--muted); font-size: 0.78rem; }
    .meta .v { font-size: 0.95rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# -------------------- HELPERS --------------------
def pretty(label: str) -> str:
    return str(label).replace("_", " ").strip().title()


@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)


def clear_selection():
    st.session_state["selected_symptoms"] = []


# -------------------- HEADER --------------------
st.markdown("""
<div class="title-wrap">
    <p class="page-title">Disease Prediction System</p>
    <p class="page-sub">
        Symptom-based disease classification using a trained machine learning model.
    </p>
</div>
""", unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.info("Make sure models/model_bundle.pkl exists beside app.py.")
    st.stop()

try:
    bundle = load_bundle()
    model = bundle["model"]
    symptoms = list(bundle["symptoms"])
    classes = list(bundle["classes"])
except Exception as exc:
    st.error(f"Could not load the model bundle: {exc}")
    st.stop()

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.subheader("About")
    st.write(
        "Maps a selected set of symptoms to a disease label using a "
        "model trained on a symptom dataset."
    )
    st.markdown(
        f"""
        <div class="meta"><div class="k">Model</div>
        <div class="v">{html.escape(str(bundle.get("selected_model", type(model).__name__)))}</div></div>
        <div class="meta"><div class="k">Symptoms available</div>
        <div class="v">{len(symptoms)}</div></div>
        <div class="meta"><div class="k">Diseases covered</div>
        <div class="v">{len(classes)}</div></div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption("Educational prototype. Not a medical diagnosis.")

# -------------------- INPUT --------------------
left, right = st.columns([1.4, 1], gap="large")

with left:
    st.markdown('<div class="section-label">Symptoms</div>', unsafe_allow_html=True)

    selected_symptoms = st.multiselect(
        "Select symptoms",
        options=symptoms,
        key="selected_symptoms",
        format_func=pretty,
        placeholder="Search and select symptoms",
        label_visibility="collapsed",
    )

    st.caption(f"{len(selected_symptoms)} selected")

    col_a, col_b, _ = st.columns([1, 1, 1.2])
    with col_a:
        predict_clicked = st.button(
            "Run prediction", type="primary", use_container_width=True
        )
    with col_b:
        st.button("Clear", on_click=clear_selection, use_container_width=True)

with right:
    st.markdown('<div class="section-label">Selected symptoms</div>', unsafe_allow_html=True)
    if selected_symptoms:
        items = "".join(f"<li>{html.escape(pretty(s))}</li>" for s in selected_symptoms)
        st.markdown(f'<div class="panel"><ul>{items}</ul></div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="panel"><span class="empty">No symptoms selected.</span></div>',
            unsafe_allow_html=True,
        )

# -------------------- PREDICTION --------------------
if predict_clicked:
    if not selected_symptoms:
        st.warning("Select at least one symptom to run the prediction.")
    else:
        # Binary feature vector in the exact model order.
        symptom_set = set(selected_symptoms)
        input_vector = np.array(
            [[int(symptom in symptom_set) for symptom in symptoms]],
            dtype=np.uint8,
        )

        try:
            prediction = model.predict(input_vector)[0]
            predicted_label = (
                classes[int(prediction)]
                if isinstance(prediction, (int, np.integer))
                else str(prediction)
            )

            st.markdown("<div style='height:1.8rem'></div>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Result</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="result">'
                '<div class="k">Predicted condition</div>'
                f'<div class="v">{html.escape(pretty(predicted_label))}</div>'
                '</div>',
                unsafe_allow_html=True,
            )

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_vector)[0]
                top_indices = np.argsort(probabilities)[::-1][:3]

                rows = ""
                for index in top_indices:
                    score = min(max(float(probabilities[index]), 0.0), 1.0)
                    rows += (
                        '<div class="row">'
                        '<div class="top">'
                        f'<span>{html.escape(pretty(classes[index]))}</span>'
                        f'<span class="pct">{score:.1%}</span>'
                        '</div>'
                        f'<div class="bar"><span style="width:{score*100:.1f}%"></span></div>'
                        '</div>'
                    )

                st.markdown(
                    '<div class="section-label" style="margin-top:1.6rem">Top 3 model scores</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(f'<div class="panel">{rows}</div>', unsafe_allow_html=True)

            st.markdown(
                '<div class="note">This tool is for educational purposes only. '
                'It cannot confirm or rule out a disease. Consult a qualified '
                'healthcare professional about any symptoms.</div>',
                unsafe_allow_html=True,
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.caption(
                "Check that the saved model and symptom feature order "
                "match the notebook."
            )