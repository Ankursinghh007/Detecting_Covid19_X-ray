import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="COVID-19 X-Ray Detection",
    page_icon="🫁",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.block-container {
    max-width: 900px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(
            circle at top left,
            rgba(59, 130, 246, 0.08),
            transparent 30%
        ),
        radial-gradient(
            circle at top right,
            rgba(124, 58, 237, 0.08),
            transparent 30%
        );
}


/* HERO HEADER */

.hero-container {
    background: linear-gradient(
        135deg,
        #2563eb 0%,
        #4f46e5 50%,
        #7c3aed 100%
    );

    border-radius: 22px;
    padding: 42px 25px;
    text-align: center;
    margin-bottom: 35px;

    box-shadow:
        0 12px 35px rgba(79, 70, 229, 0.25);
}


.hero-icon {
    font-size: 50px;
    line-height: 1;
    margin-bottom: 12px;
}


.hero-title {
    color: white;
    font-size: 40px;
    font-weight: 800;
    line-height: 1.2;
    margin-bottom: 10px;
}


.hero-subtitle {
    color: rgba(255, 255, 255, 0.88);
    font-size: 17px;
    margin: 0;
}


/* SECTION */

.section-title {
    font-size: 24px;
    font-weight: 700;
    margin-top: 28px;
    margin-bottom: 7px;
}


.section-description {
    font-size: 16px;
    opacity: 0.75;
    margin-bottom: 18px;
}


/* UPLOADER */

[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed rgba(99, 102, 241, 0.60);
    border-radius: 16px;
    padding: 18px;
    transition: 0.2s ease;
}


[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(99, 102, 241, 1);
    background: rgba(99, 102, 241, 0.06);
}


/* IMAGE */

[data-testid="stImage"] img {
    border-radius: 16px;
    border: 1px solid rgba(128, 128, 128, 0.20);

    box-shadow:
        0 8px 25px rgba(0, 0, 0, 0.10);
}


/* BUTTON */

.stButton > button {
    width: 100%;
    height: 52px;

    border-radius: 13px;

    font-size: 17px;
    font-weight: 700;

    transition:
        transform 0.2s ease,
        box-shadow 0.2s ease;
}


.stButton > button:hover {
    transform: translateY(-2px);

    box-shadow:
        0 8px 22px rgba(79, 70, 229, 0.28);
}


/* RESULT */

.result-card {
    border-radius: 20px;

    border:
        1px solid rgba(128, 128, 128, 0.22);

    padding: 30px 20px;

    text-align: center;

    margin-top: 15px;
    margin-bottom: 20px;

    box-shadow:
        0 8px 25px rgba(0, 0, 0, 0.08);
}


.result-label {
    font-size: 13px;
    letter-spacing: 1.5px;
    opacity: 0.65;
    margin-bottom: 8px;
}


.result-class {
    font-size: 34px;
    font-weight: 800;
    margin-bottom: 10px;
}


.result-score {
    font-size: 17px;
    opacity: 0.85;
}


/* PROBABILITY CARDS */

.probability-card {
    border-radius: 16px;

    border:
        1px solid rgba(128, 128, 128, 0.22);

    padding: 22px 12px;

    text-align: center;

    min-height: 115px;

    box-shadow:
        0 5px 18px rgba(0, 0, 0, 0.05);
}


.probability-label {
    font-size: 15px;
    opacity: 0.70;
    margin-bottom: 8px;
}


.probability-value {
    font-size: 29px;
    font-weight: 800;
}


/* MODEL CARD */

.model-card {
    border-radius: 18px;

    border:
        1px solid rgba(128, 128, 128, 0.22);

    padding: 25px;

    margin-top: 15px;

    box-shadow:
        0 5px 18px rgba(0, 0, 0, 0.05);
}


.model-row {
    display: flex;
    justify-content: space-between;

    gap: 20px;

    padding: 13px 0;

    border-bottom:
        1px solid rgba(128, 128, 128, 0.15);
}


.model-row:last-child {
    border-bottom: none;
}


.model-label {
    font-weight: 700;
}


.model-value {
    opacity: 0.80;
    text-align: right;
}


/* FOOTER */

.footer {
    text-align: center;

    opacity: 0.55;

    font-size: 13px;

    margin-top: 35px;

    padding-top: 20px;

    border-top:
        1px solid rgba(128, 128, 128, 0.15);
}


/* MOBILE */

@media (max-width: 600px) {

    .block-container {
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }


    .hero-container {
        padding: 32px 15px;
        border-radius: 18px;
    }


    .hero-title {
        font-size: 29px;
    }


    .hero-subtitle {
        font-size: 15px;
    }


    .result-class {
        font-size: 28px;
    }


    .model-row {
        flex-direction: column;
        gap: 5px;
    }


    .model-value {
        text-align: left;
    }
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model.keras"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_covid_model():

    if not MODEL_PATH.exists():

        available_files = [
            file.name
            for file in BASE_DIR.iterdir()
        ]

        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            f"Available files: {available_files}"
        )

    loaded_model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )

    return loaded_model


try:

    model = load_covid_model()


except Exception as error:

    st.error("Unable to load the trained model.")

    st.exception(error)

    st.stop()


# ============================================================
# HERO HEADER
# ============================================================

hero_html = """
<div class="hero-container">
<div class="hero-icon">🫁</div>
<div class="hero-title">COVID-19 X-Ray Detection</div>
<p class="hero-subtitle">Deep Learning Based Chest X-Ray Image Classification</p>
</div>
"""

st.markdown(
    hero_html,
    unsafe_allow_html=True
)


# ============================================================
# UPLOAD SECTION
# ============================================================

upload_html = """
<div class="section-title">📤 Upload Chest X-Ray</div>
<div class="section-description">Select a chest X-ray image to classify it as COVID-19 or Normal.</div>
"""

st.markdown(
    upload_html,
    unsafe_allow_html=True
)


uploaded_file = st.file_uploader(

    "Upload Chest X-Ray Image",

    type=[
        "jpg",
        "jpeg",
        "png"
    ],

    label_visibility="collapsed"
)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(uploaded_image):

    original_image = Image.open(
        uploaded_image
    ).convert("RGB")


    resized_image = original_image.resize(
        (299, 299)
    )


    image_array = np.asarray(
        resized_image,
        dtype=np.float32
    )


    image_array = image_array / 255.0


    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    return original_image, image_array


# ============================================================
# IMAGE UPLOAD AND PREDICTION
# ============================================================

if uploaded_file is not None:

    try:

        original_image, processed_image = preprocess_image(
            uploaded_file
        )


        # ====================================================
        # PREVIEW
        # ====================================================

        preview_html = """
<div class="section-title">🩻 X-Ray Preview</div>
<div class="section-description">Review the uploaded chest X-ray image before running the model.</div>
"""

        st.markdown(
            preview_html,
            unsafe_allow_html=True
        )


        left_space, image_column, right_space = st.columns(
            [1, 2.2, 1]
        )


        with image_column:

            st.image(
                original_image,
                use_container_width=True
            )


        # ====================================================
        # BUTTON
        # ====================================================

        analyze_button = st.button(

            "🔍 Analyze Chest X-Ray",

            type="primary",

            use_container_width=True
        )


        # ====================================================
        # PREDICTION
        # ====================================================

        if analyze_button:

            with st.spinner(
                "Analyzing chest X-ray..."
            ):

                raw_prediction = model.predict(
                    processed_image,
                    verbose=0
                )


            # The trained model has one sigmoid output.

            probability = float(
                np.squeeze(raw_prediction)
            )


            # Prevent invalid percentage values.

            probability = float(
                np.clip(
                    probability,
                    0.0,
                    1.0
                )
            )


            # =================================================
            # CLASS MAPPING
            #
            # Training classes:
            #
            # covid  = 0
            # normal = 1
            #
            # Therefore:
            #
            # output < 0.5  => COVID-19
            # output >= 0.5 => NORMAL
            # =================================================

            covid_probability = (
                1.0 - probability
            ) * 100.0


            normal_probability = (
                probability
            ) * 100.0


            if probability >= 0.5:

                predicted_class = "NORMAL"

                prediction_score = normal_probability

                result_icon = "✅"


            else:

                predicted_class = "COVID-19"

                prediction_score = covid_probability

                result_icon = "⚠️"


            # =================================================
            # RESULT
            # =================================================

            result_heading_html = """
<div class="section-title">📊 Analysis Result</div>
<div class="section-description">Prediction generated by the trained CNN model.</div>
"""

            st.markdown(
                result_heading_html,
                unsafe_allow_html=True
            )


            result_html = f"""
<div class="result-card">
<div class="result-label">PREDICTED CLASS</div>
<div class="result-class">{result_icon} {predicted_class}</div>
<div class="result-score">Prediction Score: <b>{prediction_score:.2f}%</b></div>
</div>
"""

            st.markdown(
                result_html,
                unsafe_allow_html=True
            )


            # =================================================
            # SCORE BAR
            # =================================================

            st.write("**Prediction Score**")

            st.progress(
                prediction_score / 100.0
            )


            # =================================================
            # CLASS PROBABILITIES
            # =================================================

            probability_heading_html = """
<div class="section-title">📈 Class Probabilities</div>
"""

            st.markdown(
                probability_heading_html,
                unsafe_allow_html=True
            )


            covid_column, normal_column = st.columns(2)


            with covid_column:

                covid_html = f"""
<div class="probability-card">
<div class="probability-label">COVID-19 Probability</div>
<div class="probability-value">{covid_probability:.2f}%</div>
</div>
"""

                st.markdown(
                    covid_html,
                    unsafe_allow_html=True
                )


            with normal_column:

                normal_html = f"""
<div class="probability-card">
<div class="probability-label">Normal Probability</div>
<div class="probability-value">{normal_probability:.2f}%</div>
</div>
"""

                st.markdown(
                    normal_html,
                    unsafe_allow_html=True
                )


    except Exception as error:

        st.error(
            "Unable to process the uploaded image."
        )

        st.exception(error)


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown(
    "<br>",
    unsafe_allow_html=True
)

st.divider()


model_heading_html = """
<div class="section-title">🤖 Model Information</div>
<div class="section-description">Technical details of the deployed image classification model.</div>
"""

st.markdown(
    model_heading_html,
    unsafe_allow_html=True
)


model_information_html = """
<div class="model-card">
<div class="model-row">
<span class="model-label">Architecture</span>
<span class="model-value">Convolutional Neural Network (CNN)</span>
</div>
<div class="model-row">
<span class="model-label">Input Image Size</span>
<span class="model-value">299 × 299 × 3</span>
</div>
<div class="model-row">
<span class="model-label">Classification</span>
<span class="model-value">Binary Classification</span>
</div>
<div class="model-row">
<span class="model-label">Classes</span>
<span class="model-value">COVID-19 / Normal</span>
</div>
<div class="model-row">
<span class="model-label">Framework</span>
<span class="model-value">TensorFlow / Keras</span>
</div>
<div class="model-row">
<span class="model-label">Deployment</span>
<span class="model-value">Streamlit Community Cloud</span>
</div>
</div>
"""

st.markdown(
    model_information_html,
    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================

footer_html = """
<div class="footer">
CNN-Based Chest X-Ray Image Classification Project
</div>
"""

st.markdown(
    footer_html,
    unsafe_allow_html=True
)
