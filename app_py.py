import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="COVID-19 X-Ray Detection",
    page_icon="🫁",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

/* Main container */
.block-container {
    max-width: 850px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Header */
.hero-container {
    padding: 35px 25px;
    border-radius: 20px;
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    text-align: center;
    margin-bottom: 25px;
}

.hero-title {
    color: white;
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 8px;
}

.hero-subtitle {
    color: #e5e7eb;
    font-size: 17px;
    margin: 0;
}

/* Section headings */
.section-heading {
    font-size: 23px;
    font-weight: 650;
    margin-top: 25px;
    margin-bottom: 15px;
}

/* Result card */
.result-card {
    padding: 25px;
    border-radius: 18px;
    border: 1px solid rgba(128, 128, 128, 0.25);
    text-align: center;
    margin-top: 20px;
    margin-bottom: 20px;
}

.result-label {
    font-size: 15px;
    opacity: 0.7;
    margin-bottom: 5px;
}

.result-value {
    font-size: 32px;
    font-weight: 750;
}

/* Probability cards */
.probability-card {
    padding: 18px;
    border-radius: 14px;
    border: 1px solid rgba(128, 128, 128, 0.25);
    text-align: center;
    margin-bottom: 10px;
}

.probability-title {
    font-size: 15px;
    opacity: 0.7;
}

.probability-value {
    font-size: 25px;
    font-weight: 700;
}

/* Model information */
.model-card {
    padding: 22px;
    border-radius: 16px;
    border: 1px solid rgba(128, 128, 128, 0.25);
    margin-top: 15px;
}

/* Button */
.stButton > button {
    border-radius: 12px;
    height: 48px;
    font-size: 17px;
    font-weight: 600;
}

/* Uploaded image */
[data-testid="stImage"] img {
    border-radius: 15px;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_covid_model():
    return tf.keras.models.load_model("model.keras")


try:
    model = load_covid_model()

except Exception as error:
    st.error("Unable to load the trained model.")
    st.exception(error)
    st.stop()


# ==================================================
# HEADER
# ==================================================

st.markdown("""
<div class="hero-container">

    <div class="hero-title">
        🫁 COVID-19 X-Ray Detection
    </div>

    <p class="hero-subtitle">
        Deep Learning based Chest X-Ray Image Classification
    </p>

</div>
""", unsafe_allow_html=True)


# ==================================================
# UPLOAD SECTION
# ==================================================

st.markdown(
    '<div class="section-heading">📤 Upload Chest X-Ray</div>',
    unsafe_allow_html=True
)

st.write(
    "Upload a chest X-ray image to classify it as **COVID-19** or **Normal**."
)

uploaded_file = st.file_uploader(
    "Select an X-Ray Image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)


# ==================================================
# PREPROCESS IMAGE
# ==================================================

def preprocess_image(uploaded_image):

    image = Image.open(uploaded_image).convert("RGB")

    resized_image = image.resize((299, 299))

    image_array = np.array(
        resized_image,
        dtype=np.float32
    )

    image_array = image_array / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image, image_array


# ==================================================
# IMAGE AND PREDICTION
# ==================================================

if uploaded_file is not None:

    try:

        image, processed_image = preprocess_image(uploaded_file)


        # ------------------------------------------
        # IMAGE PREVIEW
        # ------------------------------------------

        st.markdown(
            '<div class="section-heading">🩻 X-Ray Preview</div>',
            unsafe_allow_html=True
        )


        left_space, image_column, right_space = st.columns(
            [1, 2, 1]
        )


        with image_column:

            st.image(
                image,
                use_container_width=True
            )


        # ------------------------------------------
        # ANALYZE BUTTON
        # ------------------------------------------

        analyze_button = st.button(
            "🔍 Analyze X-Ray",
            use_container_width=True,
            type="primary"
        )


        if analyze_button:

            with st.spinner("Analyzing chest X-ray..."):

                prediction = model.predict(
                    processed_image,
                    verbose=0
                )

                probability = float(
                    prediction[0][0]
                )


            # --------------------------------------
            # CLASS MAPPING
            #
            # covid  = 0
            # normal = 1
            # --------------------------------------

            covid_probability = (
                1 - probability
            ) * 100

            normal_probability = (
                probability
            ) * 100


            if probability >= 0.5:

                result = "NORMAL"

                confidence = normal_probability

                result_icon = "✅"

            else:

                result = "COVID-19"

                confidence = covid_probability

                result_icon = "⚠️"


            # --------------------------------------
            # RESULT
            # --------------------------------------

            st.markdown(
                '<div class="section-heading">📊 Analysis Result</div>',
                unsafe_allow_html=True
            )


            st.markdown(
                f"""
                <div class="result-card">

                    <div class="result-label">
                        PREDICTED CLASS
                    </div>

                    <div class="result-value">
                        {result_icon} {result}
                    </div>

                    <div style="
                        margin-top:10px;
                        font-size:17px;
                    ">
                        Prediction Score:
                        <b>{confidence:.2f}%</b>
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            # --------------------------------------
            # CONFIDENCE PROGRESS
            # --------------------------------------

            st.write("**Prediction Score**")

            st.progress(
                min(
                    max(confidence / 100, 0.0),
                    1.0
                )
            )


            # --------------------------------------
            # PROBABILITY CARDS
            # --------------------------------------

            st.markdown(
                '<div class="section-heading">📈 Class Probabilities</div>',
                unsafe_allow_html=True
            )


            col1, col2 = st.columns(2)


            with col1:

                st.markdown(
                    f"""
                    <div class="probability-card">

                        <div class="probability-title">
                            COVID-19
                        </div>

                        <div class="probability-value">
                            {covid_probability:.2f}%
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            with col2:

                st.markdown(
                    f"""
                    <div class="probability-card">

                        <div class="probability-title">
                            NORMAL
                        </div>

                        <div class="probability-value">
                            {normal_probability:.2f}%
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


    except Exception as error:

        st.error(
            "Unable to process the uploaded image."
        )

        st.exception(error)


# ==================================================
# MODEL INFORMATION
# ==================================================

st.divider()

st.markdown(
    '<div class="section-heading">🤖 Model Information</div>',
    unsafe_allow_html=True
)


st.markdown("""
<div class="model-card">

<b>Architecture:</b> Convolutional Neural Network (CNN)

<br><br>

<b>Input Size:</b> 299 × 299 × 3

<br><br>

<b>Classification:</b> Binary Classification

<br><br>

<b>Classes:</b> COVID-19 and Normal

<br><br>

<b>Framework:</b> TensorFlow / Keras

<br><br>

<b>Deployment:</b> Streamlit Community Cloud

</div>
""", unsafe_allow_html=True)
