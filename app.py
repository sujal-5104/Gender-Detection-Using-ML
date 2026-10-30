
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# --------------------------------------------------
# 1. Application Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Gender Detection ML Demo",
    page_icon="🧠",
    layout="centered"
)

IMG_SIZE = 160
MODEL_PATH = Path(__file__).parent / "models" / "gender_detection_final.keras"


# --------------------------------------------------
# 2. Load Trained Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


# --------------------------------------------------
# 3. Image Prediction Function
# --------------------------------------------------

def predict_image(image, model):

    image = image.convert("RGB")

    image_array = np.array(image)

    image_tensor = tf.convert_to_tensor(image_array)

    image_tensor = tf.cast(image_tensor, tf.float32)

    image_tensor = tf.image.resize_with_pad(
        image_tensor,
        IMG_SIZE,
        IMG_SIZE
    )

    image_tensor = image_tensor / 255.0

    image_tensor = tf.expand_dims(image_tensor, axis=0)

    probability = float(
        model.predict(image_tensor, verbose=0)[0][0]
    )

    predicted_label = (
        "Woman" if probability >= 0.5 else "Man"
    )

    return predicted_label, probability


# --------------------------------------------------
# 4. Application Interface
# --------------------------------------------------

st.title("🧠 Gender Detection - ML Demo")

st.write(
    "An educational image-classification application "
    "built using a Convolutional Neural Network (CNN)."
)

st.divider()

st.subheader("Upload an Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# 5. Process Uploaded Image
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width="stretch"
    )

    if st.button("Run Prediction", type="primary"):

        try:
            with st.spinner("Analyzing image..."):

                model = load_model()

                predicted_label, probability = predict_image(
                    image,
                    model
                )

            st.divider()

            st.subheader("Prediction Result")

            st.success(
                f"Predicted dataset label: {predicted_label}"
            )

            st.metric(
                "Predicted label probability",
                f"{probability * 100:.2f}%"
                if predicted_label == "Woman"
                else f"{(1 - probability) * 100:.2f}%"
            )

            st.write("**Model probability scores**")

            st.write(
                f"Woman: {probability * 100:.2f}%"
            )

            st.progress(probability)

            st.write(
                f"Man: {(1 - probability) * 100:.2f}%"
            )

            st.progress(1 - probability)

        except Exception as error:
            st.error(f"Prediction failed: {error}")

else:
    st.info("Please upload an image to begin.")


# --------------------------------------------------
# 6. Model Information and Disclaimer
# --------------------------------------------------

st.divider()

st.subheader("Model Information")

st.write("""
- Model: Convolutional Neural Network (CNN)
- Input image size: 160 × 160 pixels
- Number of output classes: 2
- Test accuracy: 82.08%
- ROC-AUC: 0.8878
""")

st.warning("""
**Educational demonstration only:** This model predicts
the binary labels present in its training dataset. It
cannot determine a person's gender identity. Its predictions
should not be used for identifying, profiling, or making
decisions about real people.
""")

st.caption("Developed as a Data Science educational project.")
