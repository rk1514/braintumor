import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os

# Initialize Roboflow Inference Client
CLIENT = InferenceHTTPClient(
    api_url="https://outline.roboflow.com",
    api_key="AUriIUOQuEbHt8npqPyt"
)

# Streamlit UI
st.title("🧠 Brain Tumor Detection")
st.write("Upload a brain scan to detect tumor regions using AI.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name, format="JPEG")
        temp_path = temp_file.name

    if st.button("🔍 Detect Tumor"):
        with st.spinner("Detecting..."):
            try:
                result = CLIENT.infer(temp_path, model_id="brain-tumor-detection-lovmz/5")
            except Exception as e:
                st.error(f"❌ Inference failed: {e}")
                os.remove(temp_path)
                st.stop()

            os.remove(temp_path)

            st.write("### 📋 Prediction Result")
            st.json(result)

            # Draw bounding boxes on image
            if "predictions" in result and len(result["predictions"]) > 0:
                draw = ImageDraw.Draw(image)

                for pred in result["predictions"]:
                    x, y = pred["x"], pred["y"]
                    w, h = pred["width"], pred["height"]
                    conf = pred.get("confidence", 0)

                    # Define bounding box
                    left = x - w / 2
                    top = y - h / 2
                    right = x + w / 2
                    bottom = y + h / 2

                    draw.rectangle([(left, top), (right, bottom)], outline="red", width=3)
                    draw.text((left, top - 10), f"Tumor: {conf:.2f}", fill="red")

                st.image(image, caption="🟥 Tumor Region(s) Detected", use_column_width=True)
            else:
                st.success("✅ No tumor detected in the image.")
