import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import tempfile, os, io
from datetime import datetime

# App config
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="centered")

# Roboflow Client
CLIENT = InferenceHTTPClient(
    api_url="https://outline.roboflow.com",
    api_key="AUriIUOQuEbHt8npqPyt"
)

# Sidebar Instructions
with st.sidebar:
    st.title("ℹ️ How to Use")
    st.markdown("""
    1. Upload a brain MRI image (JPG/PNG).
    2. Click **Detect Tumor**.
    3. View AI predictions.
    4. Download PDF Report.
    """)

# Upload Section
st.title("🧠 Brain Tumor Detection")
st.markdown("Upload your MRI scan image for AI-based analysis.")

uploaded_file = st.file_uploader("📤 Upload Brain MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name, format="JPEG")
        temp_path = temp_file.name

    # Button for starting detection
    if st.button("🔍 Detect Tumor", key="detect", help="Click to detect brain tumor in the MRI scan"):
        with st.spinner("Analyzing with AI model..."):
            try:
                result = CLIENT.infer(temp_path, model_id="brain-tumor-detection-lovmz/5")
            except Exception as e:
                st.error(f"❌ Inference failed: {e}")
                os.remove(temp_path)
                st.stop()

            os.remove(temp_path)

            st.markdown("### 📝 Inference Results")
            st.json(result)

            prediction_found = False
            draw = ImageDraw.Draw(image)

            if result.get("predictions"):
                prediction_found = True
                for pred in result["predictions"]:
                    x, y = pred["x"], pred["y"]
                    w, h = pred["width"], pred["height"]
                    conf = pred.get("confidence", 0)
                    left = x - w / 2
                    top = y - h / 2
                    right = x + w / 2
                    bottom = y + h / 2
                    draw.rectangle([(left, top), (right, bottom)], outline="red", width=3)
                    draw.text((left, top - 15), f"Tumor: {conf:.2f}", fill="red")

                st.image(image, caption="🟥 Detected Tumor Region(s)", use_column_width=True)
                st.success("🔬 Tumor detected. Download report below.")
            else:
                st.image(image, caption="✅ No Tumor Detected", use_column_width=True)
                st.success("🎉 No tumor regions detected in the image.")

            # PDF Generation
            if prediction_found:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img_file:
                    image.save(temp_img_file.name, format="JPEG")
                    image_path = temp_img_file.name

                def generate_pdf():
                    buffer = io.BytesIO()
                    c = canvas.Canvas(buffer, pagesize=A4)
                    width, height = A4

                    c.setFont("Helvetica-Bold", 20)
                    c.drawCentredString(width / 2, height - 50, "Brain Tumor Detection Report")

                    c.setFont("Helvetica", 12)
                    c.drawString(50, height - 80, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    c.drawString(50, height - 100, f"Image Name: {uploaded_file.name}")

                    c.drawString(50, height - 130, "🧠 Prediction Summary:")
                    y_cursor = height - 150
                    for i, pred in enumerate(result["predictions"]):
                        conf = pred.get("confidence", 0)
                        c.drawString(60, y_cursor, f"• Tumor {i+1}: Confidence {conf:.2f}")
                        y_cursor -= 20

                    c.drawImage(image_path, 100, 100, width=400, preserveAspectRatio=True, mask='auto')
                    c.showPage()
                    c.save()
                    buffer.seek(0)
                    return buffer

                pdf_buffer = generate_pdf()

                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_buffer,
                    file_name="Brain_Tumor_Report.pdf",
                    mime="application/pdf"
                )
