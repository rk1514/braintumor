import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import tempfile, os, io
import openai
import pandas as pd
from datetime import datetime
import qrcode
import time
import random

# App config
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="centered")

# OpenAI API Key (for Chatbot Assistance)
openai.api_key = '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'

# Roboflow Client
CLIENT = InferenceHTTPClient(
    api_url="https://outline.roboflow.com",
    api_key="AUriIUOQuEbHt8npqPyt"
)

# Custom HTML & CSS Styling
st.markdown("""
    <style>
    body, h1, h2, h3, h4, h5, h6 {
        font-family: 'Raleway', Arial, Helvetica, sans-serif;
    }
    .header-img {
        width: 100%;
        height: 700px;
        background: url('https://th.bing.com/th/id/OIP.xYrQ8Rd-tCPzzdvZOfNNoAHaHa?w=500&h=500&rs=1&pid=ImgDetMain') no-repeat center center;
        background-size: cover;
        animation: fadeIn 2s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .content-section {
        padding: 20px;
        text-align: center;
    }
    .button-style {
        background-color: #e53935;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        font-size: 16px;
        border: none;
    }
    .button-style:hover {
        background-color: #d32f2f;
        cursor: pointer;
    }
    .image-container {
        text-align: center;
        margin: 20px 0;
    }
    </style>
    <div class="header-img">
        <h1 style="text-align: center; padding-top: 250px; color: red; font-size: 48px;">Brain Tumor Detection</h1>
        <p style="text-align: center; color: blue; font-size: 26px;">Detect tumor regions from MRI scans using AI</p>
    </div>
""", unsafe_allow_html=True)

# Light/Dark Theme Toggle
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

def toggle_theme():
    if st.session_state.theme == 'light':
        st.session_state.theme = 'dark'
    else:
        st.session_state.theme = 'light'

# Create the toggle button
st.button("🔄 Toggle Theme", on_click=toggle_theme)

# Apply custom CSS for light/dark theme
if st.session_state.theme == 'light':
    theme_css = """
        <style>
        body {
            background-color: #ffffff;
            color: #000000;
        }
        .button-style {
            background-color: #e53935;
            color: white;
        }
        .button-style:hover {
            background-color: #d32f2f;
        }
        </style>
    """
else:
    theme_css = """
        <style>
        body {
            background-color: #121212;
            color: #ffffff;
        }
        .button-style {
            background-color: #e53935;
            color: white;
        }
        .button-style:hover {
            background-color: #d32f2f;
        }
        </style>
    """
st.markdown(theme_css, unsafe_allow_html=True)

# Sidebar Instructions
with st.sidebar:
    st.title("ℹ️ How to Use")
    st.markdown("""
    1. Upload a brain MRI image (JPG/PNG).
    2. Click **Detect Tumor**.
    3. View AI predictions.
    4. Download PDF Report.
    """)

# Title and Upload Section
st.markdown("""
    <div class="content-section">
        <p>Upload your MRI scan image for AI-based analysis.</p>
    </div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📄 Upload Brain MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name, format="JPEG")
        temp_path = temp_file.name

    # Add patient details inputs
    st.markdown("### 🩺 Patient Information")
    patient_name = st.text_input("Patient Name")
    patient_age = st.number_input("Age", min_value=0)
    patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    # Medical History Section
    st.markdown("### 📋 Medical History")
    previous_illnesses = st.text_area("Previous illnesses or conditions")
    allergies = st.text_area("Allergies")
    medications = st.text_area("Medications currently taking")
    family_history = st.text_area("Family medical history")

    # Data Privacy & Consent Notice
    st.markdown("### 🔒 Data Privacy & Consent")
    consent_given = st.checkbox("I consent to the use of my data (image and medical details) for diagnostic and research purposes.", value=False)

    with st.expander("View Our Data Privacy Policy"):
        st.markdown("""
        - We do **not** store any personally identifiable data without your consent.
        - Your uploaded MRI scans and details are used **only for AI analysis** during this session.
        - We follow standard security protocols to ensure your data is protected.
        - You can request deletion of any report generated.
        """)

    # Stepper for Progress
    steps = ['MRI Upload', 'Detection', 'Results', 'Report']
    step_idx = 0
    progress = st.progress(0)
    for step in steps:
        st.write(f"Step {step_idx + 1}: {step}")
        progress.progress((step_idx + 1) * 25)
        time.sleep(1)  # Simulate each step
        step_idx += 1

    # Button for starting detection
    if st.button("🔍 Detect Tumor", key="detect", help="Click to detect brain tumor in the MRI scan"):
        if not consent_given:
            st.warning("⚠️ Please provide consent before proceeding with the analysis.")
            st.stop()

        with st.spinner("Analyzing with AI model..."):
            try:
                result = CLIENT.infer(temp_path, model_id="brain-tumor-detection-lovmz/5")
            except Exception as e:
                st.error(f"❌ Inference failed: {e}")
                os.remove(temp_path)
                st.stop()

            os.remove(temp_path)

            # Show Inference Results
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

                st.image(image, caption="🔵 Detected Tumor Region(s)", use_column_width=True)
                st.success("🔬 Tumor detected. Download report below.")
            else:
                st.image(image, caption="✅ No Tumor Detected", use_column_width=True)
                st.success("🎉 No tumor regions detected in the image.")
                st.balloons()  # Show confetti animation

            # Personalized Health Insights
            st.markdown("### Personalized Health Insights")
            st.write("Based on your MRI scan, we recommend that you consult with your doctor for a follow-up MRI scan and discuss treatment options.")

            # Patient Education Section
            st.markdown("### Patient Education Section")
            st.write("Brain tumors can be classified into benign and malignant types. Treatment depends on the tumor type, location, and size. Surgery, radiation, and chemotherapy are common treatment options.")

            # Interactive QR Code for patient details
            qr_data = f"Patient Name: {patient_name}\nAge: {patient_age}\nGender: {patient_gender}\nTumor Type: {'Benign' if prediction_found else 'None'}"
            qr = qrcode.make(qr_data)
            qr_path = "/tmp/qr_code.png"
            qr.save(qr_path)
            st.image(qr_path, caption="Scan for Patient Details (QR Code)")

            # PDF Generation with Patient Details and Results in Table Form
            if prediction_found:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img_file:
                    image.save(temp_img_file.name, format="JPEG")
                    image_path = temp_img_file.name

                def generate_pdf():
                    buffer = io.BytesIO()
                    c = canvas.Canvas(buffer, pagesize=A4)
                    c.drawString(100, 800, f"Patient Name: {patient_name}")
                    c.drawString(100, 780, f"Age: {patient_age}")
                    c.drawString(100, 760, f"Gender: {patient_gender}")
                    c.drawString(100, 740, "Tumor Type: Benign")
                    c.drawImage(image_path, 100, 200, width=400, height=400)
                    c.save()
                    buffer.seek(0)
                    return buffer

                st.download_button(
                    label="Download PDF Report",
                    data=generate_pdf(),
                    file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )

        st.stop()
