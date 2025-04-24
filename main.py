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
from googletrans import Translator
import time


# App config
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="centered")

# Initialize Translator object
translator = Translator()

# Define translations for various sections of the app
translations = {
    "en": {
        "title": "Brain Tumor Detection",
        "instructions": """
        1. Upload a brain MRI image (JPG/PNG).
        2. Click **Detect Tumor**.
        3. View AI predictions.
        4. Download PDF Report.
        """,
        "upload_text": "Upload your MRI scan image for AI-based analysis.",
        "patient_info": "Patient Information",
        "medical_history": "Medical History",
        "data_privacy": "Data Privacy & Consent",
        "detect_tumor": "🔍 Detect Tumor",
        "thank_you_feedback": "Thank you for your feedback! Rating: {rating} stars",
        "contact_us": "📞 Contact Us",
    },
    "es": {  # Spanish Translation
        "title": "Detección de Tumor Cerebral",
        "instructions": """
        1. Cargue una imagen de MRI cerebral (JPG/PNG).
        2. Haga clic en **Detectar Tumor**.
        3. Ver predicciones de IA.
        4. Descargar informe en PDF.
        """,
        "upload_text": "Suba su imagen de MRI para el análisis basado en IA.",
        "patient_info": "Información del Paciente",
        "medical_history": "Historial Médico",
        "data_privacy": "Privacidad de los Datos y Consentimiento",
        "detect_tumor": "🔍 Detectar Tumor",
        "thank_you_feedback": "¡Gracias por su retroalimentación! Calificación: {rating} estrellas",
        "contact_us": "📞 Contáctenos",
    },
    # Add more languages as needed
}

# Language Selector
language = st.selectbox("Choose your language", options=["English", "Español"], index=0)
lang_key = "en" if language == "English" else "es"

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
        background: url('https://th.bing.com/th/id/OIP._8eGDGy8q02rSWkuvrrhhAHaEw?rs=1&pid=ImgDetMain') no-repeat center center;
        background-size: cover;
        animation: fadeIn 2s ease-in-out;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .text-below-image {
        text-align: center;
        padding: 20px;
    }
    .text-below-image h1 {
        color: red;
        font-size: 48px;
    }
    .text-below-image p {
        color: white;
        font-size: 26px;
    }
    </style>
    <div class="header-img"></div>
    <div class="text-below-image">
        <h1>Brain Tumor Detection</h1>
        <p>Detect tumor regions from MRI scans using AI</p>
    </div>
""", unsafe_allow_html=True)


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

            # Interactive Story Style (Patient's Journey)
            st.markdown("### Interactive Story")
            st.write("Here's your health journey: Step 1 - MRI Scan, Step 2 - Diagnosis, Step 3 - Treatment Recommendations.")

            # PDF Generation with Patient Details and Results in Table Form
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
                    c.drawString(50, height - 100, f"Patient Name: {patient_name}")
                    c.drawString(50, height - 120, f"Age: {patient_age}")
                    c.drawString(50, height - 140, f"Gender: {patient_gender}")
                    c.drawString(50, height - 160, f"Image Name: {uploaded_file.name}")

                    c.drawString(50, height - 190, "Medical History:")
                    c.drawString(60, height - 210, f"Previous illnesses or conditions: {previous_illnesses}")
                    c.drawString(60, height - 230, f"Allergies: {allergies}")
                    c.drawString(60, height - 250, f"Medications: {medications}")
                    c.drawString(60, height - 270, f"Family History: {family_history}")

                    y_cursor = height - 300
                    c.drawString(50, y_cursor, "🧠 Prediction Summary:")
                    y_cursor -= 20
                    for i, pred in enumerate(result["predictions"]):
                        conf = pred.get("confidence", 0)
                        c.drawString(60, y_cursor, f"• Tumor {i+1}: Confidence {conf:.2f}")
                        y_cursor -= 20

                    img_width = 300
                    img_height = 300
                    c.drawImage(image_path, width - img_width - 50, 100, width=img_width, height=img_height, preserveAspectRatio=True, mask='auto')
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

# --- User Feedback Section ---
st.title("We Value Your Feedback")
rating = st.radio("How would you rate your experience with the AI model?", options=[1, 2, 3, 4, 5])
feedback = st.text_area("Any comments or suggestions?")

if st.button("Submit Feedback"):
    st.write(f"Thank you for your feedback! Rating: {rating} stars")
    if feedback:
        st.write(f"Your comments: {feedback}")

# --- Contact Section ---
st.title("📞 Contact Us")
st.markdown("""
    If you have any issues or inquiries, feel free to contact us:
    - **Email**: [support@braintumordetector.com](mailto:support@braintumordetector.com)
    - **WhatsApp**: [+91 7092309109](https://wa.me/7092309109)
""")
