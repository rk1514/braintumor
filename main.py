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

# App config
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="centered")

# OpenAI API Key (for Chatbot Assistance)
openai.api_key = 'your-api-key-here'

# Roboflow Client
CLIENT = InferenceHTTPClient(
    api_url="https://outline.roboflow.com",
    api_key="your-api-key-here"
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
        background: url('https://example.com/image.jpg') no-repeat center center;
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

# Sidebar Instructions
with st.sidebar:
    st.title("ℹ️ How to Use")
    st.markdown("""
    1. Upload a brain MRI image (JPG/PNG).
    2. Click **Detect Tumor**.
    3. View AI predictions.
    4. Download PDF Report.
    """)

# Initialize variables with default values
patient_name = ""
patient_age = 0
patient_gender = "Male"
detection_results = None

# Title and Upload Section
st.markdown("""
    <div class="content-section">
        <p>Upload your MRI scan image for AI-based analysis.</p>
    </div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload Brain MRI Image", type=["jpg", "jpeg", "png"])

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

    # Button for starting detection
    if st.button("🔍 Detect Tumor", key="detect", help="Click to detect brain tumor in the MRI scan"):
        with st.spinner("Analyzing with AI model..."):
            try:
                result = CLIENT.infer(temp_path, model_id="brain-tumor-detection-lovmz/5")
                detection_results = result
            except Exception as e:
                st.error(f"❌ Inference failed: {e}")
                os.remove(temp_path)
                st.stop()

            os.remove(temp_path)

            # Show Inference Results
            st.markdown("### 📝 Inference Results")
            st.json(detection_results)

            prediction_found = False
            draw = ImageDraw.Draw(image)

            if detection_results.get("predictions"):
                prediction_found = True
                for pred in detection_results["predictions"]:
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

                    # Table Header
                    c.setFont("Helvetica-Bold", 10)
                    c.drawString(50, height - 200, "Predictions:")
                    c.setFont("Helvetica", 8)
                    table_header = ["Tumor No.", "Confidence", "X-Coordinate", "Y-Coordinate", "Width", "Height"]
                    y_position = height - 220
                    for col, header in enumerate(table_header):
                        c.drawString(50 + col * 100, y_position, header)

                    y_position -= 20
                    for idx, pred in enumerate(detection_results["predictions"]):
                        c.drawString(50, y_position, str(idx + 1))
                        c.drawString(150, y_position, f"{pred.get('confidence', 0):.2f}")
                        c.drawString(250, y_position, str(pred.get('x', '')))
                        c.drawString(350, y_position, str(pred.get('y', '')))
                        c.drawString(450, y_position, str(pred.get('width', '')))
                        c.drawString(550, y_position, str(pred.get('height', '')))
                        y_position -= 20

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

# Real-Time Chatbot Assistance
st.title("Real-Time Chatbot Assistance")
user_input = st.text_input("Ask a question about your diagnosis:")

if user_input:
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=user_input,
        max_tokens=150
    )
    st.write(response.choices[0].text.strip())

# --- User Feedback Section ---
st.title("📝 User Feedback")
feedback = st.text_area("Please provide your feedback on the app or any suggestions for improvement.")
if st.button("Submit Feedback"):
    if feedback:
        st.success("Thank you for your feedback! We value your input to improve the app.")
        # You can also save the feedback into a database or a file for further analysis
    else:
        st.warning("Please enter your feedback before submitting.")

# --- Contact Section ---
st.title("📞 Contact Us")
st.markdown("""
    If you have any issues or inquiries, feel free to contact us:
    - **Email**: [support@braintumordetector.com](mailto:support@braintumordetector.com)
    - **WhatsApp**: [+91 7092309109](https://wa.me/7092309109)
""")
