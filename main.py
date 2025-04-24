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

# App config
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="centered")

# Roboflow Client
CLIENT = InferenceHTTPClient(
    api_url="https://outline.roboflow.com",
    api_key="AUriIUOQuEbHt8npqPyt"
)

# Custom CSS Styling
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
        .styled-box {
            border: 1px solid #e6e6e6;
            padding: 20px;
            border-radius: 10px;
            background-color: #f9f9f9;
            margin-bottom: 20px;
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

# Upload Section
st.markdown("""
    <div class="styled-box">
        <p>📤 Upload your MRI scan image for AI-based analysis.</p>
    </div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📄 Upload Brain MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name, format="JPEG")
        temp_path = temp_file.name

    # Patient Info
    with st.container():
        st.markdown("### 🩺 Patient Information")
        with st.expander("Fill out patient details"):
            patient_name = st.text_input("Patient Name")
            patient_age = st.number_input("Age", min_value=0)
            patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    # Medical History
    with st.container():
        st.markdown("### 📋 Medical History")
        with st.expander("Fill out medical history"):
            previous_illnesses = st.text_area("Previous illnesses or conditions")
            allergies = st.text_area("Allergies")
            medications = st.text_area("Medications currently taking")
            family_history = st.text_area("Family medical history")

    # Privacy Notice
    with st.container():
        st.markdown("### 🔒 Data Privacy & Consent")
        consent_given = st.checkbox("I consent to the use of my data (image and medical details) for diagnostic and research purposes.", value=False)
        with st.expander("View Our Data Privacy Policy"):
            st.markdown("""
            - We do **not** store any personally identifiable data without your consent.
            - Your uploaded MRI scans and details are used **only for AI analysis** during this session.
            - We follow standard security protocols to ensure your data is protected.
            - You can request deletion of any report generated.
            """)

    # Stepper Progress
    with st.container():
        steps = ['MRI Upload', 'Detection', 'Results', 'Report']
        step_idx = 0
        progress = st.progress(0)
        for step in steps:
            st.write(f"Step {step_idx + 1}: {step}")
            progress.progress((step_idx + 1) * 25)
            time.sleep(0.5)
            step_idx += 1

    # Detect Button
    if st.button("🔍 Detect Tumor", key="detect", help="Click to detect brain tumor in the MRI scan"):
        if not consent_given:
            st.error("⚠️ Please provide consent to proceed with the analysis.")
        elif not uploaded_file:
            st.warning("⬆️ Please upload an MRI image first.")
        elif not patient_name:
            st.warning("✍️ Please enter the patient's name.")
        elif patient_age <= 0:
            st.warning("🔢 Please enter a valid age.")
        else:
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

                # Health Insights
                st.markdown("### Personalized Health Insights")
                st.info("AI has detected potential tumor regions. Further medical investigation is strongly recommended." if prediction_found else "AI did not detect any significant tumor regions. However, clinical evaluation is always advisable.")

                # Education Section
                st.markdown("### Patient Education Section")
                st.write("Understanding your condition is important. Brain tumors can vary significantly...")

                # QR Code
                qr_data = f"Patient Name: {patient_name}\nAge: {patient_age}\nGender: {patient_gender}\nTumor Detected: {'Yes' if prediction_found else 'No'}"
                qr = qrcode.make(qr_data)
                qr_path = "/tmp/qr_code.png"
                qr.save(qr_path)
                st.image(qr_path, caption="Scan for Patient Details (QR Code)")

                # Patient Journey
                st.markdown("### Your Health Journey")
                st.write(f"**Step 1: MRI Scan Uploaded** - Your MRI image has been successfully uploaded for analysis.")
                st.write(f"**Step 2: AI Diagnosis** - {'Potential tumor regions were detected.' if prediction_found else 'No significant tumor regions were detected.'}")
                st.write("**Step 3: Next Steps** - We recommend consulting with your doctor...")

                # PDF Report
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
                        c.drawString(60, height - 210, f"Previous illnesses or conditions: {previous_illnesses if previous_illnesses else 'N/A'}")
                        c.drawString(60, height - 230, f"Allergies: {allergies if allergies else 'N/A'}")
                        c.drawString(60, height - 250, f"Medications: {medications if medications else 'N/A'}")
                        c.drawString(60, height - 270, f"Family History: {family_history if family_history else 'N/A'}")

                        y_cursor = height - 300
                        c.drawString(50, y_cursor, "🧠 Prediction Summary:")
                        y_cursor -= 20
                        if result.get("predictions"):
                            for i, pred in enumerate(result["predictions"]):
                                conf = pred.get("confidence", 0)
                                c.drawString(60, y_cursor, f"• Tumor {i+1}: Confidence {conf:.2f}")
                                y_cursor -= 20
                        else:
                            c.drawString(60, y_cursor, "• No tumor detected by AI.")
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
                        file_name=f"{patient_name.replace(' ', '_')}_Brain_Tumor_Report.pdf",
                        mime="application/pdf"
                    )

# Feedback
st.title("We Value Your Feedback")
rating = st.radio("How would you rate your experience with the AI model?", options=[1, 2, 3, 4, 5])
feedback = st.text_area("Any comments or suggestions?")

if st.button("Submit Feedback"):
    st.write(f"Thank you for your feedback! Rating: {rating} stars")
    if feedback:
        st.write(f"Your comments: {feedback}")

# Contact Section
st.title("📞 Contact Us")
st.markdown("""
    If you have any issues or inquiries, feel free to contact us:
    - **Email**: [rkumar1514@gmail.com](mailto:rkumar1514@gmail.com)
    - **WhatsApp**: [+91 7092309109](https://wa.me/7092309109)
""")
