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

# Custom HTML & CSS Styling - Consider moving this to a separate CSS file for better organization
st.markdown("""
    <style>
        body, h1, h2, h3, h4, h5, h6 {
            font-family: 'Raleway', Arial, Helvetica, sans-serif;
        }
        .header-img {
            width: 100%;
            height: 700px; /* Consider adjusting height for different screen sizes */
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
            font-size: 48px; /* Consider making font size responsive */
        }
        .text-below-image p {
            color: white;
            font-size: 26px; /* Consider making font size responsive */
        }
        .report-section { /* Added style for report section */
            border: 1px solid #ddd;
            padding: 10px;
            margin-bottom: 20px;
            border-radius: 5px;
            background-color: #f9f9f9; /* Light grey background */
        }
        .feedback-section, .contact-section { /* Added style for feedback and contact sections */
            border: 1px solid #ddd;
            padding: 10px;
            margin-top: 20px;
            border-radius: 5px;
            background-color: #f9f9f9;
        }
    </style>
    <div class="header-img"></div>
    <div class="text-below-image">
        <h1>Brain Tumor Detection</h1>
        <p>Detect tumor regions from MRI scans using AI</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar Instructions - Use st.expander for better organization
with st.sidebar:
    st.title("ℹ️ How to Use")
    with st.expander("Instructions"):
        st.markdown("""
        1. Upload a brain MRI image (JPG/PNG).
        2. Click **Detect Tumor**.
        3. View AI predictions.
        4. Download PDF Report.
        """)

# Title and Upload Section - Use st.container for better layout
with st.container():
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

        # Patient Details - Use st.form for better data collection
        with st.form("patient_details_form"):
            st.markdown("### 🩺 Patient Information")
            patient_name = st.text_input("Patient Name", key="patient_name") # Added key
            patient_age = st.number_input("Age", min_value=0, key="patient_age") # Added key
            patient_gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="patient_gender") # Added key

            # Medical History - Use st.expander for better organization
            with st.expander("📋 Medical History"):
                previous_illnesses = st.text_area("Previous illnesses or conditions", key="previous_illnesses") # Added key
                allergies = st.text_area("Allergies", key="allergies") # Added key
                medications = st.text_area("Medications currently taking", key="medications") # Added key
                family_history = st.text_area("Family medical history", key="family_history") # Added key

            # Data Privacy & Consent - Use st.checkbox with a clear label
            consent_given = st.checkbox("I consent to the use of my data (image and medical details) for diagnostic and research purposes.", value=False, key="consent_given") # Added key
            st.markdown("### 🔒 Data Privacy & Consent")

            with st.expander("View Our Data Privacy Policy"):
                st.markdown("""
                - We do **not** store any personally identifiable data without your consent.
                - Your uploaded MRI scans and details are used **only for AI analysis** during this session.
                - We follow standard security protocols to ensure your data is protected.
                - You can request deletion of any report generated.
                """)
            submit_button = st.form_submit_button("Detect Tumor") # Moved the button inside form

        # Stepper for Progress - Consider using a different visual representation
        steps = ['MRI Upload', 'Detection', 'Results', 'Report']
        step_idx = 0
        progress = st.progress(0)
        for step in steps:
            st.write(f"Step {step_idx + 1}: {step}")
            progress.progress((step_idx + 1) * 25)
            time.sleep(0.5)  # Simulate each step, reduced time for better flow
            step_idx += 1

        # Button for starting detection
        if submit_button: # Changed from st.button to submit_button
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

                    # Show Inference Results - Use st.expander for better organization
                    with st.expander("📝 Inference Results"):
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

                    # Personalized Health Insights - Use st.info for better visual presentation
                    st.markdown("### Personalized Health Insights")
                    st.write(f"Based on the analysis of the MRI scan for {patient_name}, the AI model has provided the following insights. It is crucial to consult with a medical professional for accurate diagnosis and treatment plans.")
                    if prediction_found:
                        st.info("AI has detected potential tumor regions. Further medical investigation is strongly recommended.")
                    else:
                        st.info("AI did not detect any significant tumor regions. However, clinical evaluation is always advisable.")

                    # Patient Education Section
                    st.markdown("### Patient Education Section")
                    st.write("Understanding your condition is important. Brain tumors can vary significantly in their nature and behavior. Benign tumors are non-cancerous and typically grow slowly, while malignant tumors are cancerous and can grow aggressively. Treatment strategies are highly personalized and depend on various factors including the type, size, and location of the tumor, as well as the patient's overall health.")
                    st.write("Common treatment options include surgery to remove the tumor, radiation therapy to target and destroy tumor cells, and chemotherapy, which uses drugs to kill cancer cells. Regular follow-up and monitoring are essential to manage the condition effectively.")

                    # Interactive QR Code for patient details
                    qr_data = f"Patient Name: {patient_name}\nAge: {patient_age}\nGender: {patient_gender}\nTumor Detected: {'Yes' if prediction_found else 'No'}"
                    qr = qrcode.make(qr_data)
                    qr_path = "/tmp/qr_code.png"
                    qr.save(qr_path)
                    st.image(qr_path, caption="Scan for Patient Details (QR Code)")

                    # Interactive Story Style (Patient's Journey) - Use st.write with bold text for emphasis
                    st.markdown("### Your Health Journey")
                    st.write(f"**Step 1: MRI Scan Uploaded** - Your MRI image has been successfully uploaded for analysis.")
                    st.write(f"**Step 2: AI Diagnosis** - The AI model has processed the image. {'Potential tumor regions were detected.' if prediction_found else 'No significant tumor regions were detected.'}")
                    st.write(f"**Step 3: Next Steps** - We recommend consulting with your doctor for a comprehensive evaluation and to discuss any necessary next steps.")

                    # PDF Generation with Patient Details and Results in Table Form - Enclose in a styled div
                    if prediction_found:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img_file:
                            image.save(temp_img_file.name, format="JPEG")
                            image_path = temp_img_file.name

                        def generate_pdf():
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    title_style = styles["h1"]
    normal_style = styles["normal"]
    bold_style = styles["b"]

    width, height = A4
    margin = inch

    # --- Report Header ---
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(width / 2, height - margin, "Brain Tumor Detection Report")
    c.line(margin, height - margin - 0.2 * inch, width - margin, height - margin - 0.2 * inch)
    c.setFont("Helvetica", 10)
    c.drawRightString(width - margin, height - margin - 0.4 * inch, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    y_position = height - margin - 0.7 * inch

    # --- Patient Information Section ---
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y_position, "Patient Information")
    y_position -= 0.2 * inch
    c.setFont("Helvetica", 12)
    c.drawString(margin, y_position, f"Name: {patient_name}")
    y_position -= 0.2 * inch
    c.drawString(margin, y_position, f"Age: {patient_age}")
    y_position -= 0.2 * inch
    c.drawString(margin, y_position, f"Gender: {patient_gender}")
    y_position -= 0.2 * inch
    c.drawString(margin, y_position, f"Image Name: {uploaded_file.name}")
    y_position -= 0.3 * inch

    # --- Medical History Section ---
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y_position, "Medical History")
    y_position -= 0.2 * inch
    c.setFont("Helvetica", 12)
    c.drawString(margin, y_position, f"Previous illnesses/conditions: {previous_illnesses if previous_illnesses else 'N/A'}")
    y_position -= 0.2 * inch
    c.drawString(margin, y_position, f"Allergies: {allergies if allergies else 'N/A'}")
    y_position -= 0.2 * inch
    c.drawString(margin, y_position, f"Medications: {medications if medications else 'N/A'}")
    y_position -= 0.2 * inch
    c.drawString(margin, y_position, f"Family History: {family_history if family_history else 'N/A'}")
    y_position -= 0.3 * inch

    # --- AI Prediction Summary Section ---
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y_position, "AI Prediction Summary")
    y_position -= 0.2 * inch
    c.setFont("Helvetica", 12)
    if result.get("predictions"):
        data = [["#", "Confidence"]]
        for i, pred in enumerate(result["predictions"]):
            conf = pred.get("confidence", 0)
            data.append([f"Tumor {i+1}", f"{conf:.2f}"])

        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        table.wrapOn(c, width - 2 * margin, height)
        table_height = table.drawOn(c, margin, y_position - table.height)
        y_position -= (table_height + 0.2 * inch)
    else:
        c.drawString(margin, y_position, "No tumor detected by the AI model.")
        y_position -= 0.3 * inch

    # --- Detected Image ---
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin, y_position, "Detected Tumor Region (if any)")
    y_position -= 0.2 * inch
    img_width = 4 * inch
    img_height = 4 * inch
    try:
        c.drawImage(image_path, margin, y_position - img_height, width=img_width, height=img_height, preserveAspectRatio=True, mask='auto')
        y_position -= (img_height + 0.3 * inch)
    except Exception as e:
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(margin, y_position, "No detection image available.")
        y_position -= 0.3 * inch

    # --- Disclaimer ---
    c.setFont("Helvetica", 10)
    c.setFillColorRGB(0.5, 0.5, 0.5)
    disclaimer_text = "Disclaimer: This report is based on an AI analysis of the provided MRI scan and patient information. It should not be considered a definitive medical diagnosis. Consultation with a qualified medical professional is essential for accurate diagnosis and treatment recommendations."
    disclaimer = Paragraph(disclaimer_text, normal_style)
    disclaimer.wrapOn(c, width - 2 * margin, height)
    disclaimer.drawOn(c, margin, margin + 0.5 * inch)

    c.save()
    buffer.seek(0)
    return buffer

# --- User Feedback Section --- - Enclose in a styled div
st.markdown("<div class='feedback-section'>", unsafe_allow_html=True) # Added a styled div
st.title("We Value Your Feedback")
rating = st.radio("How would you rate your experience with the AI model?", options=[1, 2, 3, 4, 5])
feedback = st.text_area("Any comments or suggestions?")

if st.button("Submit Feedback"):
    st.write(f"Thank you for your feedback! Rating: {rating} stars")
    if feedback:
        st.write(f"Your comments: {feedback}")
st.markdown("</div>", unsafe_allow_html=True) # Closing the div

# --- Contact Section --- - Enclose in a styled div
st.markdown("<div class='contact-section'>", unsafe_allow_html=True) # Added a styled div
st.title("📞 Contact Us")
st.markdown("""
    If you have any issues or inquiries, feel free to contact us:
    - **Email**: [rkumar1514@gmail.com](mailto:rkumar1514@gmail.com)
    - **WhatsApp**: [+91 7092309109](https://wa.me/7092309109)
""")
st.markdown("</div>", unsafe_allow_html=True) # Closing the div
