import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import tempfile, os, io, requests
from datetime import datetime

# App config
st.set_page_config(page_title="Brain Tumor Detection", page_icon="🧠", layout="centered")

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

                    # Download the logo image
                    logo_url = "https://static.vecteezy.com/system/resources/previews/011/863/863/non_2x/brain-connection-logo-design-digital-brain-logo-template-brainstorm-icon-logo-ideas-think-idea-concept-free-vector.jpg"
                    response = requests.get(logo_url)
                    logo_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
                    logo_path.write(response.content)
                    logo_path.close()

                    # Add Logo
                    c.drawImage(logo_path.name, 30, height - 60, width=50, height=50)

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

                    os.remove(logo_path.name)
                    return buffer

                pdf_buffer = generate_pdf()

                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_buffer,
                    file_name="Brain_Tumor_Report.pdf",
                    mime="application/pdf"
                )
