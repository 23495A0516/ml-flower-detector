import cv2
import joblib
import numpy as np
import streamlit as st
from skimage.feature import hog

# =================== Custom Background and Styling =================== #
def set_bg():
    st.markdown(
        """
        <style>
        body {
            background-color: #f5f5f5;
        }
        .stApp {
            # background-image: url("https://images.unsplash.com/photo-1501004318641-b39e6451bec6?auto=format&fit=crop&w=1950&q=80");
            background-size: cover;
            background-color: royalblue;
            background-attachment: fixed;
            background-position: center;
        }
        .title {
            font-size: 48px;
            font-weight: bold;
            color: white;
            text-align: center;
            margin-top: 20px;
            animation: fadeInDown 1s ease-out;
        }
        .subtitle {
            font-size: 20px;
            color: black;
            text-align: center;
            margin-bottom: 30px;
            animation: fadeIn 2s ease-in;
        }
        @keyframes fadeInDown {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# =================== Load Model and Encoder =================== #
def load_model():
    model = joblib.load("flower_classifier.pkl")  # Classifier model
    le = joblib.load("label_encoder.pkl")         # Label encoder
    return model, le

# =================== Prediction Function =================== #
def predict(image):
    model, le = load_model()
    image = cv2.resize(image, (128, 128))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog(gray_image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    features = features.reshape(1, -1)
    prediction = model.predict(features)
    return le.inverse_transform(prediction)[0]

# =================== Main App =================== #
set_bg()

st.markdown('<div class="title">🌸 Flower Image Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a flower image to identify its type using Machine Learning</div>', unsafe_allow_html=True)

file = st.file_uploader("Choose a flower image...", type=["jpg", "jpeg", "png"])

if file is not None:
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(image, caption="Uploaded Image", use_container_width=True)
# =================== Flower Info Dictionary =================== #
    flower_info = {
    "Lily": "🌸 **Lily** (*Lilium*): Lilies are graceful, trumpet-shaped flowers often associated with purity, renewal, and motherhood. They come in a variety of colors like white, pink, and orange.",
    "Lotus": "🌺 **Lotus** (*Nelumbo nucifera*): The lotus is a sacred aquatic flower symbolizing purity, enlightenment, and rebirth. It blooms beautifully even in muddy waters.",
    "Orchid": "🌼 **Orchid** (*Orchidaceae*): Known for their vibrant colors and delicate structure, orchids symbolize love, beauty, and strength. With over 25,000 species, they’re among the largest plant families.",
    "Sunflower": "🌻 **Sunflower** (*Helianthus annuus*): Famous for their large yellow blooms and heliotropic nature (turning to follow the sun), sunflowers represent positivity, loyalty, and joy.",
    "Tulip": "🌷 **Tulip** (*Tulipa*): Tulips are cup-shaped flowers that come in a rainbow of colors. A symbol of perfect love, they were once so valuable they sparked 'Tulip Mania' in the 17th century!"
}

    if st.button("🔍 Classify Image"):
        with st.spinner("Classifying... please wait ⏳"):
            result = predict(image)
        # 📘 Show flower info if available
            if result in flower_info:
                st.info(f"📖 **About {result}:** {flower_info[result]}")        
        # st.success(f"🌼 The flower is classified as: **{result}**",)
        st.markdown(f"""
            <div style='
                padding: 25px;
                margin-top: 20px;
                background: linear-gradient(135deg, #f3e5f5, #e1bee7);
                border-left: 8px solid #8e24aa;
                border-radius: 12px;
                text-align: center;
                box-shadow: 0 0 20px rgba(142, 36, 170, 0.4);
                animation: fadeIn 1.5s ease-in-out;
            '>
                <h2 style='color: #6a1b9a; font-size: 28px; font-weight: bold;'>
                    🌼 The flower is classified as: <span style="color: #4a0072;">{result}</span>
                </h2>
            </div>
        """, unsafe_allow_html=True)
        

        st.markdown("""
    <hr style="margin-top:50px; border-top: 1px solid #bbb;">
    <div style='text-align: center; color: black;'>
        🌼 Built with 💖 using Python & Streamlit <br> 
        Developed by Maaz_Bhai | © 2025
    </div>
""", unsafe_allow_html=True)
