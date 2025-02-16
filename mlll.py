import streamlit as st
import pickle
import numpy as np
import base64

# Function to encode an image using base64
def get_base64_of_bin_file(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Set Streamlit page configuration
st.set_page_config(
    page_title="Placement Eligibility Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Encode the background image
bg_image_path = "u.webp"  # Change this to the actual image file path
base64_bg = get_base64_of_bin_file(bg_image_path)

# Apply Background Image for the Whole Page
st.markdown(
    f"""
    <style>
        /* Full-screen background */
        .stApp {{
            background-image: url("data:image/jpg;base64,{base64_bg}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}

        /* Glassmorphic Container */
        .main-container {{
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(12px);
            padding: 30px;
            border-radius: 15px;
            box-shadow: 5px 5px 20px rgba(0, 0, 0, 0.3);
            max-width: 700px;
            margin: auto;
            color: white;
        }}

        /* Title Styling */
        .title {{
            color: #ffffff;
            text-align: center;
            font-size: 32px;
            font-weight: bold;
        }}

        /* Styled Button */
        .stButton>button {{
            background-color: #28a745;
            color: white;
            font-size: 18px;
            border-radius: 10px;
            padding: 10px 20px;
            transition: 0.3s;
        }}

        .stButton>button:hover {{
            background-color: #218838;
            transform: scale(1.05);
        }}

        /* Success Message */
        .stSuccess {{
            background-color: rgba(255, 255, 255, 0.2);
            padding: 15px;
            border-radius: 10px;
            font-size: 20px;
            font-weight: bold;
            text-align: center;
        }}

        /* Input Field Glassmorphism */
        .stNumberInput>div>div>input, .stSelectbox>div>div>select {{
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(10px);
            border-radius: 10px;
            padding: 8px;
            color: white !important;
            font-weight: bold;
        }}

        /* Change placeholder text color */
        ::placeholder {{
            color: white;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Load the saved model and scaler
@st.cache_data
def load_model():
    try:
        with open("main.pkl", "rb") as file:
            model_data = pickle.load(file)
        return model_data["model"], model_data["encoder"], model_data["scaler"]
    except FileNotFoundError:
        st.error("⚠️ Model file not found! Ensure `main.pkl` is in the directory.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        st.stop()

model, encoder, scaler = load_model()

# Title
#st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown("<h1 class='title'>UGANDA UNIVERSITY</h1>", unsafe_allow_html=True)
st.write("🔍 AI-powered tool to check student placement eligibility.")

st.subheader("📊 Enter Student Details:")

# User input fields
col1, col2 = st.columns(2)

with col1:
    cgpa = st.number_input("📚 CGPA", value=7.0, min_value=0.0, max_value=10.0, step=0.1)
    intern = st.number_input("💼 Internships", value=0, min_value=0, max_value=10, step=1)
    project = st.number_input("🛠️ Projects", value=0, min_value=0, max_value=10, step=1)
    certification = st.number_input("📜 Certifications", value=0, min_value=0, max_value=10, step=1)
    aptitude = st.number_input("🧠 Aptitude Score", value=50, min_value=0, max_value=100, step=1)

with col2:
    skill_rating = st.number_input("⭐ Skill Rating", value=5.0, min_value=0.0, max_value=10.0, step=0.1)
    extra_curricular = st.selectbox("🎭 Extra Activities", ['YES', 'NO'])
    placement = st.selectbox("🎓 Placement Training", ['YES', 'NO'])
    ssc = st.number_input("🏫 SSC Marks (%)", value=75, min_value=0, max_value=100, step=1)
    hsc = st.number_input("🏛️ HSC Marks (%)", value=75, min_value=0, max_value=100, step=1)

# Convert categorical inputs to numerical
extra_curricular = 1 if extra_curricular == "YES" else 0
placement = 1 if placement == "YES" else 0

# Prepare input data
input_features = np.array([[cgpa, intern, project, certification, aptitude, 
                            skill_rating, extra_curricular, placement, ssc, hsc]])

# Scale the input features
scaled_features = scaler.transform(input_features)

# Prediction Button
if st.button("🔮 Predict"):
    prediction = model.predict(scaled_features)[0]
    result = "Eligible for Placement ✅" if prediction == 1 else "Not Eligible ❌"
    st.markdown(f"<div class='stSuccess'>{result}</div>", unsafe_allow_html=True)

# Close styled container
st.markdown('</div>', unsafe_allow_html=True)
