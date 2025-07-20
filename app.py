import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("best_model.pkl")

# Page config
st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💼",
    layout="centered",
)

# Custom black & blue theme
st.markdown("""
    <style>
        html, body, .stApp {
            background-color: #0F2027 !important;
            color: white;
        }
        .stButton > button {
            background-color: #0077B6;
            color: white;
            font-weight: bold;
            border-radius: 8px;
        }
        div[data-baseweb="slider"] [role="slider"] {
            background-color: #48CAE4 !important;
            border: 2px solid #0077B6 !important;
        }
    </style>
""", unsafe_allow_html=True)


# Title
st.markdown("<h1 style='text-align: center; color: #00B4D8;'>💼 Employee Salary Classifier</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #90E0EF;'>Predict whether salary is >50K based on user profile</h4>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar Inputs (fnlwgt included)
st.sidebar.title("📋 Enter Employee Details")

age = st.sidebar.slider("Age", 18, 65, 30)
workclass = st.sidebar.selectbox("Workclass", ["Private", "Self-emp", "Government", "Other"])
fnlwgt = st.sidebar.number_input("FNLWGT", value=200000)
education_num = st.sidebar.slider("Educational Number", 1, 16, 10)
occupation = st.sidebar.selectbox("Occupation", [
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
    "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
    "Transport-moving", "Priv-house-serv", "Protective-serv",
    "Armed-Forces"
])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
hours_per_week = st.sidebar.slider("Hours per Week", 1, 100, 40)
native_country = st.sidebar.selectbox("Native Country", ["United-States", "India", "Mexico", "Other"])

# Input DataFrame
input_df = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    'fnlwgt': [fnlwgt],
    'educational-num': [education_num],
    'occupation': [occupation],
    'gender': [gender],
    'hours-per-week': [hours_per_week],
    'native-country': [native_country]
})

st.markdown("### 📌 Input Summary")
st.dataframe(input_df)

# Prediction
if st.button("🚀 Predict Salary Class"):
    prediction = model.predict(input_df)[0]
    st.success(f"🎯 Predicted Salary Class: {prediction}")

# Batch prediction
st.markdown("---")
st.markdown("### 📁 Upload CSV for Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    st.write("🧾 Uploaded Data Preview:")
    st.dataframe(batch_df.head())

    try:
        preds = model.predict(batch_df)
        batch_df['Predicted Salary'] = preds
        st.success("✅ Batch prediction complete!")
        st.write(batch_df.head())

        csv = batch_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Predictions", csv, "batch_predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #adb5bd;'>Made with ❤️ by ishu</p>", unsafe_allow_html=True)
# Footer
st.markdown("<p style='text-align: center; color: #adb5bd;'>© 2025 Employee Salary Prediction App</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #adb5bd;'>This app is for educational purposes only.</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #adb5bd;'>Contact:abc@gmail.com <a href='mailto:abc@gmail.com'>",unsafe_allow_html=True)  
