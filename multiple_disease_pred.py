import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO

# Set page configuration
st.set_page_config(page_title="Health Assistant", layout="wide", page_icon="🧑‍⚕️")

# Initialize session state
if 'language' not in st.session_state:
    st.session_state.language = 'English'
if 'theme' not in st.session_state:
    st.session_state.theme = 'Dark'

# ------------------------------------------------
# 🌐 Multilingual Support
# ------------------------------------------------
translations = {
    'English': {
        'title': "Multiple Disease Prediction System",
        'welcome': "Welcome to Health Assistant Dashboard",
        'predict_diseases': "This system can predict your risk of the following diseases:",
        'diabetes': "Diabetes",
        'heart_disease': "Heart Disease",
        'parkinsons': "Parkinson’s Disease",
        'enter_details': "Enter your medical details in the respective section and get instant insights along with personalized health tips!",
        'tip': "Tip: Regular exercise, a healthy diet, and enough sleep can prevent many chronic diseases.",
        'history': "Your Prediction History",
        'no_history': "No predictions saved yet.",
        'about_title': "About Health Assistant",
        'about_desc': "Health Assistant is an advanced, user-friendly tool designed to predict the risk of chronic diseases using machine learning models.",
        'diabetes_pred': "Diabetes Prediction using ML",
        'heart_pred': "Heart Disease Prediction using ML",
        'parkinsons_pred': "Parkinson's Disease Prediction using ML",
        'bmi_title': "Body Mass Index (BMI) Calculator",
        'bmi_desc': "Calculate your BMI to understand your body weight status.",
        'feedback_title': "Feedback & Support",
        'feedback_desc': "We value your input! Please share your feedback or report any issues.",
        'submit_feedback': "Submit Feedback",
        'feedback_success': "Thank you for your feedback!",
        'download_report': "Download Prediction Report",
        'input_order': "Enter the following in order:",
        'health_insights': "Health Insights",
        'risk_visualization': "Risk Visualization",
        'error_input': "Please enter valid numerical values for all fields.",
        'no_probability': "Probability estimates are not available for this model.",
        'What does MDVP:Fo(Hz) mean?': "What does MDVP:Fo(Hz) mean?",
        'It’s the fundamental frequency of your voice, used in Parkinson’s prediction.': "It’s the fundamental frequency of your voice, used in Parkinson’s prediction.",
        'How accurate are the predictions?': "How accurate are the predictions?",
        'The models provide estimates based on input data. Always consult a doctor for medical advice.': "The models provide estimates based on input data. Always consult a doctor for medical advice.",
        'How do I save my results?': "How do I save my results?",
        'Use the "Download Prediction Report" button after each prediction.': "Use the 'Download Prediction Report' button after each prediction.",
        'You have a': "You have a",
        'risk of Diabetes.': "risk of Diabetes.",
        'You are healthy! Low risk of Diabetes.': "You are healthy! Low risk of Diabetes.",
        'You have a significant risk of Diabetes.': "You have a significant risk of Diabetes.",
        'risk of Heart Disease.': "risk of Heart Disease.",
        'You are healthy! Low risk of Heart Disease.': "You are healthy! Low risk of Heart Disease.",
        'You have a significant risk of Heart Disease.': "You have a significant risk of Heart Disease.",
        'risk of Parkinson’s Disease.': "risk of Parkinson’s Disease.",
        'You are healthy! Low risk of Parkinson’s.': "You are healthy! Low risk of Parkinson’s.",
        'You have a significant risk of Parkinson’s Disease.': "You have a significant risk of Parkinson’s Disease.",
        'Diabetes Test Result': "Diabetes Test Result",
        'Heart Disease Test Result': "Heart Disease Test Result",
        'Parkinson’s Test Result': "Parkinson’s Test Result",
        'Analyzing your data...': "Analyzing your data...",
        'Risk Probability Chart': "Risk Probability Chart",
        'Probability (%)': "Probability (%)",
        'Diabetes Risk': "Diabetes Risk",
        'No Diabetes': "No Diabetes",
        'Diabetes Risk Probability': "Diabetes Risk Probability",
        'Heart Disease Risk': "Heart Disease Risk",
        'No Heart Disease': "No Heart Disease",
        'Heart Disease Risk Probability': "Heart Disease Risk Probability",
        'Parkinson’s Risk': "Parkinson’s Risk",
        'No Parkinson’s': "No Parkinson’s",
        'Parkinson’s Risk Probability': "Parkinson’s Risk Probability",
        'Please enter the value': "Please enter the value"
    },
    'Hindi': {
        'title': "मल्टीपल डिजीज प्रेडिक्शन सिस्टम",
        'welcome': "हेल्थ असिस्टेंट डैशबोर्ड में आपका स्वागत है",
        'predict_diseases': "यह सिस्टम निम्नलिखित बीमारियों के जोखिम की भविष्यवाणी कर सकता है:",
        'diabetes': "मधुमेह",
        'heart_disease': "हृदय रोग",
        'parkinsons': "पार्किंसंस रोग",
        'enter_details': "अपने चिकित्सा विवरण को संबंधित अनुभाग में दर्ज करें और तत्काल जानकारी के साथ व्यक्तिगत स्वास्थ्य सुझाव प्राप्त करें!",
        'tip': "सुझाव: नियमित व्यायाम, स्वस्थ आहार और पर्याप्त नींद कई पुरानी बीमारियों को रोक सकती है।",
        'history': "आपका प्रेडिक्शन इतिहास",
        'no_history': "अभी तक कोई प्रेडिक्शन सहेजा नहीं गया।",
        'about_title': "हेल्थ असिस्टेंट के बारे में",
        'about_desc': "हेल्थ असिस्टेंट एक उन्नत, उपयोगकर्ता-अनुकूल उपकरण है जो मशीन लर्निंग मॉडल्स का उपयोग करके पुरानी बीमारियों के जोखिम की भविष्यवाणी करता है।",
        'diabetes_pred': "एमएल का उपयोग करके मधुमेह की भविष्यवाणी",
        'heart_pred': "एमएल का उपयोग करके हृदय रोग की भविष्यवाणी",
        'parkinsons_pred': "एमएल का उपयोग करके पार्किंसंस रोग की भविष्यवाणी",
        'bmi_title': "बॉडी मास इंडेक्स (बीएमआई) कैलकुलेटर",
        'bmi_desc': "अपने शरीर के वजन की स्थिति को समझने के लिए अपने बीएमआई की गणना करें।",
        'feedback_title': "प्रतिक्रिया और समर्थन",
        'feedback_desc': "हम आपकी राय को महत्व देते हैं! कृपया अपनी प्रतिक्रिया साझा करें या किसी समस्या की रिपोर्ट करें।",
        'submit_feedback': "प्रतिक्रिया सबमिट करें",
        'feedback_success': "आपकी प्रतिक्रिया के लिए धन्यवाद!",
        'download_report': "प्रेडिक्शन रिपोर्ट डाउनलोड करें",
        'input_order': "निम्नलिखित को क्रम में दर्ज करें:",
        'health_insights': "स्वास्थ्य जानकारी",
        'risk_visualization': "जोखिम विज़ुअलाइज़ेशन",
        'error_input': "कृपया सभी क्षेत्रों के लिए मान्य संख्यात्मक मान दर्ज करें।",
        'no_probability': "इस मॉडल के लिए संभावना अनुमान उपलब्ध नहीं हैं।",
        'What does MDVP:Fo(Hz) mean?': "MDVP:Fo(Hz) का क्या अर्थ है?",
        'It’s the fundamental frequency of your voice, used in Parkinson’s prediction.': "यह आपकी आवाज की मूल आवृत्ति है, जिसका उपयोग पार्किंसंस की भविष्यवाणी में किया जाता है।",
        'How accurate are the predictions?': "भविष्यवाणियां कितनी सटीक हैं?",
        'The models provide estimates based on input data. Always consult a doctor for medical advice.': "मॉडल इनपुट डेटा के आधार पर अनुमान प्रदान करते हैं। हमेशा चिकित्सीय सलाह के लिए डॉक्टर से परामर्श करें।",
        'How do I save my results?': "मैं अपने परिणाम कैसे सहेजूं?",
        'Use the "Download Prediction Report" button after each prediction.': "प्रत्येक भविष्यवाणी के बाद 'प्रेडिक्शन रिपोर्ट डाउनलोड करें' बटन का उपयोग करें।",
        'You have a': "आपको है",
        'risk of Diabetes.': "मधुमेह का जोखिम।",
        'You are healthy! Low risk of Diabetes.': "आप स्वस्थ हैं! मधुमेह का कम जोखिम।",
        'You have a significant risk of Diabetes.': "आपको मधुमेह का महत्वपूर्ण जोखिम है।",
        'risk of Heart Disease.': "हृदय रोग का जोखिम।",
        'You are healthy! Low risk of Heart Disease.': "आप स्वस्थ हैं! हृदय रोग का कम जोखिम।",
        'You have a significant risk of Heart Disease.': "आपको हृदय रोग का महत्वपूर्ण जोखिम है।",
        'risk of Parkinson’s Disease.': "पार्किंसंस रोग का जोखिम।",
        'You are healthy! Low risk of Parkinson’s.': "आप स्वस्थ हैं! पार्किंसंस का कम जोखिम।",
        'You have a significant risk of Parkinson’s Disease.': "आपको पार्किंसंस रोग का महत्वपूर्ण जोखिम है।",
        'Diabetes Test Result': "मधुमेह परीक्षण परिणाम",
        'Heart Disease Test Result': "हृदय रोग परीक्षण परिणाम",
        'Parkinson’s Test Result': "पार्किंसंस परीक्षण परिणाम",
        'Analyzing your data...': "आपके डेटा का विश्लेषण किया जा रहा है...",
        'Risk Probability Chart': "जोखिम संभावना चार्ट",
        'Probability (%)': "संभावना (%)",
        'Diabetes Risk': "मधुमेह जोखिम",
        'No Diabetes': "कोई मधुमेह नहीं",
        'Diabetes Risk Probability': "मधुमेह जोखिम संभावना",
        'Heart Disease Risk': "हृदय रोग जोखिम",
        'No Heart Disease': "कोई हृदय रोग नहीं",
        'Heart Disease Risk Probability': "हृदय रोग जोखिम संभावना",
        'Parkinson’s Risk': "पार्किंसंस जोखिम",
        'No Parkinson’s': "कोई पार्किंसंस नहीं",
        'Parkinson’s Risk Probability': "पार्किंसंस जोखिम संभावना",
        'Please enter the value': "कृपया मूल्य दर्ज करें"
    },
    'Tamil': {
        'title': "பல நோய் கணிப்பு அமைப்பு",
        'welcome': "ஹெல்த் அசிஸ்டன்ட் டாஷ்போர்டுக்கு வரவேற்கிறோம்",
        'predict_diseases': "இந்த அமைப்பு பின்வரும் நோய்களின் அபாயத்தை கணிக்க முடியும்:",
        'diabetes': "நீரிழிவு",
        'heart_disease': "இதய நோய்",
        'parkinsons': "பார்கின்சன் நோய்",
        'enter_details': "தொடர்புடைய பிரிவில் உங்கள் மருத்துவ விவரங்களை உள்ளிடவும் மற்றும் உடனடி தகவல்களுடன் தனிப்பயனாக்கப்பட்ட உடல்நல குறிப்புகளைப் பெறவும்!",
        'tip': "குறிப்பு: வழக்கமான உடற்பயிற்சி, ஆரோக்கியமான உணவு மற்றும் போதுமான தூக்கம் பல நாட்பட்ட நோய்களைத் தடுக்கலாம்.",
        'history': "உங்கள் கணிப்பு வரலாறு",
        'no_history': "இதுவரை எந்த கணிப்புகளும் சேமிக்கப்படவில்லை.",
        'about_title': "ஹெல்த் அசிஸ்டன்ட் பற்றி",
        'about_desc': "ஹெல்த் அசிஸ்டன்ட் என்பது இயந்திர கற்றல் மாதிரிகளைப் பயன்படுத்தி நாட்பட்ட நோய்களின் அபாயத்தை கணிக்க வடிவமைக்கப்பட்ட மேம்பட்ட, பயனர்-நட்பு கருவியாகும்.",
        'diabetes_pred': "எம்எல் மூலம் நீரிழிவு கணிப்பு",
        'heart_pred': "எம்எல் மூலம் இதய நோய் கணிப்பு",
        'parkinsons_pred': "எம்எல் மூலம் பார்கின்சன் நோய் கணிப்பு",
        'bmi_title': "உடல் நிறை குறியீட்டு (பிஎம்ஐ) கால்குலேட்டர்",
        'bmi_desc': "உங்கள் உடல் எடை நிலையைப் புரிந்துகொள்ள உங்கள் பிஎம்ஐ-ஐ கணக்கிடவும்.",
        'feedback_title': "கருத்து மற்றும் ஆதரவு",
        'feedback_desc': "உங்கள் கருத்தை நாங்கள் மதிக்கிறோம்! தயவுசெய்து உங்கள் கருத்தைப் பகிரவும் அல்லது ஏதேனும் சிக்கல்களைப் புகாரளிக்கவும்.",
        'submit_feedback': "கருத்தை சமர்ப்பிக்கவும்",
        'feedback_success': "உங்கள் கருத்துக்கு நன்றி!",
        'download_report': "கணிப்பு அறிக்கையைப் பதிவிறக்கவும்",
        'input_order': "பின்வருவனவற்றை வரிசையில் உள்ளிடவும்:",
        'health_insights': "உடல்நல தகவல்கள்",
        'risk_visualization': "அபாய காட்சிப்படுத்தல்",
        'error_input': "எல்லா புலங்களுக்கும் செல்லுபடியாகும் எண்ணியல் மதிப்புகளை உள்ளிடவும்.",
        'no_probability': "இந்த மாதிரிக்கு நிகழ்தகவு மதிப்பீடுகள் கிடைக்கவில்லை.",
        'What does MDVP:Fo(Hz) mean?': "MDVP:Fo(Hz) என்றால் என்ன?",
        'It’s the fundamental frequency of your voice, used in Parkinson’s prediction.': "இது உங்கள் குரலின் அடிப்படை அதிர்வெண், பார்கின்சன் கணிப்பில் பயன்படுத்தப்படுகிறது.",
        'How accurate are the predictions?': "கணிப்புகள் எவ்வளவு துல்லியமானவை?",
        'The models provide estimates based on input data. Always consult a doctor for medical advice.': "மாதிரிகள் உள்ளீட்டு தரவின் அடிப்படையில் மதிப்பீடுகளை வழங்குகின்றன. மருத்துவ ஆலோசனைக்கு எப்போதும் மருத்துவரை அணுகவும்.",
        'How do I save my results?': "எனது முடிவுகளை எவ்வாறு சேமிப்பது?",
        'Use the "Download Prediction Report" button after each prediction.': "ஒவ்வொரு கணிப்புக்குப் பிறகு 'கணிப்பு அறிக்கையைப் பதிவிறக்கவும்' பொத்தானைப் பயன்படுத்தவும்.",
        'You have a': "உங்களுக்கு",
        'risk of Diabetes.': "நீரிழிவு அபாயம்.",
        'You are healthy! Low risk of Diabetes.': "நீங்கள் ஆரோக்கியமாக உள்ளீர்கள்! நீரிழிவு அபாயம் குறைவு.",
        'You have a significant risk of Diabetes.': "உங்களுக்கு நீரிழிவு அபாயம் கணிசமாக உள்ளது.",
        'risk of Heart Disease.': "இதய நோய் அபாயம்.",
        'You are healthy! Low risk of Heart Disease.': "நீங்கள் ஆரோக்கியமாக உள்ளீர்கள்! இதய நோய் அபாயம் குறைவு.",
        'You have a significant risk of Heart Disease.': "உங்களுக்கு இதய நோய் அபாயம் கணிசமாக உள்ளது.",
        'risk of Parkinson’s Disease.': "பார்கின்சன் நோய் அபாயம்.",
        'You are healthy! Low risk of Parkinson’s.': "நீங்கள் ஆரோக்கியமாக உள்ளீர்கள்! பார்கின்சன் அபாயம் குறைவு.",
        'You have a significant risk of Parkinson’s Disease.': "உங்களுக்கு பார்கின்சன் நோய் அபாயம் கணிசமாக உள்ளது.",
        'Diabetes Test Result': "நீரிழிவு பரிசோதனை முடிவு",
        'Heart Disease Test Result': "இதய நோய் பரிசோதனை முடிவு",
        'Parkinson’s Test Result': "பார்கின்சன் பரிசோதனை முடிவு",
        'Analyzing your data...': "உங்கள் தரவை பகுப்பாய்வு செய்கிறது...",
        'Risk Probability Chart': "அபாய நிகழ்தகவு விளக்கப்படம்",
        'Probability (%)': "நிகழ்தகவு (%)",
        'Diabetes Risk': "நீரிழிவு அபாயம்",
        'No Diabetes': "நீரிழிவு இல்லை",
        'Diabetes Risk Probability': "நீரிழிவு அபாய நிகழ்தகவு",
        'Heart Disease Risk': "இதய நோய் அபாயம்",
        'No Heart Disease': "இதய நோய் இல்லை",
        'Heart Disease Risk Probability': "இதய நோய் அபாய நிகழ்தகவு",
        'Parkinson’s Risk': "பார்கின்சன் அபாயம்",
        'No Parkinson’s': "பார்கின்சன் இல்லை",
        'Parkinson’s Risk Probability': "பார்கின்சன் அபாய நிகழ்தகவு",
        'Please enter the value': "தயவு செய்து மதிப்பை உள்ளிடவும்"
    }
}

input_translations = {
    'English': {
        'Pregnancies': 'Pregnancies',
        'Glucose': 'Glucose',
        'Blood Pressure': 'Blood Pressure',
        'Skin Thickness': 'Skin Thickness',
        'Insulin': 'Insulin',
        'BMI': 'BMI',
        'Diabetes Pedigree Function': 'Diabetes Pedigree Function',
        'Age': 'Age',
        'Sex': 'Sex',
        'Chest Pain types': 'Chest Pain types',
        'Resting Blood Pressure': 'Resting Blood Pressure',
        'Serum Cholestoral': 'Serum Cholestoral',
        'Fasting Blood Sugar': 'Fasting Blood Sugar',
        'Resting ECG': 'Resting ECG',
        'Maximum Heart Rate': 'Maximum Heart Rate',
        'Exercise Induced Angina': 'Exercise Induced Angina',
        'ST depression': 'ST depression',
        'Slope of ST segment': 'Slope of ST segment',
        'Major vessels': 'Major vessels',
        'Thalassemia': 'Thalassemia',
        'MDVP:Fo(Hz)': 'MDVP:Fo(Hz)',
        'MDVP:Fhi(Hz)': 'MDVP:Fhi(Hz)',
        'MDVP:Flo(Hz)': 'MDVP:Flo(Hz)',
        'MDVP:Jitter(%)': 'MDVP:Jitter(%)',
        'MDVP:Jitter(Abs)': 'MDVP:Jitter(Abs)',
        'MDVP:RAP': 'MDVP:RAP',
        'MDVP:PPQ': 'MDVP:PPQ',
        'Jitter:DDP': 'Jitter:DDP',
        'MDVP:Shimmer': 'MDVP:Shimmer',
        'MDVP:Shimmer(dB)': 'MDVP:Shimmer(dB)',
        'Shimmer:APQ3': 'Shimmer:APQ3',
        'Shimmer:APQ5': 'Shimmer:APQ5',
        'MDVP:APQ': 'MDVP:APQ',
        'Shimmer:DDA': 'Shimmer:DDA',
        'NHR': 'NHR',
        'HNR': 'HNR',
        'RPDE': 'RPDE',
        'DFA': 'DFA',
        'spread1': 'spread1',
        'spread2': 'spread2',
        'D2': 'D2',
        'PPE': 'PPE',
        'Height (cm):': 'Height (cm):',
        'Weight (kg):': 'Weight (kg):',
        'Calculate BMI': 'Calculate BMI',
        'Your BMI is:': 'Your BMI is:',
        'You are underweight. Eat nutrient-rich foods 🍞🍗': 'You are underweight. Eat nutrient-rich foods 🍞🍗',
        'You are healthy! Keep maintaining a balanced diet 🥗': 'You are healthy! Keep maintaining a balanced diet 🥗',
        'You are overweight. Exercise regularly 🏃‍♂️': 'You are overweight. Exercise regularly 🏃‍♂️',
        'You are obese. Please consult a doctor and plan weight management ⚕️': 'You are obese. Please consult a doctor and plan weight management ⚕️'
    },
    'Hindi': {
        'Pregnancies': 'गर्भावस्था',
        'Glucose': 'ग्लूकोज',
        'Blood Pressure': 'रक्तचाप',
        'Skin Thickness': 'त्वचा की मोटाई',
        'Insulin': 'इंसुलिन',
        'BMI': 'बीएमआई',
        'Diabetes Pedigree Function': 'मधुमेह वंशावली कार्य',
        'Age': 'आयु',
        'Sex': 'लिंग',
        'Chest Pain types': 'सीने में दर्द के प्रकार',
        'Resting Blood Pressure': 'विश्राम रक्तचाप',
        'Serum Cholestoral': 'सीरम कोलेस्ट्रॉल',
        'Fasting Blood Sugar': 'उपवास रक्त शर्करा',
        'Resting ECG': 'विश्राम ईसीजी',
        'Maximum Heart Rate': 'अधिकतम हृदय गति',
        'Exercise Induced Angina': 'व्यायाम प्रेरित एनजाइना',
        'ST depression': 'एसटी अवसाद',
        'Slope of ST segment': 'एसटी खंड का ढलान',
        'Major vessels': 'प्रमुख रक्त वाहिकाएं',
        'Thalassemia': 'थैलेसीमिया',
        'MDVP:Fo(Hz)': 'एमडीवीपी:एफओ(हर्ट्ज)',
        'MDVP:Fhi(Hz)': 'एमडीवीपी:एफएचआई(हर्ट्ज)',
        'MDVP:Flo(Hz)': 'एमडीवीपी:एफएलओ(हर्ट्ज)',
        'MDVP:Jitter(%)': 'एमडीवीपी:जिटर(%)',
        'MDVP:Jitter(Abs)': 'एमडीवीपी:जिटर(एब्स)',
        'MDVP:RAP': 'एमडीवीपी:रैप',
        'MDVP:PPQ': 'एमडीवीपी:पीपीक्यू',
        'Jitter:DDP': 'जिटर:डीडीपी',
        'MDVP:Shimmer': 'एमडीवीपी:शिमर',
        'MDVP:Shimmer(dB)': 'एमडीवीपी:शिमर(डीबी)',
        'Shimmer:APQ3': 'शिमर:एपीक्यू3',
        'Shimmer:APQ5': 'शिमर:एपीक्यू5',
        'MDVP:APQ': 'एमडीवीपी:एपीक्यू',
        'Shimmer:DDA': 'शिमर:डीडीए',
        'NHR': 'एनएचआर',
        'HNR': 'एचएनआर',
        'RPDE': 'आरपीडीई',
        'DFA': 'डीएफए',
        'spread1': 'स्प्रेड1',
        'spread2': 'स्प्रेड2',
        'D2': 'डी2',
        'PPE': 'पीपीई',
        'Height (cm):': 'ऊंचाई (सेमी):',
        'Weight (kg):': 'वजन (किग्रा):',
        'Calculate BMI': 'बीएमआई की गणना करें',
        'Your BMI is:': 'आपका बीएमआई है:',
        'You are underweight. Eat nutrient-rich foods 🍞🍗': 'आपका वजन कम है। पोषक तत्वों से भरपूर भोजन करें 🍞🍗',
        'You are healthy! Keep maintaining a balanced diet 🥗': 'आप स्वस्थ हैं! संतुलित आहार बनाए रखें 🥗',
        'You are overweight. Exercise regularly 🏃‍♂️': 'आपका वजन अधिक है। नियमित व्यायाम करें 🏃‍♂️',
        'You are obese. Please consult a doctor and plan weight management ⚕️': 'आप मोटापे से ग्रस्त हैं। कृपया डॉक्टर से परामर्श करें और वजन प्रबंधन की योजना बनाएं ⚕️'
    },
    'Tamil': {
        'Pregnancies': 'கர்ப்பங்கள்',
        'Glucose': 'குளுக்கோஸ்',
        'Blood Pressure': 'இரத்த அழுத்தம்',
        'Skin Thickness': 'தோல் தடிமன்',
        'Insulin': 'இன்சுலின்',
        'BMI': 'பிஎம்ஐ',
        'Diabetes Pedigree Function': 'நீரிழிவு வம்சாவளி செயல்பாடு',
        'Age': 'வயது',
        'Sex': 'பாலினம்',
        'Chest Pain types': 'மார்பு வலி வகைகள்',
        'Resting Blood Pressure': 'ஓய்வு இரத்த அழுத்தம்',
        'Serum Cholestoral': 'சீரம் கொலஸ்ட்ரால்',
        'Fasting Blood Sugar': 'நோன்பு இரத்த சர்க்கரை',
        'Resting ECG': 'ஓய்வு இசிஜி',
        'Maximum Heart Rate': 'அதிகபட்ச இதய துடிப்பு',
        'Exercise Induced Angina': 'பயிற்சியால் தூண்டப்பட்ட ஆஞ்சினா',
        'ST depression': 'எஸ்டி மனச்சோர்வு',
        'Slope of ST segment': 'எஸ்டி பிரிவின் சாய்வு',
        'Major vessels': 'முக்கிய இரத்த நாளங்கள்',
        'Thalassemia': 'தலசீமியா',
        'MDVP:Fo(Hz)': 'எம்டிவிபி:எஃப்ஒ(ஹெர்ட்ஸ்)',
        'MDVP:Fhi(Hz)': 'எம்டிவிபி:எஃப்எச்ஐ(ஹெர்ட்ஸ்)',
        'MDVP:Flo(Hz)': 'எம்டிவிபி:எஃப்எல்ஒ(ஹெர்ட்ஸ்)',
        'MDVP:Jitter(%)': 'எம்டிவிபி:ஜிட்டர்(%)',
        'MDVP:Jitter(Abs)': 'எம்டிவிபி:ஜிட்டர்(முழுமையான)',
        'MDVP:RAP': 'எம்டிவிபி:ஆர்ஏபி',
        'MDVP:PPQ': 'எம்டிவிபி:பிபிக்யூ',
        'Jitter:DDP': 'ஜிட்டர்:டிடிபி',
        'MDVP:Shimmer': 'எம்டிவிபி:ஷிம்மர்',
        'MDVP:Shimmer(dB)': 'எம்டிவிபி:ஷிம்மர்(டிபி)',
        'Shimmer:APQ3': 'ஷிம்மர்:ஏபிக்யூ3',
        'Shimmer:APQ5': 'ஷிம்மர்:ஏபிக்யூ5',
        'MDVP:APQ': 'எம்டிவிபி:ஏபிக்யூ',
        'Shimmer:DDA': 'ஷிம்மர்:டிடிஏ',
        'NHR': 'என்எச்ஆர்',
        'HNR': 'எச்என்ஆர்',
        'RPDE': 'ஆர்பிடிஇ',
        'DFA': 'டிஎஃப்ஏ',
        'spread1': 'ஸ்ப்ரெட்1',
        'spread2': 'ஸ்ப்ரெட்2',
        'D2': 'டி2',
        'PPE': 'பிபிஇ',
        'Height (cm):': 'உயரம் (செ.மீ):',
        'Weight (kg):': 'எடை (கிலோ):',
        'Calculate BMI': 'பிஎம்ஐ கணக்கிடவும்',
        'Your BMI is:': 'உங்கள் பிஎம்ஐ:',
        'You are underweight. Eat nutrient-rich foods 🍞🍗': 'உங்கள் எடை குறைவாக உள்ளது. ஊட்டச்சத்து நிறைந்த உணவுகளை உண்ணவும் 🍞🍗',
        'You are healthy! Keep maintaining a balanced diet 🥗': 'நீங்கள் ஆரோக்கியமாக உள்ளீர்கள்! சமநிலையான உணவை பராமரிக்கவும் 🥗',
        'You are overweight. Exercise regularly 🏃‍♂️': 'உங்கள் எடை அதிகமாக உள்ளது. தவறாமல் உடற்பயிற்சி செய்யவும் 🏃‍♂️',
        'You are obese. Please consult a doctor and plan weight management ⚕️': 'நீங்கள் உடல் பருமனாக உள்ளீர்கள். மருத்துவரை அணுகி எடை மேலாண்மை திட்டமிடவும் ⚕️'
    }
}

insights_translations = {
    'English': {
        'high_glucose': "High Glucose Risk: Glucose levels above 140 mg/dL may indicate prediabetes or diabetes risk. Follow a low-carb diet, focusing on vegetables, whole grains, and lean proteins. Monitor glucose regularly and consult a doctor for screenings.",
        'obesity_risk': "Obesity Risk: BMI above 30 increases diabetes risk. Aim to lose 5-10% of body weight through 150 min/week of moderate exercise (e.g., brisk walking) and a balanced diet with controlled portions.",
        'age_diabetes_risk': "Age-Related Risk: Age above 45 is a risk factor for type 2 diabetes. Get annual screenings like fasting glucose or A1C tests and maintain a healthy lifestyle.",
        'pregnancy_risk': "Pregnancy History: Multiple pregnancies (>3) may increase diabetes risk. Ensure regular check-ups and manage weight post-pregnancy.",
        'diabetes_high_risk': "High Risk: You have a significant risk of diabetes. Prioritize 150 min/week of moderate exercise, reduce sugar intake, and consult a healthcare provider for a personalized plan.",
        'diabetes_low_risk': "Low Risk: Your risk is low. Maintain healthy habits: eat fiber-rich foods, exercise regularly, and monitor glucose periodically.",
        'diabetes_general_tip': "General Tip: Stay hydrated, manage stress with mindfulness or yoga, and get 7-9 hours of sleep to support insulin sensitivity.",
        'hypertension_risk': "Hypertension Risk: Resting BP above 140 mmHg indicates high blood pressure. Reduce salt intake (<1,500 mg/day), exercise 150 min/week, and check BP regularly.",
        'high_cholesterol': "High Cholesterol: Cholesterol above 240 mg/dL increases heart disease risk. Adopt a Mediterranean diet with olive oil, fish, and nuts, and avoid trans fats.",
        'angina_warning': "Angina Warning: Exercise-induced angina suggests heart strain. Avoid intense physical exertion and consult a cardiologist for evaluation.",
        'age_heart_risk': "Age-Related Risk: Age above 55 increases heart disease risk. Get regular cholesterol and BP screenings and maintain an active lifestyle.",
        'heart_high_risk': "High Risk: You have a significant risk of heart disease. Quit smoking, limit alcohol (1-2 drinks/day), and follow a heart-healthy diet. Consult a doctor for tests like ECG or stress testing.",
        'heart_low_risk': "Low Risk: Your risk is low. Continue heart-healthy habits: 150 min/week aerobic exercise, low-sodium diet, and stress management.",
        'heart_general_tip': "General Tip: Practice deep breathing or meditation to reduce stress and aim for 7-9 hours of sleep to support heart health.",
        'voice_changes': "Voice Changes: Low fundamental frequency (<100 Hz) may indicate vocal issues linked to Parkinson’s. Engage in speech therapy and practice vocal exercises daily.",
        'vocal_instability': "Vocal Instability: High jitter (>0.05%) suggests vocal tremors, a potential Parkinson’s symptom. Consult a neurologist and consider speech therapy.",
        'parkinsons_high_risk': "High Risk: You have a significant risk of Parkinson’s. Start physical therapy for mobility, engage in 150 min/week of exercise like tai chi, and seek a neurological evaluation.",
        'parkinsons_low_risk': "Low Risk: Your risk is low. Stay active with coordination-focused exercises (e.g., yoga) and monitor for symptoms like tremors.",
        'parkinsons_general_tip': "General Tip: Follow a Mediterranean diet, stay socially active, and adapt your home (e.g., remove rugs) to prevent falls."
    },
    'Hindi': {
        'high_glucose': "उच्च ग्लूकोज जोखिम: 140 mg/dL से ऊपर ग्लूकोज स्तर प्री-डायबिटीज या डायबिटीज के जोखिम को दर्शा सकते हैं। कम कार्ब आहार का पालन करें, सब्जियों, साबुत अनाज और लीन प्रोटीन पर ध्यान दें। नियमित रूप से ग्लूकोज की निगरानी करें और स्क्रीनिंग के लिए डॉक्टर से परामर्श करें।",
        'obesity_risk': "मोटापा जोखिम: 30 से ऊपर बीएमआई डायबिटीज के जोखिम को बढ़ाता है। प्रति सप्ताह 150 मिनट मध्यम व्यायाम (जैसे तेज चलना) और संतुलित आहार के साथ 5-10% वजन कम करने का लक्ष्य रखें।",
        'age_diabetes_risk': "आयु-संबंधी जोखिम: 45 वर्ष से अधिक आयु टाइप 2 डायबिटीज के लिए जोखिम कारक है। उपवास ग्लूकोज या A1C परीक्षण जैसे वार्षिक स्क्रीनिंग करें और स्वस्थ जीवनशैली बनाए रखें।",
        'pregnancy_risk': "गर्भावस्था इतिहास: एक से अधिक गर्भावस्थाएं (>3) डायबिटीज जोखिम को बढ़ा सकती हैं। नियमित जांच करें और गर्भावस्था के बाद वजन प्रबंधन करें।",
        'diabetes_high_risk': "उच्च जोखिम: आपके पास डायबिटीज का महत्वपूर्ण जोखिम है। प्रति सप्ताह 150 मिनट मध्यम व्यायाम को प्राथमिकता दें, चीनी का सेवन कम करें, और व्यक्तिगत योजना के लिए स्वास्थ्य सेवा प्रदाता से परामर्श करें।",
        'diabetes_low_risk': "कम जोखिम: आपका जोखिम कम है। स्वस्थ आदतें बनाए रखें: फाइबर युक्त भोजन करें, नियमित व्यायाम करें, और समय-समय पर ग्लूकोज की निगरानी करें।",
        'diabetes_general_tip': "सामान्य सुझाव: हाइड्रेटेड रहें, माइंडफुलनेस या योग के साथ तनाव प्रबंधन करें, और इंसुलिन संवेदनशीलता का समर्थन करने के लिए 7-9 घंटे की नींद लें।",
        'hypertension_risk': "उच्च रक्तचाप जोखिम: 140 mmHg से ऊपर विश्राम बीपी उच्च रक्तचाप को दर्शाता है। नमक का सेवन कम करें (<1,500 mg/दिन), प्रति सप्ताह 150 मिनट व्यायाम करें, और नियमित रूप से बीपी की जांच करें।",
        'high_cholesterol': "उच्च कोलेस्ट्रॉल: 240 mg/dL से ऊपर कोलेस्ट्रॉल हृदय रोग जोखिम को बढ़ाता है। जैतून का तेल, मछली और नट्स के साथ भूमध्यसागरीय आहार अपनाएं, और ट्रांस वसा से बचें।",
        'angina_warning': "एनजाइना चेतावनी: व्यायाम-प्रेरित एनजाइना हृदय तनाव का सुझाव देता है। तीव्र शारीरिक परिश्रम से बचें और मूल्यांकन के लिए हृदय रोग विशेषज्ञ से परामर्श करें।",
        'age_heart_risk': "आयु-संबंधी जोखिम: 55 वर्ष से अधिक आयु हृदय रोग जोखिम को बढ़ाती है। नियमित कोलेस्ट्रॉल और बीपी स्क्रीनिंग करें और सक्रिय जीवनशैली बनाए रखें।",
        'heart_high_risk': "उच्च जोखिम: आपके पास हृदय रोग का महत्वपूर्ण जोखिम है। धूम्रपान छोड़ें, शराब सीमित करें (1-2 ड्रिंक/दिन), और हृदय-स्वस्थ आहार का पालन करें। ईसीजी या स्ट्रेस टेस्टिंग जैसे परीक्षणों के लिए डॉक्टर से परामर्श करें।",
        'heart_low_risk': "कम जोखिम: आपका जोखिम कम है। हृदय-स्वस्थ आदतें जारी रखें: प्रति सप्ताह 150 मिनट एरोबिक व्यायाम, कम सोडियम आहार, और तनाव प्रबंधन।",
        'heart_general_tip': "सामान्य सुझाव: तनाव कम करने के लिए गहरी सांस या ध्यान का अभ्यास करें और हृदय स्वास्थ्य के लिए 7-9 घंटे की नींद लें।",
        'voice_changes': "आवाज परिवर्तन: कम मूल आवृत्ति (<100 Hz) पार्किंसंस से संबंधित वोकल समस्याओं को दर्शा सकती है। वाणी चिकित्सा में शामिल हों और दैनिक वोकल व्यायाम करें।",
        'vocal_instability': "वोकल अस्थिरता: उच्च जिटर (>0.05%) वोकल कंपन का सुझाव देता है, जो पार्किंसंस का संभावित लक्षण है। न्यूरोलॉजिस्ट से परामर्श करें और वाणी चिकित्सा पर विचार करें।",
        'parkinsons_high_risk': "उच्च जोखिम: आपके पास पार्किंसंस का महत्वपूर्ण जोखिम है। गतिशीलता के लिए भौतिक चिकित्सा शुरू करें, प्रति सप्ताह 150 मिनट ताइ ची जैसे व्यायाम करें, और न्यूरोलॉजिकल मूल्यांकन लें।",
        'parkinsons_low_risk': "कम जोखिम: आपका जोखिम कम है। समन्वय-केंद्रित व्यायाम (जैसे योग) के साथ सक्रिय रहें और कंपन जैसे लक्षणों की निगरानी करें।",
        'parkinsons_general_tip': "सामान्य सुझाव: भूमध्यसागरीय आहार का पालन करें, सामाजिक रूप से सक्रिय रहें, और गिरने से रोकने के लिए अपने घर को अनुकूलित करें (उदाहरण के लिए, गलीचे हटाएं)।"
    },
    'Tamil': {
        'high_glucose': "உயர் குளுக்கோஸ் அபாயம்: 140 mg/dL-ஐ தாண்டிய குளுக்கோஸ் அளவுகள் முன்-நீரிழிவு அல்லது நீரிழிவு அபாயத்தைக் குறிக்கலாம். குறைந்த கார்போஹைட்ரேட் உணவைப் பின்பற்றவும், காய்கறிகள், முழு தானியங்கள் மற்றும் மெலிந்த புரதங்களை மையமாகக் கொள்ளவும். தவறாமல் குளுக்கோஸை கண்காணிக்கவும் மற்றும் பரிசோதனைகளுக்கு மருத்துவரை அணுகவும்.",
        'obesity_risk': "உடல் பருமன் அபாயம்: 30-ஐ தாண்டிய பிஎம்ஐ நீரிழிவு அபாயத்தை அதிகரிக்கிறது. வாரத்திற்கு 150 நிமிட மிதமான உடற்பயிற்சி (எ.கா., வேகமாக நடப்பது) மற்றும் பகுதி கட்டுப்பாட்டுடன் கூடிய சமநிலையான உணவு மூலம் 5-10% உடல் எடையை குறைக்க இலக்கு வைக்கவும்.",
        'age_diabetes_risk': "வயது தொடர்பான அபாயம்: 45 வயதுக்கு மேல் உள்ளவர்களுக்கு டைப் 2 நீரிழிவு அபாய காரணியாக உள்ளது. நோன்பு குளுக்கோஸ் அல்லது A1C பரிசோதனைகள் போன்ற ஆண்டு பரிசோதனைகளை மேற்கொள்ளவும் மற்றும் ஆரோக்கியமான வாழ்க்கை முறையை பராமரிக்கவும்.",
        'pregnancy_risk': "கர்ப்ப வரலாறு: பல கர்ப்பங்கள் (>3) நீரிழிவு அபாயத்தை அதிகரிக்கலாம். தவறாமல் பரிசோதனைகள் செய்யவும் மற்றும் கர்ப்பத்திற்குப் பின் எடை மேலாண்மை செய்யவும்.",
        'diabetes_high_risk': "உயர் அபாயம்: உங்களுக்கு நீரிழிவு அபாயம் கணிசமாக உள்ளது. வாரத்திற்கு 150 நிமிட மிதமான உடற்பயிற்சிக்கு முன்னுரிமை கொடுக்கவும், சர்க்கரை உட்கொள்ளலை குறைக்கவும், மற்றும் தனிப்பயனாக்கப்பட்ட திட்டத்திற்கு மருத்துவரை அணுகவும்.",
        'diabetes_low_risk': "குறைந்த அபாயம்: உங்கள் அபாயம் குறைவாக உள்ளது. ஆரோக்கியமான பழக்கங்களை பராமரிக்கவும்: நார்ச்சத்து நிறைந்த உணவுகளை உண்ணவும், தவறாமல் உடற்பயிற்சி செய்யவும், மற்றும் அவ்வப்போது குளுக்கோஸை கண்காணிக்கவும்.",
        'diabetes_general_tip': "பொதுவான குறிப்பு: நீரேற்றமாக இருக்கவும், மனநிறைவு அல்லது யோகாவுடன் மன அழுத்தத்தை மேலாண்மை செய்யவும், மற்றும் இன்சுலின் உணர்திறனை ஆதரிக்க 7-9 மணிநேர தூக்கம் பெறவும்.",
        'hypertension_risk': "உயர் இரத்த அழுத்த அபாயம்: 140 mmHg-ஐ தாண்டிய ஓய்வு இரத்த அழுத்தம் உயர் இரத்த அழுத்தத்தைக் குறிக்கிறது. உப்பு உட்கொள்ளலை குறைக்கவும் (<1,500 mg/நாள்), வாரத்திற்கு 150 நிமிட உடற்பயிற்சி செய்யவும், மற்றும் தவறாமல் இரத்த அழுத்தத்தை சரிபார்க்கவும்.",
        'high_cholesterol': "உயர் கொலஸ்ட்ரால்: 240 mg/dL-ஐ தாண்டிய கொலஸ்ட்ரால் இதய நோய் அபாயத்தை அதிகரிக்கிறது. ஆலிவ் எண்ணெய், மீன் மற்றும் கொட்டைகள் கொண்ட மத்தியதரைக் கடல் உணவு முறையை பின்பற்றவும், மற்றும் டிரான்ஸ் கொழுப்புகளை தவிர்க்கவும்.",
        'angina_warning': "ஆஞ்சினா எச்சரிக்கை: பயிற்சியால் தூண்டப்பட்ட ஆஞ்சினா இதய அழுத்தத்தைக் குறிக்கிறது. தீவிர உடல் உழைப்பை தவிர்க்கவும் மற்றும் மதிப்பீட்டிற்கு இதயநோய் நிபுணரை அணுகவும்.",
        'age_heart_risk': "வயது தொடர்பான அபாயம்: 55 வயதுக்கு மேல் இதய நோய் அபாயத்தை அதிகரிக்கிறது. தவறாமல் கொலஸ்ட்ரால் மற்றும் இரத்த அழுத்த பரிசோதனைகளை மேற்கொள்ளவும் மற்றும் செயலில் உள்ள வாழ்க்கை முறையை பராமரிக்கவும்.",
        'heart_high_risk': "உயர் அபாயம்: உங்களுக்கு இதய நோய் அபாயம் கணிசமாக உள்ளது. புகைபிடிப்பதை நிறுத்தவும், மது அருந்துதலை கட்டுப்படுத்தவும் (1-2 பானங்கள்/நாள்), மற்றும் இதய-ஆரோக்கியமான உணவை பின்பற்றவும். இசிஜி அல்லது மன அழுத்த பரிசோதனை போன்ற பரிசோதனைகளுக்கு மருத்துவரை அணுகவும்.",
        'heart_low_risk': "குறைந்த அபாயம்: உங்கள் அபாயம் குறைவாக உள்ளது. இதய-ஆரோக்கியமான பழக்கங்களை தொடரவும்: வாரத்திற்கு 150 நிமிட ஏரோபிக் உடற்பயிற்சி, குறைந்த உப்பு உணவு, மற்றும் மன அழுத்த மேலாண்மை.",
        'heart_general_tip': "பொதுவான குறிப்பு: மன அழுத்தத்தை குறைக்க ஆழமான சுவாசம் அல்லது தியானத்தை பயிற்சி செய்யவும் மற்றும் இதய ஆரோக்கியத்தை ஆதரிக்க 7-9 மணிநேர தூக்கத்தை இலக்காகக் கொள்ளவும்.",
        'voice_changes': "குரல் மாற்றங்கள்: குறைந்த அடிப்படை அதிர்வெண் (<100 Hz) பார்கின்சனுடன் தொடர்புடைய குரல் பிரச்சினைகளைக் குறிக்கலாம். பேச்சு சிகிச்சையில் ஈடுபடவும் மற்றும் தினசரி குரல் பயிற்சிகளை மேற்கொள்ளவும்.",
        'vocal_instability': "குரல் அநிலைத்தன்மை: உயர் ஜிட்டர் (>0.05%) குரல் நடுக்கத்தைக் குறிக்கிறது, இது பார்கின்சனின் ஒரு சாத்தியமான அறிகுறியாகும். நரம்பியல் நிபுணரை அணுகவும் மற்றும் பேச்சு சிகிச்சையை கருத்தில் கொள்ளவும்.",
        'parkinsons_high_risk': "உயர் அபாயம்: உங்களுக்கு பார்கின்சன் அபாயம் கணிசமாக உள்ளது. இயக்கத்திற்கு உடல் சிகிச்சையை தொடங்கவும், வாரத்திற்கு 150 நிமிட தை சி போன்ற உடற்பயிற்சிகளில் ஈடுபடவும், மற்றும் நரம்பியல் மதிப்பீட்டை பெறவும்.",
        'parkinsons_low_risk': "குறைந்த அபாயம்: உங்கள் அபாயம் குறைவாக உள்ளது. ஒருங்கிணைப்பு மையப்படுத்தப்பட்ட உடற்பயிற்சிகளுடன் (எ.கா., யோகா) செயலில் இருக்கவும் மற்றும் நடுக்கம் போன்ற அறிகுறிகளை கண்காணிக்கவும்.",
        'parkinsons_general_tip': "பொதுவான குறிப்பு: மத்தியதரைக் கடல் உணவு முறையை பின்பற்றவும், சமூக ரீதியாக செயலில் இருக்கவும், மற்றும் விழுதலைத் தடுக்க உங்கள் வீட்டை மாற்றியமைக்கவும் (எ.கா., கம்பளங்களை அகற்றவும்)."
    }
}

def t(key):
    return translations[st.session_state.language].get(key, key)

def t_input(key):
    return input_translations[st.session_state.language].get(key, key)

def t_insight(key):
    return insights_translations[st.session_state.language].get(key, key)

# ------------------------------------------------
# 🎬 Lottie Animation Loader
# ------------------------------------------------
@st.cache_data(show_spinner=False)
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load animations
home_animation = load_lottieurl("https://lottie.host/7c7125a7-37ec-44ac-8b8d-11a9b988d3bb/cxtshyITFK.json")
diabetes_animation = load_lottieurl("https://lottie.host/e4d1c1b4-1a6d-4b16-bb28-548abebd7a94/VNnDL3lbvM.json")
heart_animation = load_lottieurl("https://lottie.host/3c4cb70b-8e8c-4c1a-9c5a-fefcbe4e056f/R2BzYbCjCI.json")
parkinson_animation = load_lottieurl("https://lottie.host/760cb2d4-16fa-4899-a53f-95b3c9f784d1/hPl6x8nG6u.json")

# ------------------------------------------------
# ⚙️ Load Models
# ------------------------------------------------
working_dir = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model_with_scaler(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data["model"], data["scaler"]

MODEL_DIR = os.path.join(working_dir, 'saved models')
# Load all models with scalers
diabetes_model, diabetes_scaler = load_model_with_scaler(os.path.join(MODEL_DIR, 'diabetes_model.sav'))
heart_model, heart_scaler = load_model_with_scaler(os.path.join(MODEL_DIR, 'heart_disease_model.sav'))
parkinson_model, parkinson_scaler = load_model_with_scaler(os.path.join(MODEL_DIR, 'parkinson_model.sav'))

# ------------------------------------------------
# 🌗 Theme Customization and Mobile Optimization
# ------------------------------------------------
light_theme = """
<style>
body, .stMarkdown, .stText, p, div, h1, h2, h3, h4, h5, h6 {
    color: #333333 !important;
    background-color: #f5f5f5 !important;
}
[data-testid="stSidebar"] {
    width: 270px;
    background-color: #e0e0e0 !important;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
}
.footer {
    text-align: center;
    color: #333333;
    margin-top: 50px;
    font-size: 13px;
}
.tooltip {
    position: relative;
    display: inline-block;
    cursor: pointer;
}
.tooltip .tooltiptext {
    visibility: hidden;
    width: 200px;
    background-color: #555;
    color: #fff;
    text-align: center;
    border-radius: 6px;
    padding: 5px;
    position: absolute;
    z-index: 1;
    bottom: 125%;
    left: 50%;
    margin-left: -100px;
    opacity: 0;
    transition: opacity 0.3s;
}
.tooltip:hover .tooltiptext {
    visibility: visible;
    opacity: 1;
}
@media (max-width: 600px) {
    [data-testid="stSidebar"] {
        width: 100%;
    }
    .stButton>button {
        width: 100%;
        font-size: 14px;
    }
    .tooltip .tooltiptext {
        width: 150px;
        margin-left: -75px;
    }
    .stNumberInput, .stSelectbox {
        font-size: 12px;
    }
    .stColumn {
        display: block;
        width: 100%;
    }
}
</style>
"""

dark_theme = """
<style>
body, .stMarkdown, .stText, p, div, h1, h2, h3, h4, h5, h6 {
    color: white !important;
    background-color: #1a1a1a !important;
}
[data-testid="stSidebar"] {
    width: 270px;
    background-color: #2c2c2c !important;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
}
.footer {
    text-align: center;
    color: white;
    margin-top: 50px;
    font-size: 13px;
}
.tooltip {
    position: relative;
    display: inline-block;
    cursor: pointer;
}
.tooltip .tooltiptext {
    visibility: hidden;
    width: 200px;
    background-color: #ddd;
    color: #333;
    text-align: center;
    border-radius: 6px;
    padding: 5px;
    position: absolute;
    z-index: 1;
    bottom: 125%;
    left: 50%;
    margin-left: -100px;
    opacity: 0;
    transition: opacity 0.3s;
}
.tooltip:hover .tooltiptext {
    visibility: visible;
    opacity: 1;
}
@media (max-width: 600px) {
    [data-testid="stSidebar"] {
        width: 100%;
    }
    .stButton>button {
        width: 100%;
        font-size: 14px;
    }
    .tooltip .tooltiptext {
        width: 150px;
        margin-left: -75px;
    }
    .stNumberInput, .stSelectbox {
        font-size: 12px;
    }
    .stColumn {
        display: block;
        width: 100%;
    }
}
</style>
"""

# Apply selected theme
if st.session_state.theme == 'Light':
    st.markdown(light_theme, unsafe_allow_html=True)
else:
    st.markdown(dark_theme, unsafe_allow_html=True)

# ------------------------------------------------
# 🩺 Health Insights Function
# ------------------------------------------------
def get_health_insights(disease, inputs, prediction):
    insights = []
    if disease == "Diabetes":
        pregnancies, glucose, bp, skin, insulin, bmi, dpf, age = inputs
        if glucose > 140:
            insights.append(t_insight('high_glucose'))
        if bmi > 30:
            insights.append(t_insight('obesity_risk'))
        if age > 45:
            insights.append(t_insight('age_diabetes_risk'))
        if pregnancies > 3:
            insights.append(t_insight('pregnancy_risk'))
        if prediction == 1:
            insights.append(t_insight('diabetes_high_risk'))
        else:
            insights.append(t_insight('diabetes_low_risk'))
        insights.append(t_insight('diabetes_general_tip'))
    elif disease == "Heart Disease":
        age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal = inputs
        if trestbps > 140:
            insights.append(t_insight('hypertension_risk'))
        if chol > 240:
            insights.append(t_insight('high_cholesterol'))
        if exang == 1:
            insights.append(t_insight('angina_warning'))
        if age > 55:
            insights.append(t_insight('age_heart_risk'))
        if prediction == 1:
            insights.append(t_insight('heart_high_risk'))
        else:
            insights.append(t_insight('heart_low_risk'))
        insights.append(t_insight('heart_general_tip'))
    elif disease == "Parkinsons":
        fo = inputs[0]
        jitter = inputs[3]
        if fo < 100:
            insights.append(t_insight('voice_changes'))
        if jitter > 0.05:
            insights.append(t_insight('vocal_instability'))
        if prediction == 1:
            insights.append(t_insight('parkinsons_high_risk'))
        else:
            insights.append(t_insight('parkinsons_low_risk'))
        insights.append(t_insight('parkinsons_general_tip'))
    return insights

# ------------------------------------------------
# 📊 Generate PDF Report
# ------------------------------------------------
def generate_pdf_report(disease, inputs, diagnosis, insights):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("Helvetica", 12)
    c.drawString(100, 750, f"Health Assistant - {t(disease)} Prediction Report")
    c.drawString(100, 730, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(100, 710, t("Prediction Result:"))
    c.drawString(120, 690, diagnosis)
    c.drawString(100, 670, t("Health Insights:"))
    y = 650
    for insight in insights:
        c.drawString(120, y, insight[:80])  # Truncate for simplicity
        y -= 20
    c.drawString(100, y-20, t("Inputs Provided:"))
    if disease == "Diabetes":
        labels = ["Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness", "Insulin", "BMI", "Diabetes Pedigree Function", "Age"]
        for i, (label, value) in enumerate(zip(labels, inputs)):
            c.drawString(120, y-40-i*20, f"{t_input(label)}: {value}")
    elif disease == "Heart Disease":
        labels = ["Age", "Sex", "Chest Pain types", "Resting Blood Pressure", "Serum Cholestoral", "Fasting Blood Sugar", "Resting ECG", "Maximum Heart Rate", "Exercise Induced Angina", "ST depression", "Slope of ST segment", "Major vessels", "Thalassemia"]
        for i, (label, value) in enumerate(zip(labels, inputs)):
            c.drawString(120, y-40-i*20, f"{t_input(label)}: {value}")
    elif disease == "Parkinsons":
        labels = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"]
        for i, (label, value) in enumerate(zip(labels, inputs)):
            c.drawString(120, y-40-i*20, f"{t_input(label)}: {value}")
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# ------------------------------------------------
# 🧭 Sidebar Menu
# ------------------------------------------------
with st.sidebar:
    # Language selection
    language = st.selectbox("Select Language / भाषा चुनें / மொழியைத் தேர்ந்தெடு", ["English", "Hindi", "Tamil"])
    if language != st.session_state.language:
        st.session_state.language = language
        st.rerun()
    
    # Theme selection
    theme_choice = st.selectbox(t("Select Theme"), ["Dark", "Light"], index=0 if st.session_state.theme == 'Dark' else 1)
    if theme_choice != st.session_state.theme:
        st.session_state.theme = theme_choice
        st.rerun()
    
    selected = option_menu(
        t('title'),
        ['Home', 'About', 'BMI Calculator', 'Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction', 'Feedback'],
        icons=['house', 'info-circle', 'calculator', 'activity', 'heart', 'brain', 'envelope'],
        default_index=0
    )

# ------------------------------------------------
# 🏠 Home Page
# ------------------------------------------------
if selected == "Home":
    st.title(t("title"))
    st.subheader(t("welcome"))
    if home_animation:
        st_lottie(home_animation, height=300, key="home")
    st.markdown(f"""
    <h4>{t('predict_diseases')}</h4>
    <ul>
        <li>🩸 {t('diabetes')}</li>
        <li>❤️ {t('heart_disease')}</li>
        <li>🧠 {t('parkinsons')}</li>
    </ul>
    <p>💡 {t('enter_details')}</p>
    <p>✅ {t('tip')}</p>
    <h4>{t('history')}</h4>
    <p>{t('no_history')}</p>
    """, unsafe_allow_html=True)

# ------------------------------------------------
# ℹ️ About Page
# ------------------------------------------------
elif selected == "About":
    st.title(t("about_title"))
    if home_animation:
        st_lottie(home_animation, height=200, key="about")
    st.markdown(f"""
    ### {t('about_title')}
    {t('about_desc')}
    - 🩸 **{t('diabetes')}**: Assesses risk based on factors like glucose levels and BMI.
    - ❤️ **{t('heart_disease')}**: Evaluates heart health using metrics like cholesterol and blood pressure.
    - 🧠 **{t('parkinsons')}**: Analyzes voice and motor-related features to detect early signs.

    ### Purpose
    The goal is to empower users with instant health insights, enabling early detection and proactive health management. The application includes a BMI calculator, visualizations, personalized health tips, and report generation.

    ### Technology
    - **Machine Learning Models**: Trained models for precise predictions.
    - **Streamlit**: Powers the interactive web interface.
    - **Lottie Animations**: Enhances user experience with engaging visuals.
    - **Chart.js**: Provides insightful visualizations of health data.
    - **Theme Customization**: Toggle between light and dark modes for accessibility.

    ### Resources
    - [CDC Diabetes Resources](https://www.cdc.gov/diabetes): Learn about diabetes prevention and management.
    - [American Heart Association](https://www.heart.org): Heart health tips and guidelines.
    - [Parkinson’s Foundation](https://www.parkinson.org): Support and information for Parkinson’s disease.
    - [Mayo Clinic](https://www.mayoclinic.org): General health advice and disease information.

    ### Credits
    - **Developer**: Built with ❤️ by a passionate developer to make healthcare accessible.
    - **Inspired by**: xAI's mission to advance human scientific discovery.
    - **Version**: 1.3 (October 2025)

    <p>💡 <b>Note</b>: This tool is for informational purposes only. Always consult a healthcare professional for medical advice.</p>
    """, unsafe_allow_html=True)

# ------------------------------------------------
# ⚖️ BMI Calculator
# ------------------------------------------------
elif selected == "BMI Calculator":
    st.title(t("bmi_title"))
    st.write(t("bmi_desc"))
    height = st.number_input(t_input("Height (cm):"), min_value=50.0, max_value=250.0, value=170.0)
    weight = st.number_input(t_input("Weight (kg):"), min_value=10.0, max_value=200.0, value=65.0)
    if st.button(t_input("Calculate BMI")):
        bmi = weight / ((height / 100) ** 2)
        st.subheader(f"{t_input('Your BMI is:')} {bmi:.2f}")
        if bmi < 18.5:
            st.warning(t_input("You are underweight. Eat nutrient-rich foods 🍞🍗"))
        elif 18.5 <= bmi <= 24.9:
            st.success(t_input("You are healthy! Keep maintaining a balanced diet 🥗"))
        elif 25 <= bmi <= 29.9:
            st.warning(t_input("You are overweight. Exercise regularly 🏃‍♂️"))
        else:
            st.error(t_input("You are obese. Please consult a doctor and plan weight management ⚕️"))

# ------------------------------------------------
# 💉 Diabetes Prediction
# ------------------------------------------------
elif selected == 'Diabetes Prediction':
    st.title(t('diabetes_pred'))
    if diabetes_animation:
        st_lottie(diabetes_animation, height=200, key="diabetes")
    
    st.markdown(f"### {t('input_order')}")
    st.markdown("""
    1. Pregnancies
    2. Glucose
    3. Blood Pressure
    4. Skin Thickness
    5. Insulin
    6. BMI
    7. Diabetes Pedigree Function
    8. Age
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="tooltip">1. {t_input("Pregnancies")}<span class="tooltiptext">{t("Number of times pregnant (0-20)")}</span></div>', unsafe_allow_html=True)
        Pregnancies = st.number_input(t_input("Pregnancies"), min_value=0, max_value=20, value=0, key="pregnancies")
        st.markdown(f'<div class="tooltip">2. {t_input("Glucose")}<span class="tooltiptext">{t("Plasma glucose concentration (0-300 mg/dL)")}</span></div>', unsafe_allow_html=True)
        Glucose = st.number_input(t_input("Glucose"), min_value=0, max_value=300, value=0, key="glucose")
        st.markdown(f'<div class="tooltip">3. {t_input("Blood Pressure")}<span class="tooltiptext">{t("Diastolic blood pressure (mmHg)")}</span></div>', unsafe_allow_html=True)
        BloodPressure = st.number_input(t_input("Blood Pressure"), min_value=0, max_value=200, value=0, key="bp")
    with col2:
        st.markdown(f'<div class="tooltip">4. {t_input("Skin Thickness")}<span class="tooltiptext">{t("Triceps skin fold thickness (mm)")}</span></div>', unsafe_allow_html=True)
        SkinThickness = st.number_input(t_input("Skin Thickness"), min_value=0, max_value=100, value=0, key="skin")
        st.markdown(f'<div class="tooltip">5. {t_input("Insulin")}<span class="tooltiptext">{t("2-Hour serum insulin (mu U/ml)")}</span></div>', unsafe_allow_html=True)
        Insulin = st.number_input(t_input("Insulin"), min_value=0, max_value=900, value=0, key="insulin")
        st.markdown(f'<div class="tooltip">6. {t_input("BMI")}<span class="tooltiptext">{t("Body Mass Index (weight in kg/(height in m)^2)")}</span></div>', unsafe_allow_html=True)
        BMI = st.number_input(t_input("BMI"), min_value=0.0, max_value=70.0, value=0.0, key="bmi")
    with col3:
        st.markdown(f'<div class="tooltip">7. {t_input("Diabetes Pedigree Function")}<span class="tooltiptext">{t("Diabetes pedigree function (0-2.5)")}</span></div>', unsafe_allow_html=True)
        DiabetesPedigreeFunction = st.number_input(t_input("Diabetes Pedigree Function"), min_value=0.0, max_value=2.5, value=0.0, key="dpf")
        st.markdown(f'<div class="tooltip">8. {t_input("Age")}<span class="tooltiptext">{t("Age in years (20-80)")}</span></div>', unsafe_allow_html=True)
        Age = st.number_input(t_input("Age"), min_value=0, max_value=80, value=0, key="age")

    if st.button(t("Diabetes Test Result")):
        input_data = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        if any(x < 0 for x in input_data) or Glucose > 300 or BloodPressure > 200 or Age > 80:
            st.error(t('error_input'))
        else:
            try:
               with st.spinner(t('Analyzing your data...')):
                    input_arr = np.asarray(input_data).reshape(1, -1)
                    input_scaled = diabetes_scaler.transform(input_arr)
                    prob = diabetes_model.predict_proba(input_scaled)[0][1] * 100

                    if prob >= 60:
                       diagnosis = f"{t('You have a')} {prob:.1f}% {t('risk of Diabetes.')}"
                       st.error(diagnosis)
                    else:
                        diagnosis = f"{t('You are healthy! Low risk of Diabetes.')} ({prob:.1f}% {t('risk')})"
                        st.success(diagnosis)
                    
                    # Generate health insights
                    insights = get_health_insights("Diabetes", input_data, 1 if probability >= 60 else 0)
                    st.subheader(t("health_insights"))
                    for insight in insights:
                        st.write(f"💡 {insight}")

                    # Generate and download PDF report
                    pdf_buffer = generate_pdf_report("Diabetes", input_data, diagnosis, insights)
                    st.download_button(
                        label=t("download_report"),
                        data=pdf_buffer,
                        file_name="diabetes_report.pdf",
                        mime="application/pdf"
                    )
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                print(f"Exception: {e}")

# ------------------------------------------------
# ❤️ Heart Disease Prediction
# ------------------------------------------------
elif selected == 'Heart Disease Prediction':
    st.title(t('heart_pred'))
    if heart_animation:
        st_lottie(heart_animation, height=200, key="heart")
    
    st.markdown(f"### {t('input_order')}")
    st.markdown("""
    1. Age
    2. Sex (1 = Male, 0 = Female)
    3. Chest Pain Type (0-3)
    4. Resting Blood Pressure
    5. Serum Cholesterol
    6. Fasting Blood Sugar (> 120 mg/dl, 1 = Yes, 0 = No)
    7. Resting ECG (0-2)
    8. Maximum Heart Rate
    9. Exercise Induced Angina (1 = Yes, 0 = No)
    10. ST Depression
    11. Slope of ST Segment (0-2)
    12. Number of Major Vessels (0-3)
    13. Thalassemia (0-3)
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="tooltip">1. {t_input("Age")}<span class="tooltiptext">{t("Age in years (20-80)")}</span></div>', unsafe_allow_html=True)
        Age = st.number_input(t_input("Age"), min_value=0, max_value=80, value=0, key="age_heart")
        st.markdown(f'<div class="tooltip">2. {t_input("Sex")}<span class="tooltiptext">{t("Sex (1 = Male, 0 = Female)")}</span></div>', unsafe_allow_html=True)
        Sex = st.number_input(t_input("Sex"), min_value=0, max_value=1, value=0, key="sex")
        st.markdown(f'<div class="tooltip">3. {t_input("Chest Pain types")}<span class="tooltiptext">{t("Chest Pain Type (0-3)")}</span></div>', unsafe_allow_html=True)
        ChestPainType = st.number_input(t_input("Chest Pain types"), min_value=0, max_value=3, value=0, key="cp")
    with col2:
        st.markdown(f'<div class="tooltip">4. {t_input("Resting Blood Pressure")}<span class="tooltiptext">{t("Resting BP (mmHg)")}</span></div>', unsafe_allow_html=True)
        RestingBP = st.number_input(t_input("Resting Blood Pressure"), min_value=0, max_value=200, value=0, key="trestbps")
        st.markdown(f'<div class="tooltip">5. {t_input("Serum Cholestoral")}<span class="tooltiptext">{t("Serum Cholesterol (mg/dl)")}</span></div>', unsafe_allow_html=True)
        Cholesterol = st.number_input(t_input("Serum Cholestoral"), min_value=0, max_value=600, value=0, key="chol")
        st.markdown(f'<div class="tooltip">6. {t_input("Fasting Blood Sugar")}<span class="tooltiptext">{t("Fasting BS (> 120 mg/dl, 1 = Yes, 0 = No)")}</span></div>', unsafe_allow_html=True)
        FastingBS = st.number_input(t_input("Fasting Blood Sugar"), min_value=0, max_value=1, value=0, key="fbs")
    with col3:
        st.markdown(f'<div class="tooltip">7. {t_input("Resting ECG")}<span class="tooltiptext">{t("Resting ECG (0-2)")}</span></div>', unsafe_allow_html=True)
        RestingECG = st.number_input(t_input("Resting ECG"), min_value=0, max_value=2, value=0, key="restecg")
        st.markdown(f'<div class="tooltip">8. {t_input("Maximum Heart Rate")}<span class="tooltiptext">{t("Maximum Heart Rate (bpm)")}</span></div>', unsafe_allow_html=True)
        MaxHeartRate = st.number_input(t_input("Maximum Heart Rate"), min_value=0, max_value=250, value=0, key="thalach")
        st.markdown(f'<div class="tooltip">9. {t_input("Exercise Induced Angina")}<span class="tooltiptext">{t("Exercise Angina (1 = Yes, 0 = No)")}</span></div>', unsafe_allow_html=True)
        ExerciseAngina = st.number_input(t_input("Exercise Induced Angina"), min_value=0, max_value=1, value=0, key="exang")
    with col1:
        st.markdown(f'<div class="tooltip">10. {t_input("ST depression")}<span class="tooltiptext">{t("ST Depression induced by exercise")}</span></div>', unsafe_allow_html=True)
        STdepression = st.number_input(t_input("ST depression"), min_value=0.0, max_value=6.0, value=0.0, key="oldpeak")
        st.markdown(f'<div class="tooltip">11. {t_input("Slope of ST segment")}<span class="tooltiptext">{t("Slope of ST Segment (0-2)")}</span></div>', unsafe_allow_html=True)
        Slope = st.number_input(t_input("Slope of ST segment"), min_value=0, max_value=2, value=0, key="slope")
    with col2:
        st.markdown(f'<div class="tooltip">12. {t_input("Major vessels")}<span class="tooltiptext">{t("Number of Major Vessels (0-3)")}</span></div>', unsafe_allow_html=True)
        MajorVessels = st.number_input(t_input("Major vessels"), min_value=0, max_value=3, value=0, key="ca")
        st.markdown(f'<div class="tooltip">13. {t_input("Thalassemia")}<span class="tooltiptext">{t("Thalassemia (0-3)")}</span></div>', unsafe_allow_html=True)
        Thal = st.number_input(t_input("Thalassemia"), min_value=0, max_value=3, value=0, key="thal")

    if st.button(t("Heart Disease Test Result")):
        input_data = [Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHeartRate, ExerciseAngina, STdepression, Slope, MajorVessels, Thal]
        if any(x < 0 for x in input_data) or RestingBP > 200 or Cholesterol > 600 or MaxHeartRate > 250 or Age > 80:
            st.error(t('error_input'))
        else:
            try:
                with st.spinner(t('Analyzing your data...')):
                     input_arr = np.asarray(input_data).reshape(1, -1)
                     input_scaled = heart_scaler.transform(input_arr)
                     prob = heart_model.predict_proba(input_scaled)[0][1] * 100

                     if prob >= 50:
                         diagnosis = f"{t('You have a')} {prob:.1f}% {t('risk of Heart Disease.')}"
                         st.error(diagnosis)
                     else:
                         diagnosis = f"{t('You are healthy! Low risk of Heart Disease.')} ({prob:.1f}% {t('risk')})"
                         st.success(diagnosis)
                    
                    # Generate health insights
                     insights = get_health_insights("Heart Disease", input_data, 1 if probability >= 60 else 0)
                     st.subheader(t("health_insights"))
                     for insight in insights:
                        st.write(f"💡 {insight}")

                    # Generate and download PDF report
                     pdf_buffer = generate_pdf_report("Heart Disease", input_data, diagnosis, insights)
                     st.download_button(
                        label=t("download_report"),
                        data=pdf_buffer,
                        file_name="heart_disease_report.pdf",
                        mime="application/pdf"
                    )
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                print(f"Exception: {e}")

# ------------------------------------------------
# 🧠 Parkinson's Prediction
# ------------------------------------------------
elif selected == 'Parkinsons Prediction':
    st.title(t('parkinsons_pred'))
    if parkinson_animation:
        st_lottie(parkinson_animation, height=200, key="parkinson")
    
    st.markdown(f"### {t('input_order')}")
    st.markdown("""
    1. MDVP:Fo(Hz)
    2. MDVP:Fhi(Hz)
    3. MDVP:Flo(Hz)
    4. MDVP:Jitter(%)
    5. MDVP:Jitter(Abs)
    6. MDVP:RAP
    7. MDVP:PPQ
    8. Jitter:DDP
    9. MDVP:Shimmer
    10. MDVP:Shimmer(dB)
    11. Shimmer:APQ3
    12. Shimmer:APQ5
    13. MDVP:APQ
    14. Shimmer:DDA
    15. NHR
    16. HNR
    17. RPDE
    18. DFA
    19. spread1
    20. spread2
    21. D2
    22. PPE
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="tooltip">1. {t_input("MDVP:Fo(Hz)")}<span class="tooltiptext">{t("Fundamental frequency in Hz")}</span></div>', unsafe_allow_html=True)
        fo = st.number_input(t_input("MDVP:Fo(Hz)"), min_value=0.0, max_value=1000.0, value=0.0, key="fo")
        st.markdown(f'<div class="tooltip">2. {t_input("MDVP:Fhi(Hz)")}<span class="tooltiptext">{t("Maximum vocal frequency in Hz")}</span></div>', unsafe_allow_html=True)
        fhi = st.number_input(t_input("MDVP:Fhi(Hz)"), min_value=0.0, max_value=1000.0, value=0.0, key="fhi")
        st.markdown(f'<div class="tooltip">3. {t_input("MDVP:Flo(Hz)")}<span class="tooltiptext">{t("Minimum vocal frequency in Hz")}</span></div>', unsafe_allow_html=True)
        flo = st.number_input(t_input("MDVP:Flo(Hz)"), min_value=0.0, max_value=1000.0, value=0.0, key="flo")
    with col2:
        st.markdown(f'<div class="tooltip">4. {t_input("MDVP:Jitter(%)")}<span class="tooltiptext">{t("Jitter percentage")}</span></div>', unsafe_allow_html=True)
        jitter_percent = st.number_input(t_input("MDVP:Jitter(%)"), min_value=0.0, max_value=10.0, value=0.0, key="jitter_percent")
        st.markdown(f'<div class="tooltip">5. {t_input("MDVP:Jitter(Abs)")}<span class="tooltiptext">{t("Jitter in absolute terms")}</span></div>', unsafe_allow_html=True)
        jitter_abs = st.number_input(t_input("MDVP:Jitter(Abs)"), min_value=0.0, max_value=0.1, value=0.0, key="jitter_abs")
        st.markdown(f'<div class="tooltip">6. {t_input("MDVP:RAP")}<span class="tooltiptext">{t("Relative amplitude perturbation")}</span></div>', unsafe_allow_html=True)
        rap = st.number_input(t_input("MDVP:RAP"), min_value=0.0, max_value=0.1, value=0.0, key="rap")
    with col3:
        st.markdown(f'<div class="tooltip">7. {t_input("MDVP:PPQ")}<span class="tooltiptext">{t("Five-point period perturbation quotient")}</span></div>', unsafe_allow_html=True)
        ppq = st.number_input(t_input("MDVP:PPQ"), min_value=0.0, max_value=0.1, value=0.0, key="ppq")
        st.markdown(f'<div class="tooltip">8. {t_input("Jitter:DDP")}<span class="tooltiptext">{t("Average absolute difference of differences")}</span></div>', unsafe_allow_html=True)
        ddp = st.number_input(t_input("Jitter:DDP"), min_value=0.0, max_value=0.1, value=0.0, key="ddp")
        st.markdown(f'<div class="tooltip">9. {t_input("MDVP:Shimmer")}<span class="tooltiptext">{t("Shimmer percentage")}</span></div>', unsafe_allow_html=True)
        shimmer = st.number_input(t_input("MDVP:Shimmer"), min_value=0.0, max_value=10.0, value=0.0, key="shimmer")
    with col1:
        st.markdown(f'<div class="tooltip">10. {t_input("MDVP:Shimmer(dB)")}<span class="tooltiptext">{t("Shimmer in decibels")}</span></div>', unsafe_allow_html=True)
        shimmer_db = st.number_input(t_input("MDVP:Shimmer(dB)"), min_value=0.0, max_value=5.0, value=0.0, key="shimmer_db")
        st.markdown(f'<div class="tooltip">11. {t_input("Shimmer:APQ3")}<span class="tooltiptext">{t("Three-point amplitude perturbation quotient")}</span></div>', unsafe_allow_html=True)
        apq3 = st.number_input(t_input("Shimmer:APQ3"), min_value=0.0, max_value=0.1, value=0.0, key="apq3")
        st.markdown(f'<div class="tooltip">12. {t_input("Shimmer:APQ5")}<span class="tooltiptext">{t("Five-point amplitude perturbation quotient")}</span></div>', unsafe_allow_html=True)
        apq5 = st.number_input(t_input("Shimmer:APQ5"), min_value=0.0, max_value=0.1, value=0.0, key="apq5")
    with col2:
        st.markdown(f'<div class="tooltip">13. {t_input("MDVP:APQ")}<span class="tooltiptext">{t("Average amplitude perturbation quotient")}</span></div>', unsafe_allow_html=True)
        apq = st.number_input(t_input("MDVP:APQ"), min_value=0.0, max_value=0.1, value=0.0, key="apq")
        st.markdown(f'<div class="tooltip">14. {t_input("Shimmer:DDA")}<span class="tooltiptext">{t("Average absolute differences of amplitudes")}</span></div>', unsafe_allow_html=True)
        dda = st.number_input(t_input("Shimmer:DDA"), min_value=0.0, max_value=0.1, value=0.0, key="dda")
        st.markdown(f'<div class="tooltip">15. {t_input("NHR")}<span class="tooltiptext">{t("Noise-to-harmonics ratio")}</span></div>', unsafe_allow_html=True)
        nhr = st.number_input(t_input("NHR"), min_value=0.0, max_value=0.5, value=0.0, key="nhr")
    with col3:
        st.markdown(f'<div class="tooltip">16. {t_input("HNR")}<span class="tooltiptext">{t("Harmonics-to-noise ratio")}</span></div>', unsafe_allow_html=True)
        hnr = st.number_input(t_input("HNR"), min_value=0.0, max_value=50.0, value=0.0, key="hnr")
        st.markdown(f'<div class="tooltip">17. {t_input("RPDE")}<span class="tooltiptext">{t("Recurrence period density entropy")}</span></div>', unsafe_allow_html=True)
        rpde = st.number_input(t_input("RPDE"), min_value=0.0, max_value=1.0, value=0.0, key="rpde")
        st.markdown(f'<div class="tooltip">18. {t_input("DFA")}<span class="tooltiptext">{t("Detrended fluctuation analysis")}</span></div>', unsafe_allow_html=True)
        dfa = st.number_input(t_input("DFA"), min_value=0.0, max_value=1.0, value=0.0, key="dfa")
    with col1:
        st.markdown(f'<div class="tooltip">19. {t_input("spread1")}<span class="tooltiptext">{t("Non-linear measure of fundamental frequency variation")}</span></div>', unsafe_allow_html=True)
        spread1 = st.number_input(t_input("spread1"), min_value=0.0, max_value=1.0, value=0.0, key="spread1")
        st.markdown(f'<div class="tooltip">20. {t_input("spread2")}<span class="tooltiptext">{t("Second nonlinear measure of variation")}</span></div>', unsafe_allow_html=True)
        spread2 = st.number_input(t_input("spread2"), min_value=0.0, max_value=1.0, value=0.0, key="spread2")
        st.markdown(f'<div class="tooltip">21. {t_input("D2")}<span class="tooltiptext">{t("Correlation dimension")}</span></div>', unsafe_allow_html=True)
        d2 = st.number_input(t_input("D2"), min_value=0.0, max_value=5.0, value=0.0, key="d2")
    with col2:
        st.markdown(f'<div class="tooltip">22. {t_input("PPE")}<span class="tooltiptext">{t("Pitch period entropy")}</span></div>', unsafe_allow_html=True)
        ppe = st.number_input(t_input("PPE"), min_value=0.0, max_value=1.0, value=0.0, key="ppe")

    if st.button(t("Parkinson’s Test Result")):
        input_data = [fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp, shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]
        if any(x < 0 for x in input_data):
            st.error(t('error_input'))
        else:
            try:
                with st.spinner(t('Analyzing your data...')):
                     input_arr = np.asarray(input_data).reshape(1, -1)
                     input_scaled = parkinson_scaler.transform(input_arr)
                     prob = parkinson_model.predict_proba(input_scaled)[0][1] * 100

                     if prob >= 50:
                         diagnosis = f"{t('You have a')} {prob:.1f}% {t('risk of Parkinson\'s Disease.')}"
                         st.error(diagnosis)
                     else:
                         diagnosis = f"{t('You are healthy! Low risk of Parkinson\'s.')} ({prob:.1f}% {t('risk')})"
                         st.success(diagnosis) 
                    
                    # Generate health insights
                     insights = get_health_insights("Parkinsons", input_data, 1 if probability >= 60 else 0)
                     st.subheader(t("health_insights"))
                     for insight in insights:
                        st.write(f"💡 {insight}")

                    # Generate and download PDF report
                     pdf_buffer = generate_pdf_report("Parkinsons", input_data, diagnosis, insights)
                     st.download_button(
                        label=t("download_report"),
                        data=pdf_buffer,
                        file_name="parkinsons_report.pdf",
                        mime="application/pdf"
                    )
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                print(f"Exception: {e}")

# ------------------------------------------------
# 📝 Feedback Section
# ------------------------------------------------
elif selected == 'Feedback':
    st.title(t("feedback_title"))
    st.write(t("feedback_desc"))
    feedback = st.text_area("Your Feedback", height=200)
    if st.button(t("submit_feedback")):
        st.success(t("feedback_success"))
        st.write("Feedback submitted: ", feedback)

# ------------------------------------------------
# 📌 Footer
# ------------------------------------------------
st.markdown('<div class="footer">© 2025 Health Assistant | Built with ❤️ | Version 1.3</div>', unsafe_allow_html=True)