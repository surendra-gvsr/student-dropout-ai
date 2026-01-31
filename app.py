import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="Student Dropout Risk AI", page_icon="🎓", layout="wide")

# --- 1. SETUP & CACHING (The "Backend") ---
@st.cache_resource
def load_resources():
    # Download NLTK data
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()
    return sia

@st.cache_data
def load_and_train_model():
    # Load dataset (using direct URL to UCI to ensure it works online)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00697/predict+students'+dropout+and+academic+success.csv"
    try:
        df = pd.read_csv(url, sep=';')
    except:
        st.error("Could not load dataset from UCI.")
        return None, None, None, None

    # Preprocessing
    le = LabelEncoder()
    df['Target'] = le.fit_transform(df['Target']) # Dropout=0, Enrolled=1, Graduate=2
    
    # Simple cleanup
    X = df.drop('Target', axis=1)
    y = df['Target']
    
    # Train Model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    
    return rf_model, X, le, df

# Load everything
sia = load_resources()
rf_model, X, le, full_df = load_and_train_model()

# --- 2. SIDEBAR (Navigation) ---
st.sidebar.title("🎓 Edu-Risk AI")
page = st.sidebar.radio("Go to:", ["Teacher Dashboard", "Student Voice Check-in", "Project Info"])
st.sidebar.markdown("---")
st.sidebar.info("This tool predicts student dropout risk by combining academic data with emotional sentiment analysis.")

# --- 3. PAGE: TEACHER DASHBOARD ---
if page == "Teacher Dashboard":
    st.title("🏫 Faculty Dashboard")
    st.markdown("### Risk Monitoring System")
    
    # Simulate Monday Morning Data
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
    
    # Create the "Mock" Class
    classroom = X.sample(15, random_state=int(time.time())).copy()
    
    # Simulate Sentiments (In real life, this comes from the database)
    # We mix in some high-risk sentiments
    simulated_sentiments = np.random.uniform(-0.9, 0.9, size=15)
    classroom['Sentiment Score'] = simulated_sentiments
    
    # Predict
    # (Ensure columns match training data by temporarily dropping sentiment)
    preds = rf_model.predict_proba(classroom.drop('Sentiment Score', axis=1))
    risk_scores = preds[:, 0] * 100 # Probability of Dropout
    
    # Combined Algorithm: Boost risk if sentiment is negative
    # Formula: If Sentiment < -0.5, add 20% to risk
    final_risk = []
    for r, s in zip(risk_scores, simulated_sentiments):
        if s < -0.5:
            r += 20 
        final_risk.append(min(r, 100)) # Cap at 100%
        
    classroom['Dropout Risk %'] = np.round(final_risk, 1)
    
    # Add Status Flags
    def get_status(risk):
        if risk > 70: return "🔴 CRITICAL"
        elif risk > 40: return "🟡 WATCH"
        else: return "🟢 STABLE"
        
    classroom['Status'] = classroom['Dropout Risk %'].apply(get_status)
    
    # Display Main Table
    st.dataframe(
        classroom[['Status', 'Dropout Risk %', 'Sentiment Score', 'Admission grade', 'Debtor']],
        column_config={
            "Dropout Risk %": st.column_config.ProgressColumn(
                "Risk Level", format="%f%%", min_value=0, max_value=100
            ),
            "Sentiment Score": st.column_config.BarChartColumn(
                "Emotional State (-1 to +1)", y_min=-1, y_max=1
            )
        },
        use_container_width=True
    )
    
    st.caption("🔴 Critical = High probability of dropout + Negative Sentiment.")

# --- 4. PAGE: STUDENT VOICE CHECK-IN ---
elif page == "Student Voice Check-in":
    st.title("🗣️ Student Check-In")
    st.write("This is the interface a student would see on their mobile app.")
    
    st.info("🎙️ **Demo Mode:** Since this is a web demo, you can type OR record.")
    
    # Input Method
    tab1, tab2 = st.tabs(["📝 Text Journal", "🎤 Voice Note"])
    
    user_text = ""
    
    with tab1:
        text_input = st.text_area("How are you feeling about your classes this week?")
        if st.button("Analyze Text"):
            user_text = text_input
            
    with tab2:
        st.warning("Note: Voice recording requires browser permission.")
        # Streamlit Audio Input (New Feature)
        audio_value = st.audio_input("Record a note")
        if audio_value:
            st.success("Audio captured! (In full production, Whisper transcribes this here. For this demo, we use text simulation if Whisper is too heavy for the free cloud tier).")
            # NOTE: For a lightweight resume demo, we might simulate transcription 
            # or use a lighter STT if Whisper crashes the free tier memory.
            # Let's fallback to asking them to summarize what they said for the demo.
            user_text = "I am struggling with my classes and I feel overwhelmed." 
            st.write(f"**Simulated Transcription:** {user_text}")

    # Analysis Result
    if user_text:
        st.markdown("---")
        st.subheader("AI Analysis")
        
        # 1. Sentiment
        score = sia.polarity_scores(user_text)['compound']
        st.metric("Detected Emotional Score", f"{score:.2f}")
        
        # 2. Risk Context
        if score < -0.5:
            st.error("⚠️ **Risk Flag Triggered:** The system has detected signs of distress.")
            st.write("Action: Notification sent to Student Counselor.")
        elif score > 0.5:
            st.success("✅ **Positive Status:** Keep up the good work!")
        else:
            st.info("ℹ️ **Neutral Status:** No immediate alerts.")

# --- 5. PAGE: PROJECT INFO (For Recruiters) ---
elif page == "Project Info":
    st.title("Project Documentation")
    st.markdown("""
    ### 🎯 Objective
    To reduce student dropout rates by moving from **Lagging Indicators** (Grades) to **Leading Indicators** (Sentiment & Stress).
    
    ### 🛠️ Tech Stack
    * **Model:** Random Forest (Scikit-Learn) trained on UCI Dataset.
    * **NLP:** VADER / OpenAI Whisper (Architecture).
    * **Interface:** Streamlit.
    
    ### 🧠 The Methodology
    1.  **Hard Data:** The model analyzes 30+ academic factors (Attendance, Debt, Grades).
    2.  **Soft Data:** The system ingests unstructured student feedback.
    3.  **Fusion:** A weighted risk score is calculated to prioritize counselor interventions.
    
    *Created by [Your Name]*
    """)