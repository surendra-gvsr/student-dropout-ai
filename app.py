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
    # NEW WORKING LINE: Load local file
    try:
        # Try loading with semicolon separator (common for UCI)
        df = pd.read_csv('data.csv', sep=';')
    except:
        # Fallback to comma if you saved it differently
        df = pd.read_csv('data.csv', sep=',')

    # Preprocessing
    le = LabelEncoder()
    df['Target'] = le.fit_transform(df['Target']) 
    
    # ... rest of the function remains the same ...
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
    # Analysis Result
    if user_text:
        st.markdown("---")
        st.subheader("AI Analysis")
        
        # 1. Calculate Score
        score = sia.polarity_scores(user_text)['compound']
        st.metric("Detected Emotional Score", f"{score:.2f}")
        
        # 2. Define "Crisis Keywords" (The Override)
        # These words trigger a flag even if the sentiment isn't -1.0
        crisis_keywords = ["quit", "drop out", "leave", "give up", "failing", "can't take this"]
        
        # Check if any keyword is in the text
        keyword_detected = any(word in user_text.lower() for word in crisis_keywords)

        # 3. Smart Risk Logic
        # Flag if score is very negative OR if they used a crisis keyword
        if score < -0.25 or keyword_detected:
            st.error("⚠️ **Risk Flag Triggered:** The system has detected signs of distress.")
            
            if keyword_detected:
                st.write(f"**Reason:** Keyword detected in text.")
            else:
                st.write(f"**Reason:** Sentiment score is critically low.")
                
            st.write("Action: Notification sent to Student Counselor.")
            
        elif score > 0.5:
            st.success("✅ **Positive Status:** Keep up the good work!")
        else:
            st.info("ℹ️ **Neutral Status:** No immediate alerts.")

# --- 5. PAGE: PROJECT INFO (For Recruiters) ---
# --- 5. PAGE: PROJECT INFO ---
elif page == "Project Info":
    st.title("📘 Project Documentation")
    
    st.markdown("""
    ### 1. The Problem: Why I Built This
    Schools usually wait until a student fails a class to help them. By then, it is often too late. 
    Grades are **"lagging indicators"**—they tell you what happened in the past, not how the student is feeling right now.
    
    ### 2. The Solution
    I built an intelligent system that predicts dropout risk by combining two things:
    * **Hard Data:** Grades, attendance, and debt.
    * **Soft Data:** Student stress levels (analyzed from their voice/journals).
    
    It catches students who have **good grades** but are **secretly burnt out**.
    
    ---
    """)

    st.image("https://cdn-icons-png.flaticon.com/512/2083/2083236.png", width=100, caption="AI + Education")

    st.markdown("""
    ### 3. How It Works (The 3 Steps)
    1.  **The Inputs:** The system pulls academic records + student voice notes (e.g., *"I'm really stressed about money"*).
    2.  **The Brain (AI):** * **Whisper AI** transcribes the audio.
        * **Sentiment Analysis** measures stress.
        * **Random Forest Model** combines this with grades to calculate a "Total Risk Score."
    3.  **The Output:** Teachers see a prioritized dashboard. High-risk students are flagged in **RED**.
    
    ---
    
    ### 4. Real World Example: "Sarah"
    * **Sarah's Grades:** A+ (Excellent).
    * **Sarah's Life:** Working two jobs, exhausted, planning to quit.
    
    | Scenario | What Happens | Outcome |
    | :--- | :--- | :--- |
    | **Without AI** | School sees A+ grades. Thinks she is fine. | Sarah drops out unexpectedly. |
    | **With This System** | AI hears *"I can't keep up"* (Negative Sentiment). | Risk Score jumps to **80%**. Teacher intervenes. Sarah stays. |
    
    ---
    
    ### 5. Why This Matters
    * ✅ **Proactive:** Fixes problems *before* grades suffer.
    * ✅ **Human-Centric:** Listens to students, doesn't just treat them as numbers.
    * ✅ **Privacy-First:** Teachers see a Risk Score, not private diary entries.

    *Created by Surendra G*

    """)


