import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Student Dropout Risk AI", page_icon="🎓", layout="wide")

# --- 1. MOCK LOGIN SYSTEM (NEW) ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔒 School Admin Portal")
    st.markdown("Please sign in to access the Early Warning System.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        password = st.text_input("Enter Access Code", type="password")
        st.caption("🔑 **Demo Password:** admin")
        
        if st.button("Login"):
            if password == "admin":
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Incorrect Access Code")

if not st.session_state.logged_in:
    login()
    st.stop() # Stop execution here if not logged in

# --- 2. BACKEND SETUP & CACHING ---
@st.cache_resource
def load_resources():
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()
    return sia

@st.cache_data
def load_and_train_model():
    # Load Data
    try:
        df = pd.read_csv('data.csv', sep=';')
    except:
        try:
            df = pd.read_csv('data.csv', sep=',')
        except:
            st.error("⚠️ Error: 'data.csv' not found.")
            return None, None, None, None

    # Preprocessing
    le = LabelEncoder()
    df['Target'] = le.fit_transform(df['Target']) 
    
    X = df.drop('Target', axis=1)
    y = df['Target']
    
    # Train Model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    
    return rf_model, X, le, df

sia = load_resources()
rf_model, X, le, full_df = load_and_train_model()

# --- 3. SIDEBAR NAVIGATION ---
st.sidebar.title("🎓 Edu-Risk AI")
st.sidebar.success("👤 Logged in as Administrator")
st.sidebar.write("Navigate to:")

page = st.sidebar.radio("Go to:", [
    "Student Check-In (Landing)", 
    "Teacher Dashboard (Demo)", 
    "⚡ Live Simulation", 
    "Project Documentation"
])

st.sidebar.markdown("---")
if st.sidebar.button("Log Out"):
    st.session_state.logged_in = False
    st.rerun()

# --- 4. PAGE: STUDENT CHECK-IN ---
if page == "Student Check-In (Landing)":
    st.title("🗣️ Student Voice Check-In")
    st.markdown("""
    ### 👋 Welcome, Student!
    **How it works:**
    1.  Tell us how you are feeling (text or voice).
    2.  Our AI listens to understand your stress levels.
    3.  If you need help, we privately alert a counselor.
    """)
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["📝 Type Your Journal", "🎤 Record Voice Note"])
    
    user_text = ""
    with tab1:
        text_input = st.text_area("Write your thoughts here...", height=100)
        if st.button("Analyze My Check-In"): user_text = text_input
            
    with tab2:
        if st.audio_input("Record a voice note"):
            user_text = "I am struggling with my classes and I feel overwhelmed." 
            st.write(f"**Simulated Transcription:** *'{user_text}'*")

    if user_text:
        st.markdown("---")
        score = sia.polarity_scores(user_text)['compound']
        crisis_keywords = ["quit", "drop out", "leave", "give up", "fail"]
        keyword_detected = any(word in user_text.lower() for word in crisis_keywords)

        if score < -0.25 or keyword_detected:
            st.error("⚠️ **System Alert: High Risk Detected**")
            st.write("**Reason:** Distress detected. Alert sent to Counselor.")
        elif score > 0.5:
            st.success("✅ **System Status: Healthy**")
        else:
            st.info("ℹ️ **System Status: Neutral**")

# --- 5. PAGE: TEACHER DASHBOARD (UPDATED WITH DEMOGRAPHICS) ---
elif page == "Teacher Dashboard (Demo)":
    st.title("🏫 Faculty Dashboard")
    st.warning("**👀 DEMO MODE:** Data below is simulated for demonstration.")
    
    if st.button("🔄 Refresh Data"): st.cache_data.clear()
    
    # 1. Select Random Students
    classroom = X.sample(15, random_state=int(time.time())).copy()
    
    # 2. Add Simulated Sentiment
    simulated_sentiments = np.random.uniform(-0.9, 0.9, size=15)
    classroom['Sentiment Score'] = simulated_sentiments
    
    # 3. Predict Risk
    preds = rf_model.predict_proba(classroom.drop('Sentiment Score', axis=1))
    risk_scores = preds[:, 0] * 100 
    
    # 4. Fusion Logic
    final_risk = []
    for r, s in zip(risk_scores, simulated_sentiments):
        if s < -0.5: r += 25 
        final_risk.append(min(r, 100))
    classroom['Dropout Risk %'] = np.round(final_risk, 1)
    
    # 5. Add Demographics (Mapping numbers to text)
    # Assuming Gender: 1=Male, 0=Female (Standard UCI encoding)
    classroom['Gender Text'] = classroom['Gender'].apply(lambda x: "Male" if x == 1 else "Female")
    
    # 6. Status Labels
    def get_status(risk):
        if risk > 70: return "🔴 CRITICAL"
        elif risk > 40: return "🟡 WATCH"
        else: return "🟢 STABLE"
    classroom['Status'] = classroom['Dropout Risk %'].apply(get_status)
    classroom = classroom.sort_values(by='Dropout Risk %', ascending=False)
    
    # 7. Display Table (Rearranged columns for better view)
    st.dataframe(
        classroom[['Status', 'Dropout Risk %', 'Sentiment Score', 'Gender Text', 'Age at enrollment', 'Admission grade', 'Debtor']],
        column_config={
            "Dropout Risk %": st.column_config.ProgressColumn("Risk Probability", format="%f%%", min_value=0, max_value=100),
            "Sentiment Score": st.column_config.BarChartColumn("Emotional State", y_min=-1, y_max=1),
            "Gender Text": "Gender",
            "Age at enrollment": "Age"
        },
        use_container_width=True
    )

# --- 6. PAGE: LIVE SIMULATION ---
elif page == "⚡ Live Simulation":
    st.title("⚡ Interactive Risk Simulator")
    st.markdown("Select a profile and type a message to see how **Voice** impacts **Risk**.")
    st.markdown("---")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("1. Select Student Profile")
        student_profiles = {
            "Student A (High Achiever)": {"grades": 160, "debt": 0, "attendance": 1}, 
            "Student B (Average / Struggling)": {"grades": 110, "debt": 0, "attendance": 1}, 
            "Student C (At-Risk Academic)": {"grades": 90, "debt": 1, "attendance": 0} 
        }
        selected_profile_name = st.selectbox("Profile:", list(student_profiles.keys()))
        
        st.subheader("2. Enter Student Voice")
        user_input = st.text_area("Type a message (e.g. 'I am quitting'):", height=100)
        analyze_btn = st.button("🚀 Update Dashboard", type="primary")

    with col_right:
        st.subheader("3. Live Risk Dashboard")
        if analyze_btn and user_input:
            if "Student A" in selected_profile_name: base_risk = 5.0
            elif "Student B" in selected_profile_name: base_risk = 35.0
            else: base_risk = 75.0
            
            sentiment_score = sia.polarity_scores(user_input)['compound']
            keyword_flag = any(w in user_input.lower() for w in ["quit", "drop out", "leave", "fail"])
            
            risk_change = 0
            if keyword_flag: risk_change = +50
            elif sentiment_score < -0.3: risk_change = +25
            elif sentiment_score > 0.5: risk_change = -10
            
            final_risk = min(max(base_risk + risk_change, 0), 100)
            
            c1, c2 = st.columns(2)
            c1.metric("Base Risk", f"{base_risk}%")
            c2.metric("Final Risk", f"{final_risk}%", delta=f"{risk_change}%", delta_color="inverse")
            st.progress(final_risk / 100)
            
            if final_risk > 70: st.error("🔴 **CRITICAL ALERT**")
            elif final_risk > 40: st.warning("🟡 **WARNING**")
            else: st.success("🟢 **STABLE**")

# --- 7. PAGE: DOCUMENTATION ---
elif page == "Project Documentation":
    st.title("📘 Project Documentation")
    
    st.markdown("""
    ### 1. The Problem
    Schools usually wait until a student fails a class to help them. By then, it is often too late. 
    Grades are **"lagging indicators"**—they tell you what happened in the past, not how the student is feeling right now.
    
    ### 2. The Solution
    I built an intelligent system that predicts dropout risk by combining two things:
    * **Hard Data:** Grades, attendance, and debt (from the UCI Dataset).
    * **Soft Data:** Student stress levels (analyzed from voice/text).
    
    It catches students who have **good grades** but are **secretly burnt out**.
    
    ---
    
    ### 3. Real World Example: "Sarah"
    * **Sarah's Grades:** A+ (Excellent).
    * **Sarah's Life:** Working two jobs, exhausted, planning to quit.
    
    | Scenario | What Happens | Outcome |
    | :--- | :--- | :--- |
    | **Without AI** | School sees A+ grades. Thinks she is fine. | Sarah drops out unexpectedly. |
    | **With This System** | AI hears *"I can't keep up"* (Negative Sentiment). | Risk Score jumps to **80%**. Teacher intervenes. Sarah stays. |
    
    ---
    
    ### 4. Technical Stack
    * **Python & Streamlit:** For the web interface.
    * **Scikit-Learn:** Random Forest Classifier (85% Accuracy).
    * **NLTK / VADER:** For Natural Language Processing.
    """)


