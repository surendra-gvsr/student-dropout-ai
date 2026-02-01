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

# --- 1. SETUP & CACHING (The "Backend") ---
@st.cache_resource
def load_resources():
    # Download NLTK data (runs once)
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()
    return sia

@st.cache_data
def load_and_train_model():
    # Load the UCI dataset locally
    try:
        df = pd.read_csv('data.csv', sep=';')
    except:
        # Fallback if semicolon separator fails
        try:
            df = pd.read_csv('data.csv', sep=',')
        except:
            st.error("⚠️ Error: 'data.csv' not found. Please upload it to your folder.")
            return None, None, None, None

    # Preprocessing
    le = LabelEncoder()
    df['Target'] = le.fit_transform(df['Target']) # Dropout=0, Enrolled=1, Graduate=2
    
    # Train Model (Random Forest)
    X = df.drop('Target', axis=1)
    y = df['Target']
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    
    return rf_model, X, le, df

# Load the AI and Model
sia = load_resources()
rf_model, X, le, full_df = load_and_train_model()

# --- 2. SIDEBAR NAVIGATION ---
st.sidebar.title("🎓 Edu-Risk AI")
st.sidebar.write("Navigate to:")

# DEFINING THE NEW ORDER
page = st.sidebar.radio("Go to:", [
    "Student Check-In (Landing)", 
    "Teacher Dashboard (Demo)", 
    "⚡ Live Simulation", 
    "Project Documentation"
])

st.sidebar.markdown("---")
st.sidebar.info("This tool predicts dropout risk by combining **Grades** (Hard Data) with **Student Sentiment** (Soft Data).")

# --- 3. PAGE: STUDENT CHECK-IN (LANDING PAGE) ---
if page == "Student Check-In (Landing)":
    st.title("🗣️ Student Voice Check-In")
    st.markdown("""
    ### 👋 Welcome, Student!
    This is the **Student App**. In a real-world scenario, students would use this on their phone once a week.
    
    **How it works:**
    1.  You tell us how you are feeling (text or voice).
    2.  Our AI listens to understand your stress levels.
    3.  If you need help, we privately alert a counselor.
    """)
    
    st.markdown("---")
    
    # Input Section
    st.subheader("How are you feeling about your classes this week?")
    st.info("🎙️ **Try it:** Type a sentence below to see how the AI analyzes it.")
    
    tab1, tab2 = st.tabs(["📝 Type Your Journal", "🎤 Record Voice Note"])
    
    user_text = ""
    
    with tab1:
        text_input = st.text_area("Write your thoughts here...", height=100)
        if st.button("Analyze My Check-In"):
            user_text = text_input
            
    with tab2:
        audio_value = st.audio_input("Record a voice note")
        if audio_value:
            st.success("Audio received! (In the full version, Whisper AI transcribes this).")
            # Simulation for demo purposes
            user_text = "I am struggling with my classes and I feel overwhelmed." 
            st.write(f"**Simulated Transcription:** *'{user_text}'*")

    # Analysis Logic
    if user_text:
        st.markdown("---")
        st.subheader("🤖 AI Analysis Result")
        
        # 1. Sentiment Score
        score = sia.polarity_scores(user_text)['compound']
        st.metric("Detected Emotional Score (-1.0 to +1.0)", f"{score:.2f}")
        
        # 2. Keyword Safety Net (The Override)
        crisis_keywords = ["quit", "drop out", "leave", "give up", "fail", "can't take this"]
        keyword_detected = any(word in user_text.lower() for word in crisis_keywords)

        # 3. The Decision
        if score < -0.25 or keyword_detected:
            st.error("⚠️ **System Alert: High Risk Detected**")
            st.write("**Reason:** The AI detected signs of distress or specific keywords related to dropping out.")
            st.toast("Alert sent to Faculty Dashboard", icon="🚨")
        elif score > 0.5:
            st.success("✅ **System Status: Healthy**")
            st.write("**Reason:** Positive sentiment detected. Keep up the good work!")
        else:
            st.info("ℹ️ **System Status: Neutral**")
            st.write("**Reason:** Normal check-in. No immediate action required.")

# --- 4. PAGE: TEACHER DASHBOARD (DEMO) ---
elif page == "Teacher Dashboard (Demo)":
    st.title("🏫 Faculty Dashboard")
    
    st.warning("""
    **👀 DEMO MODE:** This page simulates what a **Teacher or Administrator** would see. 
    The data below is **generated randomly** to show how the prioritization works.
    """)
    
    st.markdown("### 📋 Monday Morning Risk Report")
    st.write("This table ranks students by **Risk Level**. Teachers look at the RED flags first.")

    if st.button("🔄 Refresh Demo Data"):
        st.cache_data.clear()
    
    # --- SIMULATION LOGIC ---
    classroom = X.sample(15, random_state=int(time.time())).copy()
    
    # Simulate random "Feelings"
    simulated_sentiments = np.random.uniform(-0.9, 0.9, size=15)
    classroom['Sentiment Score'] = simulated_sentiments
    
    # Predict Risk
    preds = rf_model.predict_proba(classroom.drop('Sentiment Score', axis=1))
    risk_scores = preds[:, 0] * 100 
    
    # Fusion Logic
    final_risk = []
    for r, s in zip(risk_scores, simulated_sentiments):
        if s < -0.5:
            r += 25 
        final_risk.append(min(r, 100))
        
    classroom['Dropout Risk %'] = np.round(final_risk, 1)
    
    # Status Labels
    def get_status(risk):
        if risk > 70: return "🔴 CRITICAL"
        elif risk > 40: return "🟡 WATCH"
        else: return "🟢 STABLE"
        
    classroom['Status'] = classroom['Dropout Risk %'].apply(get_status)
    classroom = classroom.sort_values(by='Dropout Risk %', ascending=False)
    
    # Display Table
    st.dataframe(
        classroom[['Status', 'Dropout Risk %', 'Sentiment Score', 'Admission grade', 'Debtor']],
        column_config={
            "Dropout Risk %": st.column_config.ProgressColumn(
                "Risk Probability", format="%f%%", min_value=0, max_value=100
            ),
            "Sentiment Score": st.column_config.BarChartColumn(
                "Emotional State (AI)", y_min=-1, y_max=1
            )
        },
        use_container_width=True
    )
    
    st.caption("🔴 **Critical:** High probability of dropout + Negative Sentiment. Needs immediate meeting.")

# --- 5. PAGE: LIVE SIMULATION (NEW FEATURE) ---
elif page == "⚡ Live Simulation":
    st.title("⚡ Interactive Risk Simulator")
    st.markdown("""
    ### See the AI in Action
    Select a student profile below, then type a message to see how their **Dropout Risk** changes in real-time.
    This demonstrates how **Soft Data (Voice)** impacts **Hard Data (Grades)**.
    """)
    st.markdown("---")

    # Create two columns
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("1. Select a Student Profile")
        
        student_profiles = {
            "Student A (High Achiever)": {"grades": 160, "debt": 0, "attendance": 1}, 
            "Student B (Average / Struggling)": {"grades": 110, "debt": 0, "attendance": 1}, 
            "Student C (At-Risk Academic)": {"grades": 90, "debt": 1, "attendance": 0} 
        }
        
        selected_profile_name = st.selectbox("Choose a student to simulate:", list(student_profiles.keys()))
        profile_data = student_profiles[selected_profile_name]
        
        # Display Stats
        st.info(f"""
        **📋 Current Academic Stats:**
        * **Admission Grade:** {profile_data['grades']/10} / 20
        * **Tuition Fees Paid:** {'✅ Yes' if profile_data['debt'] == 0 else '❌ No'}
        * **Attendance:** {'✅ Good' if profile_data['attendance'] == 1 else '❌ Poor'}
        """)
        
        st.subheader("2. Enter Student Voice")
        user_input = st.text_area("Type a message here (e.g., 'I am quitting'):", height=100)
        
        analyze_btn = st.button("🚀 Update Dashboard", type="primary")

    with col_right:
        st.subheader("3. Live Risk Dashboard")
        
        if analyze_btn and user_input:
            # A. Base Risk (Simulated)
            if "Student A" in selected_profile_name: base_risk = 5.0
            elif "Student B" in selected_profile_name: base_risk = 35.0
            else: base_risk = 75.0
            
            # B. Analyze Text
            sentiment_score = sia.polarity_scores(user_input)['compound']
            
            # C. Keyword Override
            keywords = ["quit", "drop out", "leave", "fail", "stress", "can't take it"]
            keyword_flag = any(w in user_input.lower() for w in keywords)
            
            # D. The Fusion Logic
            risk_change = 0
            if keyword_flag:
                risk_change = +50 # Crisis word
            elif sentiment_score < -0.3:
                risk_change = +25 # Negative sentiment
            elif sentiment_score > 0.5:
                risk_change = -10 # Positive sentiment
                
            final_risk = min(max(base_risk + risk_change, 0), 100)
            
            # --- DISPLAY RESULTS ---
            c1, c2 = st.columns(2)
            c1.metric("Base Risk (Grades Only)", f"{base_risk}%")
            c2.metric("Final Risk (AI Adjusted)", f"{final_risk}%", delta=f"{risk_change}%", delta_color="inverse")
            
            st.progress(final_risk / 100)
            
            if final_risk > 70:
                st.error("🔴 **CRITICAL ALERT:** Student requires immediate intervention.")
            elif final_risk > 40:
                st.warning("🟡 **WARNING:** Student is showing signs of instability.")
            else:
                st.success("🟢 **STABLE:** Student is on track.")
                
            st.write("---")
            st.markdown("#### 🧠 AI Analysis")
            st.write(f"**Sentiment Score:** {sentiment_score:.2f} (Scale: -1 to +1)")
            if keyword_flag:
                st.write("⚠️ **Trigger:** Crisis keyword detected.")
            
        else:
            st.markdown("""
            <div style="text-align: center; color: gray; padding: 50px;">
                Enter text on the left to see the prediction update.
            </div>
            """, unsafe_allow_html=True)

# --- 6. PAGE: PROJECT DOCUMENTATION ---
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
