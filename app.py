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
            st.error("⚠️ Error: 'data.csv' not found. Please upload it to your folder.")
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

# Load Resources
sia = load_resources()
rf_model, X, le, full_df = load_and_train_model()

# --- 2. SIDEBAR NAVIGATION ---
st.sidebar.title("🎓 Edu-Risk AI")
st.sidebar.write("Navigate to:")

# NEW ORDER: Live Simulation is FIRST
page = st.sidebar.radio("Go to:", [
    "⚡ Live Simulation (Landing)", 
    "🗣️ Student App View", 
    "🏫 Teacher Dashboard View", 
    "📘 Project Documentation"
])

st.sidebar.markdown("---")
st.sidebar.info("This tool predicts dropout risk by combining **Grades** (Hard Data) with **Student Sentiment** (Soft Data).")

# --- 3. PAGE: LIVE SIMULATION (THE NEW LANDING PAGE) ---
if page == "⚡ Live Simulation (Landing)":
    st.title("⚡ Interactive Risk Simulator")
    st.markdown("""
    ### See the AI in Action
    Select a student profile below, then type a message to see how their **Dropout Risk** changes in real-time.
    """)
    st.markdown("---")

    # Create two columns for the interactive layout
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("1. Select a Student Profile")
        
        # We manually define 3 distinct profiles for the demo to make it clear
        student_profiles = {
            "Student A (High Achiever)": {"grades": 160, "debt": 0, "attendance": 1}, # Good grades (approx 16/20), no debt
            "Student B (Average / Struggling)": {"grades": 110, "debt": 0, "attendance": 1}, # Avg grades (11/20)
            "Student C (At-Risk Academic)": {"grades": 90, "debt": 1, "attendance": 0} # Low grades, debt, skipped class
        }
        
        selected_profile_name = st.selectbox("Choose a student to simulate:", list(student_profiles.keys()))
        profile_data = student_profiles[selected_profile_name]
        
        # Display the "Hard Data" stats
        st.info(f"""
        **📋 Current Academic Stats:**
        * **Admission Grade:** {profile_data['grades']/10} / 20
        * **Tuition Fees Paid:** {'✅ Yes' if profile_data['debt'] == 0 else '❌ No'}
        * **Attendance:** {'✅ Good' if profile_data['attendance'] == 1 else '❌ Poor'}
        """)
        
        st.subheader("2. Enter Student Voice")
        st.write("Imagine this student sends a message to their counselor.")
        user_input = st.text_area("Type a message here (e.g., 'I am quitting'):", height=100)
        
        analyze_btn = st.button("🚀 Update Dashboard", type="primary")

    with col_right:
        st.subheader("3. Live Risk Dashboard")
        
        if analyze_btn and user_input:
            # --- CALCULATE RISK ---
            
            # A. Base Risk (Academic Only) - We simulate this based on the profile
            # (In a real app, this comes from the Random Forest model)
            if "Student A" in selected_profile_name: base_risk = 5.0
            elif "Student B" in selected_profile_name: base_risk = 35.0
            else: base_risk = 75.0
            
            # B. Analyze Text (The AI Part)
            sentiment_score = sia.polarity_scores(user_input)['compound']
            
            # C. Keyword Override
            keywords = ["quit", "drop out", "leave", "fail", "stress", "can't take it"]
            keyword_flag = any(w in user_input.lower() for w in keywords)
            
            # D. The Fusion Logic (Math)
            risk_change = 0
            if keyword_flag:
                risk_change = +50 # Massive jump for crisis words
            elif sentiment_score < -0.3:
                risk_change = +25 # Moderate jump for bad vibes
            elif sentiment_score > 0.5:
                risk_change = -10 # Reduction for happy vibes
                
            final_risk = min(max(base_risk + risk_change, 0), 100)
            
            # --- DISPLAY RESULTS ---
            
            # 1. The Gauges
            c1, c2 = st.columns(2)
            c1.metric("Base Risk (Grades Only)", f"{base_risk}%")
            c2.metric("Final Risk (AI Adjusted)", f"{final_risk}%", delta=f"{risk_change}%", delta_color="inverse")
            
            st.progress(final_risk / 100)
            
            # 2. Status Badge
            if final_risk > 70:
                st.error("🔴 **CRITICAL ALERT:** Student requires immediate intervention.")
            elif final_risk > 40:
                st.warning("🟡 **WARNING:** Student is showing signs of instability.")
            else:
                st.success("🟢 **STABLE:** Student is on track.")
                
            # 3. Explanation
            st.write("---")
            st.markdown("#### 🧠 AI Analysis")
            st.write(f"**Sentiment Score:** {sentiment_score:.2f} (Scale: -1 to +1)")
            if keyword_flag:
                st.write("⚠️ **Trigger:** Crisis keyword detected.")
            elif risk_change > 0:
                st.write("📉 **Trigger:** Negative sentiment increased the risk score.")
            
        else:
            st.markdown("""
            <div style="text-align: center; color: gray; padding: 50px;">
                Enter text on the left to see the prediction update.
            </div>
            """, unsafe_allow_html=True)

# --- 4. PAGE: STUDENT APP VIEW (Old Landing) ---
elif page == "🗣️ Student App View":
    st.title("🗣️ Student App View")
    st.write("This is what the **student** sees on their phone.")
    st.markdown("---")
    
    text_input = st.text_area("How are you feeling?", "I am doing okay.")
    if st.button("Submit Check-In"):
        score = sia.polarity_scores(text_input)['compound']
        st.info(f"Analysis Complete. Sentiment Score: {score}")

# --- 5. PAGE: TEACHER DASHBOARD (Old Dashboard) ---
elif page == "🏫 Teacher Dashboard View":
    st.title("🏫 Faculty Dashboard")
    st.write("This is what the **teacher** sees.")
    
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        
    # Same simulation logic as before...
    classroom = X.sample(10, random_state=int(time.time())).copy()
    classroom['Risk Score'] = np.random.randint(10, 90, 10)
    st.dataframe(classroom[['Admission grade', 'Debtor', 'Risk Score']], use_container_width=True)

# --- 6. PAGE: DOCUMENTATION ---
elif page == "📘 Project Documentation":
    st.title("📘 Project Documentation")
    st.markdown("### The Problem\nSchools wait too long to help students...")
    # (Insert the rest of your documentation text here)
