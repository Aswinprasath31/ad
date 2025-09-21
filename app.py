import streamlit as st
import os
import platform
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer, util
import pyttsx3
import pandas as pd
import altair as alt
import io
from fpdf import FPDF
import random

# ----------------- CLOUD DETECTION -----------------
IS_CLOUD = "STREAMLIT_CLOUD" in os.environ or platform.system() == "Linux"
if IS_CLOUD:
    voice_input = False
    st.sidebar.info("Voice input/output disabled on Streamlit Cloud")
else:
    voice_input = st.sidebar.checkbox("Enable Voice Input")

# ----------------- PAGE CONFIG -----------------
st.set_page_config(page_title="AI Interviewer Platform", page_icon="🤖", layout="wide")

# ----------------- STYLING -----------------
st.markdown("""
<style>
body {background: linear-gradient(to right, #f5f7fa, #c3cfe2);}
.card {background: #fff; padding:15px; border-radius:15px; margin-bottom:15px; box-shadow: 2px 2px 15px rgba(0,0,0,0.1);}
</style>
""", unsafe_allow_html=True)

# ----------------- LOAD MODELS -----------------
@st.cache_resource
def load_models():
    tokenizer = AutoTokenizer.from_pretrained("aswinprasath31/interviewer-bot-finetuned-v1")
    model = AutoModelForCausalLM.from_pretrained("aswinprasath31/interviewer-bot-finetuned-v1")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    return pipe, embedder

generator, embedder = load_models()

# ----------------- TTS (only local) -----------------
if not IS_CLOUD:
    import pyttsx3
    import speech_recognition as sr
    engine = pyttsx3.init()
    engine.setProperty('rate',150)

    def speak_text(text):
        engine.say(text)
        engine.runAndWait()

    def record_audio():
        r = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Listening... Speak now!")
            audio = r.listen(source, phrase_time_limit=8)
        try:
            return r.recognize_google(audio)
        except:
            return ""
else:
    def speak_text(text):
        pass
    def record_audio():
        st.warning("Voice input not available on Streamlit Cloud")
        return ""

# ----------------- SIDEBAR -----------------
st.sidebar.header("Settings")
topic = st.sidebar.selectbox("Topic", ["Data Science", "Python", "Machine Learning", "General"])
restart = st.sidebar.button("Restart Interview")
download_report = st.sidebar.button("Download Full PDF Report")

# ----------------- QUESTIONS -----------------
questions_dict = {
    "Data Science":[
        {"q":"What is data cleaning and why is it important?", "ideal":"Data cleaning removes errors, missing values, and inconsistencies to improve analysis.","level":"Easy"},
        {"q":"Explain a machine learning project you worked on.", "ideal":"I implemented a predictive model using Random Forest to forecast sales.","level":"Medium"},
        {"q":"How do you select features for a complex model?", "ideal":"I use feature importance, correlation analysis, and domain knowledge.","level":"Hard"},
    ],
    "Python":[
        {"q":"What is a list comprehension?", "ideal":"A concise way to create lists in Python using a single line.","level":"Easy"},
        {"q":"Explain a Python project you implemented.", "ideal":"I created an automated report generation tool using Python and pandas.","level":"Medium"},
        {"q":"How do you optimize Python code for performance?", "ideal":"I use vectorized operations, efficient data structures, and profiling.","level":"Hard"},
    ]
}

# ----------------- SESSION STATE -----------------
if 'messages' not in st.session_state or restart:
    st.session_state.messages=[]
    st.session_state.scores=[]
    st.session_state.level="Easy"
    st.session_state.q_count=0
    st.session_state.current_ideal=""

# ----------------- FUNCTIONS -----------------
def generate_response(user_input):
    prompt = f"Topic: {topic}\nUser: {user_input}\nBot:"
    response = generator(prompt,max_length=200,do_sample=True,temperature=0.7)[0]['generated_text']
    bot_text = response.split("Bot:")[-1].strip()
    return bot_text

def semantic_score(user_answer, ideal_answer):
    emb1 = embedder.encode(user_answer, convert_to_tensor=True)
    emb2 = embedder.encode(ideal_answer, convert_to_tensor=True)
    sim = util.pytorch_cos_sim(emb1, emb2).item()
    return round(sim*100,2)

def select_question():
    qs = [q for q in questions_dict[topic] if q["level"]==st.session_state.level]
    return random.choice(qs) if qs else random.choice(questions_dict[topic])

# ----------------- DISPLAY CHAT -----------------
st.title("🤖 AI Interviewer Platform")
st.markdown("Answer questions. Bot adapts difficulty and evaluates your answers semantically!")

for msg in st.session_state.messages:
    if msg["role"]=="user":
        st.markdown(f'<div class="card"><b>You:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="card"><b>Bot:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)
        if "tip" in msg:
            st.markdown(f'<div class="card" style="background:#E6E6FA;"><b>Feedback:</b> {msg["tip"]}</div>', unsafe_allow_html=True)

# ----------------- INTERVIEW FLOW -----------------
if st.session_state.q_count < 5:
    if st.session_state.q_count==0 or st.session_state.messages[-1]["role"]=="user":
        current_q = select_question()
        st.session_state.current_ideal = current_q["ideal"]
        st.session_state.messages.append({"role":"bot","content":current_q["q"]})
        if voice_input:
            speak_text(current_q["q"])
    
    user_input=""
    if voice_input:
        if st.button("Record Answer"):
            user_input = record_audio()
    else:
        user_input = st.text_input("Your answer:", key=f"input_{st.session_state.q_count}")

    if user_input:
        st.session_state.messages.append({"role":"user","content":user_input})
        bot_resp = generate_response(user_input)
        score = semantic_score(user_input, st.session_state.current_ideal)
        tip = f"Semantic Score: {score}/100"
        st.session_state.scores.append(score)

        # Adaptive difficulty
        if score>70:
            st.session_state.level="Hard"
        elif score<50:
            st.session_state.level="Easy"
        else:
            st.session_state.level="Medium"

        st.session_state.messages.append({"role":"bot","content":bot_resp,"tip":tip})
        if voice_input:
            speak_text(bot_resp)
        st.session_state.q_count+=1
        st.experimental_rerun()
else:
    st.success("✅ Interview Completed!")
    avg_score = round(sum(st.session_state.scores)/len(st.session_state.scores),2)
    st.info(f"Average Semantic Score: {avg_score}/100")

    # ----------------- DASHBOARD -----------------
    st.subheader("📊 Performance Dashboard")
    df = pd.DataFrame({
        "Question":[msg["content"] for msg in st.session_state.messages if msg["role"]=="bot"],
        "Score":st.session_state.scores
    })

    # Interactive Altair chart
    chart = alt.Chart(df).mark_bar(cornerRadiusTopLeft=3,cornerRadiusTopRight=3).encode(
        x=alt.X('Score:Q', title='Score'),
        y=alt.Y('Question:N', sort='-x', title='Questions'),
        color=alt.condition(
            alt.datum.Score>70,
            alt.value("green"),
            alt.condition(alt.datum.Score<50, alt.value("red"), alt.value("orange"))
        )
    ).properties(height=300)
    st.altair_chart(chart, use_container_width=True)

    # Strengths & Weaknesses
    st.subheader("🔹 Strengths & Weaknesses")
    strengths = df[df["Score"]>70]
    weaknesses = df[df["Score"]<50]
    if not strengths.empty:
        st.markdown("**Strengths:**")
        st.table(strengths)
    if not weaknesses.empty:
        st.markdown("**Needs Improvement:**")
        st.table(weaknesses)

    # ----------------- DOWNLOAD PDF REPORT -----------------
    if download_report:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial",size=12)
        pdf.cell(200,10,txt="AI Interviewer Report",ln=True,align="C")
        pdf.ln(10)
        for msg in st.session_state.messages:
            role = "User" if msg["role"]=="user" else "Bot"
            pdf.multi_cell(0,10,f"{role}: {msg['content']}")
            if "tip" in msg:
                pdf.multi_cell(0,10,f"Feedback: {msg['tip']}")
            pdf.ln(2)
        pdf.multi_cell(0,10,f"Average Semantic Score: {avg_score}/100")
        pdf_output = io.BytesIO()
        pdf.output(pdf_output)
        pdf_output.seek(0)
        st.download_button("Download Full PDF Report", data=pdf_output, file_name="ai_interviewer_report.pdf")
