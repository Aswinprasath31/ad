import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ----------------- SETUP -----------------
st.set_page_config(page_title="Interviewer Bot", page_icon="🤖", layout="wide")

st.markdown("""
<style>
.chat-message.user {background-color: #DCF8C6; padding:10px; border-radius: 10px; margin: 5px;}
.chat-message.bot {background-color: #E6E6FA; padding:10px; border-radius: 10px; margin: 5px;}
</style>
""", unsafe_allow_html=True)

# ----------------- LOAD MODEL -----------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("aswinprasath31/interviewer-bot-finetuned-v1")
    model = AutoModelForCausalLM.from_pretrained("aswinprasath31/interviewer-bot-finetuned-v1")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
    return pipe

generator = load_model()

# ----------------- SIDEBAR -----------------
st.sidebar.header("Settings")
topic = st.sidebar.selectbox("Interview Topic", ["Data Science", "Python", "Machine Learning", "General"])
tone = st.sidebar.radio("Bot Tone", ["Formal", "Friendly"])
restart = st.sidebar.button("Restart Interview")

# ----------------- CHAT HISTORY -----------------
if 'messages' not in st.session_state or restart:
    st.session_state.messages = []

# ----------------- MAIN CHAT -----------------
st.title("🤖 AI Interviewer Bot")
st.markdown("Ask me anything or start your mock interview!")

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-message user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-message bot">{msg["content"]}</div>', unsafe_allow_html=True)

# ----------------- USER INPUT -----------------
def generate_response(user_input):
    prompt = f"Topic: {topic}\nTone: {tone}\nUser: {user_input}\nBot:"
    response = generator(prompt, max_length=200, do_sample=True, temperature=0.7)[0]['generated_text']
    # Extract only the bot response after "Bot:"
    bot_response = response.split("Bot:")[-1].strip()
    return bot_response

user_input = st.text_input("Your question / answer here:", key="input")
if st.button("Send") and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    bot_response = generate_response(user_input)
    st.session_state.messages.append({"role": "bot", "content": bot_response})
    st.experimental_rerun()
