import io
import re
import random
import urllib.parse
import json
import urllib.request
import streamlit as st
from deep_translator import GoogleTranslator, MyMemoryTranslator
from gtts import gTTS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Optional transliteration for genuine Hinglish output
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    HAS_INDIC = True
except ImportError:
    HAS_INDIC = False

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Vernacular Study Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    .stTextArea textarea {
        border-radius: 10px;
        border: 1px solid #cbd5e1;
        font-size: 15px;
    }
    .summary-box {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #4f46e5;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        margin-bottom: 16px;
        line-height: 1.6;
        font-size: 16px;
        color: #1e293b;
    }
    .quiz-card {
        background-color: #f1f5f9;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        border: 1px solid #e2e8f0;
        color: #1e293b;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "results" not in st.session_state:
    st.session_state.results = None

# ---------------- MODEL LOADING ----------------
@st.cache_resource(show_spinner="AI Model load ho raha hai (First time 10-15 sec lagenge)...")
def load_model():
    model_name = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# ---------------- MULTI-FALLBACK TRANSLATOR ----------------
@st.cache_data(show_spinner=False)
def robust_translate(text, target_lang):
    """
    3-Layer Fallback Translation taaki Streamlit Cloud par kabhi 'Too Many Requests' na aaye:
    1. Google GTX API endpoint (Direct lightweight endpoint - rarely blocked)
    2. deep_translator GoogleTranslator
    3. deep_translator MyMemoryTranslator
    """
    if target_lang == "en" or not text.strip():
        return text

    # Method 1: Direct Google GTX API (Bypasses standard web scraper blocks)
    try:
         encoded_text = urllib.parse.quote(text)
         url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl={target_lang}&dt=t&q={encoded_text}"
         req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
         with urllib.request.urlopen(req, timeout=8) as response:
             data = json.loads(response.read().decode('utf-8'))
             translated = "".join([sentence[0] for sentence in data[0] if sentence[0]])
             if translated:
                 return translated
    except Exception:
        pass

    # Method 2: deep_translator GoogleTranslator
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception:
        pass

    # Method 3: MyMemory Translator Fallback
    lang_map = {
        "hi": "hi-IN",
        "mr": "mr-IN",
        "ta": "ta-IN",
        "te": "te-IN",
        "bn": "bn-IN",
        "gu": "gu-IN"
    }
    mm_target = lang_map.get(target_lang, target_lang)
    return MyMemoryTranslator(source='en-GB', target=mm_target).translate(text)

# ---------------- HELPER FUNCTIONS ----------------
def chunk_text(text, max_words=280):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i + max_words]))
    return chunks

def summarize_long_text(text, summary_length="Medium"):
    chunks = chunk_text(text)
    summaries = []
    
    length_map = {
        "Short": (20, 60),
        "Medium": (35, 110),
        "Detailed": (60, 160)
    }
    min_l, max_l = length_map.get(summary_length, (35, 110))

    for chunk in chunks:
        inputs = tokenizer(
            "summarize: " + chunk,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_l,
            min_length=min_l,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )
        part = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        part = ". ".join([s.strip().capitalize() for s in part.split(".") if s.strip()])
        summaries.append(part + ".")
        
    return " ".join(summaries)

def convert_to_hinglish(english_text):
    """English -> Hindi -> Clean Hinglish (Roman Hindi)"""
    hindi_text = robust_translate(english_text, 'hi')
    if HAS_INDIC:
        raw_roman = transliterate(hindi_text, sanscript.DEVANAGARI, sanscript.ITRANS)
        replacements = {
            "A": "aa", "I": "ee", "U": "oo", "M": "n", ".n": "n",
            ".N": "n", "Ch": "chh", "shh": "sh", "~N": "n", "RRi": "ri"
        }
        for k, v in replacements.items():
            raw_roman = raw_roman.replace(k, v)
        return raw_roman.capitalize(), hindi_text
    else:
        return hindi_text, hindi_text

def generate_quiz_questions(english_summary):
    sentences = [s.strip() for s in english_summary.split(".") if len(s.strip()) > 25]
    quizzes = []
    
    for sent in sentences[:3]:
        words = re.findall(r'\b[A-Za-z]{6,}\b', sent)
        if words:
            target_word = random.choice(words)
            blank_sent = re.sub(rf'\b{target_word}\b', "_______", sent, count=1)
            quizzes.append({
                "question": blank_sent + ".",
                "answer": target_word
            })
    return quizzes

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Study Settings")
    
    target_lang = st.selectbox(
        "🌐 Output Language:",
        options=[
            ("Hindi (हिंदी)", "hi"),
            ("Hinglish (Chat Style)", "hinglish"),
            ("Marathi (मराठी)", "mr"),
            ("Tamil (தமிழ்)", "ta"),
            ("Telugu (తెలుగు)", "te"),
            ("Bengali (বাংলা)", "bn"),
            ("Gujarati (ગુજરાતી)", "gu"),
            ("Simplified English", "en")
        ],
        format_func=lambda x: x[0]
    )
    
    summary_length = st.select_slider(
        "📏 Summary Detail Level:",
        options=["Short", "Medium", "Detailed"],
        value="Medium"
    )
    
    slow_audio = st.checkbox("🐢 Slow Audio Speed (For easy listening)", value=False)
    
    st.divider()
    st.caption("💡 **Tip:** Agar test karne ke liye text nahi hai, toh 'Load Sample Science Note' button use karo.")

# ---------------- MAIN HEADER ----------------
st.title("🎓 Vernacular Study Assistant")
st.write("Complex English chapters ya notes paste karo — aur pao **simplified summary, regional audio, aur revision quiz**.")

SAMPLE_TEXT = (
    "Photosynthesis is the biological process by which green plants, algae, and certain bacteria "
    "transform light energy from the sun into chemical energy. During this process, plants capture "
    "sunlight using chlorophyll present in their leaves. They absorb water from the soil through "
    "their roots and take in carbon dioxide from the air through tiny pores called stomata. "
    "Using solar energy, plants convert water and carbon dioxide into glucose, which serves as "
    "food for the plant to grow, and release oxygen back into the atmosphere as a byproduct. "
    "This oxygen is essential for the survival of almost all living organisms on Earth."
)

col_input, col_output = st.columns([1.05, 1], gap="large")

# ---------------- LEFT COLUMN (INPUT) ----------------
with col_input:
    st.subheader("📥 1. Study Material")
    
    btn_sample, btn_clear = st.columns(2)
    with btn_sample:
        if st.button("✨ Load Sample Science Note", use_container_width=True):
            st.session_state.input_text = SAMPLE_TEXT
    with btn_clear:
        if st.button("🗑️ Clear Text", use_container_width=True):
            st.session_state.input_text = ""
            st.session_state.results = None

    input_text = st.text_area(
        "Apna lesson ya notes yahan paste karo:",
        value=st.session_state.input_text,
        height=260,
        placeholder="Paste your textbook paragraph or lecture notes here..."
    )
    st.session_state.input_text = input_text

    word_count = len(input_text.split()) if input_text.strip() else 0
    st.caption(f"📊 Word Count: **{word_count} words** | Est. Reading Time: **{max(1, word_count // 150)} min**")

    process_btn = st.button("🚀 Summarize, Translate & Create Audio", type="primary", use_container_width=True)

# ---------------- PROCESSING LOGIC ----------------
if process_btn:
    if not input_text.strip() or word_count < 15:
        st.warning("⚠️ Please kam se kam 15-20 words ka paragraph paste karo taaki summary acchi bane!")
    else:
        with st.status("🧠 AI tumhare notes process kar raha hai...", expanded=True) as status:
            st.write("1️⃣ Complex sentences ko simplify aur summarize kiya ja raha hai...")
            eng_summary = summarize_long_text(input_text, summary_length)
            
            lang_label, lang_code = target_lang
            st.write(f"2️⃣ Summary ko **{lang_label}** me translate kiya ja raha hai...")
            
            try:
                if lang_code == "hinglish":
                    final_summary, hindi_audio_text = convert_to_hinglish(eng_summary)
                    tts_lang = "hi"
                    tts_text = hindi_audio_text
                elif lang_code != "en":
                    final_summary = robust_translate(eng_summary, lang_code)
                    tts_lang = lang_code
                    tts_text = final_summary
                else:
                    final_summary = eng_summary
                    tts_lang = "en"
                    tts_text = eng_summary
            except Exception as e:
                st.warning("Primary translator busy tha, showing English summary.")
                final_summary = eng_summary
                tts_lang = "en"
                tts_text = eng_summary

            st.write("3️⃣ Regional voice-over aur practice quiz tayyar ho raha hai...")
            audio_buffer = io.BytesIO()
            try:
                tts = gTTS(text=tts_text, lang=tts_lang, slow=slow_audio)
                tts.write_to_fp(audio_buffer)
                audio_buffer.seek(0)
                audio_data = audio
