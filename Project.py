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

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    HAS_INDIC = True
except ImportError:
    HAS_INDIC = False

st.set_page_config(page_title="Vernacular Study Assistant", page_icon="🎓", layout="wide")

st.markdown("""
    <style>
    .summary-box {
        background-color: #ffffff;
        padding: 18px;
        border-radius: 10px;
        border-left: 5px solid #4f46e5;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        margin-bottom: 14px;
        line-height: 1.6;
        color: #1e293b;
    }
    .quiz-card {
        background-color: #f1f5f9;
        padding: 14px;
        border-radius: 8px;
        margin-bottom: 8px;
        border: 1px solid #cbd5e1;
        color: #1e293b;
    }
    </style>
""", unsafe_allow_html=True)

if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "results" not in st.session_state:
    st.session_state.results = None

@st.cache_resource(show_spinner="AI Model load ho raha hai...")
def load_model():
    model_name = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

@st.cache_data(show_spinner=False)
def robust_translate(text, target_lang):
    if target_lang == "en" or not text.strip():
        return text

    # 1. Direct Google GTX API
    try:
        encoded = urllib.parse.quote(text)
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl={target_lang}&dt=t&q={encoded}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            translated = "".join([s[0] for s in data[0] if s[0]])
            if translated:
                return translated
    except Exception:
        pass

    # 2. deep_translator Google
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception:
        pass

    # 3. MyMemory Fallback
    lang_map = {"hi": "hi-IN", "mr": "mr-IN", "ta": "ta-IN", "te": "te-IN", "bn": "bn-IN", "gu": "gu-IN"}
    return MyMemoryTranslator(source='en-GB', target=lang_map.get(target_lang, target_lang)).translate(text)

def summarize_long_text(text, summary_length="Medium"):
    words = text.split()
    chunks = [" ".join(words[i:i + 280]) for i in range(0, len(words), 280)]
    summaries = []
    length_map = {"Short": (20, 60), "Medium": (35, 110), "Detailed": (60, 160)}
    min_l, max_l = length_map.get(summary_length, (35, 110))

    for chunk in chunks:
        inputs = tokenizer("summarize: " + chunk, return_tensors="pt", max_length=512, truncation=True)
        ids = model.generate(inputs["input_ids"], max_length=max_l, min_length=min_l, length_penalty=2.0, num_beams=4, early_stopping=True)
        part = tokenizer.decode(ids[0], skip_special_tokens=True)
        part = ". ".join([s.strip().capitalize() for s in part.split(".") if s.strip()])
        summaries.append(part + ".")
    return " ".join(summaries)

def convert_to_hinglish(english_text):
    hindi_text = robust_translate(english_text, 'hi')
    if HAS_INDIC:
        raw_roman = transliterate(hindi_text, sanscript.DEVANAGARI, sanscript.ITRANS)
        for k, v in {"A": "aa", "I": "ee", "U": "oo", "M": "n", ".n": "n", ".N": "n", "Ch": "chh", "shh": "sh"}.items():
            raw_roman = raw_roman.replace(k, v)
        return raw_roman.capitalize(), hindi_text
    return hindi_text, hindi_text

def generate_quiz_questions(english_summary):
    sentences = [s.strip() for s in english_summary.split(".") if len(s.strip()) > 25]
    quizzes = []
    for sent in sentences[:3]:
        words = re.findall(r'\b[A-Za-z]{6,}\b', sent)
        if words:
            target_word = random.choice(words)
            blank_sent = re.sub(rf'\b{target_word}\b', "_______", sent, count=1)
            quizzes.append({"question": blank_sent + ".", "answer": target_word})
    return quizzes

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
    summary_length = st.select_slider("📏 Summary Detail Level:", options=["Short", "Medium", "Detailed"], value="Medium")
    slow_audio = st.checkbox("🐢 Slow Audio Speed", value=False)

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

with col_input:
    st.subheader("📥 1. Study Material")
    b1, b2 = st.columns(2)
    with b1:
        if st.button("✨ Load Sample Science Note", use_container_width=True):
            st.session_state.input_text = SAMPLE_TEXT
    with b2:
        if st.button("🗑️ Clear Text", use_container_width=True):
            st.session_state.input_text = ""
            st.session_state.results = None

    input_text = st.text_area("Apna lesson ya notes yahan paste karo:", value=st.session_state.input_text, height=250)
    st.session_state.input_text = input_text
    word_count = len(input_text.split()) if input_text.strip() else 0
    st.caption(f"📊 Word Count: **{word_count} words**")
    process_btn = st.button("🚀 Summarize, Translate & Create Audio", type="primary", use_container_width=True)

if process_btn:
    if not input_text.strip() or word_count < 15:
        st.warning("⚠️ Please kam se kam 15-20 words ka paragraph paste karo!")
    else:
        with st.status("🧠 AI tumhare notes process kar raha hai...", expanded=True) as status:
            st.write("1️⃣ Summarizing notes...")
            eng_summary = summarize_long_text(input_text, summary_length)
            lang_label, lang_code = target_lang
            st.write(f"2️⃣ Translating to **{lang_label}**...")
            try:
                if lang_code == "hinglish":
                    final_summary, tts_text = convert_to_hinglish(eng_summary)
                    tts_lang = "hi"
                elif lang_code != "en":
                    final_summary = robust_translate(eng_summary, lang_code)
                    tts_lang = lang_code
                    tts_text = final_summary
                else:
                    final_summary = eng_summary
                    tts_lang = "en"
                    tts_text = eng_summary
            except Exception:
                final_summary = eng_summary
                tts_lang = "en"
                tts_text = eng_summary

            st.write("3️⃣ Generating audio & quiz...")
            audio_buffer = io.BytesIO()
            try:
                tts = gTTS(text=tts_text, lang=tts_lang, slow=slow_audio)
                tts.write_to_fp(audio_buffer)
                audio_buffer.seek(0)
                audio_data = audio_buffer.getvalue()
            except Exception:
                audio_data = None

            st.session_state.results = {
                "lang_label": lang_label,
                "final_summary": final_summary,
                "eng_summary": eng_summary,
                "audio_data": audio_data,
                "quizzes": generate_quiz_questions(eng_summary)
            }
            status.update(label="✅ Notes Ready!", state="complete", expanded=False)

with col_output:
    st.subheader("📖 2. Simplified Study Kit")
    if st.session_state.results:
        res = st.session_state.results
        tab1, tab2, tab3 = st.tabs([f"🗣️ {res['lang_label']}", "🇬🇧 English Summary", "📝 Practice Quiz"])
        
        with tab1:
            st.markdown(f"<div class='summary-box'>{res['final_summary']}</div>", unsafe_allow_html=True)
            st.markdown("**📌 Key Takeaways:**")
            for pt in [s.strip() for s in res['final_summary'].replace("।", ".").split(".") if len(s.strip()) > 5]:
                st.markdown(f"- {pt}.")
            if res["audio_data"]:
                st.markdown("### 🎧 Audio Lesson")
                st.audio(res["audio_data"], format="audio/mp3")
            d1, d2 = st.columns(2)
            with d1:
                st.download_button("📄 Download Notes (.txt)", data=res["final_summary"], file_name="notes.txt", use_container_width=True)
            with d2:
                if res["audio_data"]:
                    st.download_button("🎵 Download Audio (.mp3)", data=res["audio_data"], file_name="audio.mp3", mime="audio/mp3", use_container_width=True)
        with tab2:
            st.markdown(f"<div class='summary-box'>{res['eng_summary']}</div>", unsafe_allow_html=True)
        with tab3:
            if res["quizzes"]:
                for idx, q in enumerate(res["quizzes"], 1):
                    st.markdown(f"<div class='quiz-card'><b>Q{idx}:</b> {q['question']}</div>", unsafe_allow_html=True)
                    with st.expander(f"👀 Show Answer {idx}"):
                        st.success(f"**Correct Answer:** {q['answer']}")
            else:
                st.info("Quiz ke liye thoda bada paragraph paste karo.")
    else:
        st.info("👈 Left side me apna chapter paste karo ya **'Load Sample Science Note'** par click karke dekho!")
