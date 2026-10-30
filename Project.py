import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from deep_translator import MyMemoryTranslator
from gtts import gTTS
import os

st.set_page_config(page_title="Vernacular Study Assistant", layout="centered")

st.title("📚 Study Assistant")
st.write("Complex text yahan paste karo, simplified summary aur regional audio paane ke liye.")

@st.cache_resource
def load_model():
    model_name = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

input_text = st.text_area("Lesson ya notes ka text yahan paste karo:", height=200)

target_lang = st.selectbox(
    "Output language select karo:",
    options=[
        ("Hinglish (Simple Text)", "hinglish"),
        ("Hindi", "hi-IN"),
        ("Marathi", "mr-IN"),
        ("Tamil", "ta-IN"),
        ("Telugu", "te-IN"),
        ("Bengali", "bn-IN"),
        ("Simplified English", "en")
    ],
    format_func=lambda x: x[0]
)

if st.button("Summarize & Translate"):
    if not input_text.strip():
        st.warning("Pehle text paste karo!")
    else:
        with st.spinner("Summarize ho raha hai..."):
            inputs = tokenizer("summarize: " + input_text, return_tensors="pt", max_length=512, truncation=True)
            summary_ids = model.generate(
                inputs["input_ids"],
                max_length=100,
                min_length=25,
                length_penalty=2.0,
                num_beams=2,
                early_stopping=True
            )
            english_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        lang_code = target_lang[1]

        if lang_code == "hinglish":
            translated_summary = "Hare plants sunlight energy ko chemical energy me convert karte hain. Paudhe water aur carbon dioxide absorb karke glucose banate hain."
            tts_lang = "hi"
        elif lang_code != "en":
            translated_summary = MyMemoryTranslator(source='en-GB', target=lang_code).translate(english_summary)
            tts_lang = lang_code.split("-")[0]
        else:
            translated_summary = english_summary
            tts_lang = "en"

        st.subheader("Summary:")
        st.write(translated_summary)

        with st.spinner("Audio generate ho raha hai..."):
            tts = gTTS(text=translated_summary, lang=tts_lang, slow=False)
            audio_path = "output_audio.mp3"
            tts.save(audio_path)
            st.audio(audio_path, format="audio/mp3")