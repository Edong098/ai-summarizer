import streamlit as st # type: ignore
from services.summarizer import summarize_by_model, summarize_simple # type: ignore
from config import MODEL_CONFIG

# Page config
st.set_page_config(page_title="AI Summarizer", page_icon="🧠")

# CSS
st.markdown("""
<style>
textarea {border-radius: 10px;}
button {border-radius: 10px;}
</style>
""", unsafe_allow_html=True)

# Title
st.title("🧠 AI Text Summarizer")
st.write("Ringkas teks dengan AI 🚀")

# Input
text_input = st.text_area("Masukkan teks:", height=200)

# Method - using model keys from MODEL_CONFIG
method = st.selectbox("Pilih metode:", list(MODEL_CONFIG.keys()) + ["Simple"])

# Buttons
col1, col2 = st.columns(2)

with col1:
    clear = st.button("Hapus")

with col2:
    generate = st.button("Generate")

# Logic
if clear:
    st.rerun()

if generate:
    if text_input.strip() == "":
        st.warning("Teks kosong!")
    elif len(text_input.strip()) < 50:
        st.warning("Teks terlalu pendek! Minimal 50 karakter untuk hasil yang baik.")
    else:
        with st.spinner("Memproses... ⏳ (Pertama kali akan lebih lama karena model perlu didownload)"):
            if method == "Simple":
                result = summarize_simple(text_input)
            else:
                result = summarize_by_model(text_input, method)

        st.success(result)