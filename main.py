import streamlit as st
import torch
from models import build_model
from kokoro import generate
import time

page_title = "Voxify"
page_icon = "🗣️"
st.set_page_config(page_title=page_title, page_icon=page_icon, layout="centered")

voice_names = {'American': {'Female': ['Default', 'Bella', 'Sarah', 'Nicole', 'Sky'], 'Male': ['Adam', 'Michael']},
               'British': {'Female': ['Emma', 'Isabella'], 'Male': ['George', 'Lewis']},
               }
voice_names_model_mapping = {'Default':'af', 'Bella':'af_bella', 'Sarah':'af_sarah', 'Nicole':'af_nicole',
                             'Sky':'af_sky', 'Adam':'am_adam', 'Michael':'am_michael',
                             'Emma':'bf_emma', 'Isabella':'bf_isabella', 'George':'bm_george', 'Lewis':'bm_lewis'}
st.title(f'{page_title}🗨️🔉')
st.write('***:blue[🗣️ Turn Text into Voice, Your Way! 🎙️]***')
st.write("""
Voxify is a cutting-edge text-to-speech application that brings your words to life! 🌟 Choose from expressive male and 
female voices to create audio that resonates with your needs. Whether for storytelling, presentations, or accessibility, 
Voxify transforms text into natural, lifelike speech with ease. 📖➡️🔊
""")
st.info("[Powered by Kokoro-82M Text to Speech Model](https://huggingface.co/hexgrad/Kokoro-82M)", icon='ℹ️')
col1, col2 = st.columns(2, border=True)
with col1:
    st.subheader('Language Selection', divider="gray")
    language_selection = st.radio("Supported Language(s)", ["American English", "British English"],
                          index=0, horizontal=True, label_visibility='collapsed')
    selected_language = language_selection.split()[0]
with col2:
    st.subheader('Voice Type', divider="gray")
    voice_type = st.radio("Supported Voice Type", ["Female :female-office-worker:", "Male :male-office-worker:"],
                          index=0, horizontal=True, label_visibility='collapsed')
    selected_voice_type = voice_type.split()[0]
st.subheader('Voice Names:', divider='gray')
selected_voice_name = st.selectbox("Voice Names:", voice_names[selected_language][selected_voice_type],
                                   label_visibility='collapsed')
voice_pack = voice_names_model_mapping[selected_voice_name]
st.subheader('Input Text:', divider='gray')
text = st.text_area('Input Text', "It was the best of times, it was the worst of times, it was the age of "
    "wisdom, it was the age of foolishness, it was the epoch of belief, it "
    "was the epoch of incredulity, it was the season of Light, it was the "
    "season of Darkness, it was the spring of hope, it was the winter of "
    "despair.", label_visibility="collapsed")
generate_tts = st.button('Generate Audio', type='primary', icon=":material/text_to_speech:")
if generate_tts:
    with st.spinner('Generating Audio ...'):
        start_time = time.time()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        MODEL = build_model('kokoro-v0_19.pth', device)
        VOICE_PACK = torch.load(f'voices/{voice_pack}.pt', weights_only=True).to(device)
        audio, out_ps = generate(MODEL, text, VOICE_PACK, lang=voice_pack[0])
        end_time = time.time()
        time_taken = end_time - start_time
        st.toast(f"Audio Generated in {time_taken:.2f} seconds")
        st.subheader('Audio :loud_sound::', divider='gray')
        st.audio(audio, format="audio/mpeg", sample_rate=24000)
