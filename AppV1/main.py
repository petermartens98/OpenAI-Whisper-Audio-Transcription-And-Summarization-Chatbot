import streamlit as st
import os 
import openai
from htmlTemplates import css, bot_template

def main():
    st.set_page_config(page_title="OpenAI Audio to Text")
    st.write(css, unsafe_allow_html=True)
    st.session_state.setdefault("audio_file_path", None)
    st.session_state.setdefault("transcript", None)
    st.title("OpenAI Audio to Text")

    uploaded_file = st.file_uploader("Upload Audio File", type=['mp3', 'mp4', 'mpeg', 'mpga', 
                                                                'm4a', 'wav', 'webm'])

    if st.button("Generate Transcript") and uploaded_file:
        with st.spinner('Processing...'):
            upload_dir = 'uploads'
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, uploaded_file.name)
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.audio_file_path = file_path
            with open(st.session_state.audio_file_path, 'rb') as audio_file:
                st.session_state.transcript = openai.Audio.transcribe("whisper-1", audio_file)['text']

    if st.session_state.audio_file_path:
        st.subheader("Audio Transcriptions")
        st.write(st.session_state.audio_file_path.split("\\")[1])
        if st.session_state.transcript:
            st.markdown(bot_template.replace("{{MSG}}", st.session_state.transcript), unsafe_allow_html=True)

if __name__ == "__main__":
    openai.api_key = os.getenv("OPENAI_API_KEY")
    main()
    
