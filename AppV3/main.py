import streamlit as st
import os 
import openai
from htmlTemplates import css, bot_template
import sqlite3
from langchain.chains import LLMChain
from langchain import OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


def create_transcripts_db():
    with sqlite3.connect('MASTER.db') as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS Transcripts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT,
                transcription TEXT,
                transcription_summary TEXT
            )
        """)


def insert_into_transcripts(file_name, transcription, transcription_summary):
    with sqlite3.connect('MASTER.db') as conn:
        cursor = conn.cursor()
        query = """
            INSERT INTO Transcripts (file_name, transcription, transcription_summary) 
            VALUES (?, ?, ?)
        """
        cursor.execute(query, (file_name, transcription, transcription_summary))
        conn.commit()


def get_transcript_ids_and_names():
    with sqlite3.connect('MASTER.db') as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, file_name FROM Transcripts")
        results = cursor.fetchall()
        return [f"{row[0]} - {row[1]}" for row in results]
    

def get_transcript_by_id(selection):
    id = int(selection.split(' - ')[0])
    with sqlite3.connect('MASTER.db') as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT transcription FROM Transcripts WHERE id = ?", (id,))
        result = cursor.fetchone()
        if result is not None: return result[0]
        else: return "No transcript found for the given id"


def get_summary_by_id(selection):
    id = int(selection.split(' - ')[0])
    with sqlite3.connect('MASTER.db') as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT transcription_summary FROM Transcripts WHERE id = ?", (id,))
        result = cursor.fetchone()
        if result is not None: return result[0]
        else: return "No transcript found for the given id"


def main():
    st.set_page_config(page_title="OpenAI Audio to Text")
    create_transcripts_db()
    st.write(css, unsafe_allow_html=True)
    st.session_state.setdefault("audio_file_path", None)
    st.session_state.setdefault("transcript", None)
    st.session_state.setdefault("transcript_summary")
    st.session_state.setdefault("prev_file_path", None)
    st.session_state.setdefault("prev_transcript", None)
    st.title("OpenAI Audio to Text")
    create_tab, prev_tab = st.tabs(["Create Transcription","Previous Transctiptions"])

    with create_tab:
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
                summary_prompt = PromptTemplate(
                    input_variables=['input'],
                    template='''
                    Summarize this audio transcript: 
                    <transcript>{input}</transcript>
                    '''
                )
                llm = OpenAI(temperature=0.65, model_name="gpt-4")
                summary_chain = LLMChain(llm=llm, 
                                         prompt=summary_prompt
                )
                st.session_state.transcript_summary = summary_chain.run(input=st.session_state.transcript)
                insert_into_transcripts(file_name=(st.session_state.audio_file_path.split("\\")[1]),
                                        transcription=st.session_state.transcript,
                                        transcription_summary=st.session_state.transcript_summary)
                

        if st.session_state.audio_file_path:
            if st.session_state.transcript:
                st.write(st.session_state.audio_file_path.split("\\")[1]+" ~ Transcription")
                st.markdown(bot_template.replace("{{MSG}}", st.session_state.transcript), unsafe_allow_html=True)
            if st.session_state.transcript_summary:
                st.write(st.session_state.audio_file_path.split("\\")[1]+" ~ Summary")
                st.markdown(bot_template.replace("{{MSG}}", st.session_state.transcript_summary), unsafe_allow_html=True)
            
    with prev_tab:
        transcript_selection = st.selectbox(label="Select Transcript", options=get_transcript_ids_and_names())
        if st.button("Render Transcript") and transcript_selection:
            st.session_state.prev_file_path = transcript_selection
            st.session_state.prev_transcript = get_transcript_by_id(transcript_selection)
            st.session_state.prev_transcript_summary = get_summary_by_id(transcript_selection)
        if st.session_state.prev_transcript:
            st.write(str(st.session_state.prev_file_path) + " ~ Transcription")
            st.markdown(bot_template.replace("{{MSG}}", st.session_state.prev_transcript), unsafe_allow_html=True)
            st.write(str(st.session_state.prev_file_path) + " ~ Summary")
            st.markdown(bot_template.replace("{{MSG}}", st.session_state.prev_transcript_summary), unsafe_allow_html=True)


if __name__ == "__main__":
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    main()
    
