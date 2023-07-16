import streamlit as st
import os 
import re
import openai
from htmlTemplates import css, bot_template
import sqlite3
from langchain.chains import LLMChain
from langchain import OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


def create_users_db():
    conn = sqlite3.connect('MASTER.db')
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()


def add_user_to_db(email, password):
    conn = sqlite3.connect('MASTER.db')
    cursor = conn.cursor()
    insert_query = """
        INSERT INTO Users (email, password)
        VALUES (?, ?)
    """
    cursor.execute(insert_query, (email, password))
    conn.commit()
    conn.close()


def authenticate_user(email, password):
    conn = sqlite3.connect('MASTER.db')
    cursor = conn.cursor()
    select_query = """
        SELECT * FROM Users WHERE email = ? AND password = ?
    """
    cursor.execute(select_query, (email, password))
    user = cursor.fetchone()
    conn.close()
    if user:
        return True
    else:
        return False
    

def get_user_id(email):
    conn = sqlite3.connect('MASTER.db')
    cursor = conn.cursor()
    select_query = """
        SELECT user_id FROM Users WHERE email = ?
    """
    cursor.execute(select_query, (email,))
    user_id = cursor.fetchone()
    conn.close()
    return user_id[0] if user_id else None

    

def approve_password(password):
    if len(password) >= 8 and re.search(r"(?=.*[a-z])(?=.*[A-Z])(?=.*[0-9])(?=.*[_@$#!?&*%])", password):
        return True
    return False
    

def approve_email(email):
    email_regex = '^[a-zA-Z0-9]+[\._]?[a-zA-Z0-9]+[@]\w+[.]\w{2,3}$'
    if re.search(email_regex, email):
        return True
    else:
        return False


def create_transcripts_db():
    with sqlite3.connect('MASTER.db') as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS Transcripts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT,
                transcription TEXT,
                transcription_summary TEXT,
                user_id INTEGER,
                FOREIGN KEY(user_id) REFERENCES Users(user_id)
            )
        """)


def insert_into_transcripts(user_id, file_name, transcription, transcription_summary):
    with sqlite3.connect('MASTER.db') as conn:
        cursor = conn.cursor()
        query = """
            INSERT INTO Transcripts (user_id, file_name, transcription, transcription_summary) 
            VALUES (?, ?, ?, ?)
        """
        cursor.execute(query, (user_id, file_name, transcription, transcription_summary))
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
        cursor.execute("SELECT transcription_summary FROM Transcripts WHERE id = ? AND user_id = ?", 
                       (id, st.session_state.user_id))
        result = cursor.fetchone()
        if result is not None: return result[0]
        else: return "No transcript found for the given id"


def user_authentication_tab():
    if st.session_state.user_authenticated:
        st.success("User Succesfully Authenticated")
    else:
        with st.expander("User Authentication", expanded=True):
            login_tab, create_account_tab = st.tabs(["Login", "Create Account"])

            with login_tab:
                email = st.text_input("Email:") 
                password = st.text_input("Password:", type='password')
                if st.button("Login"):
                    if authenticate_user(email=email,password=password):
                        st.session_state.user_authenticated = True
                        st.session_state.user_id = get_user_id(email=email)
                        st.experimental_rerun()
                    else:
                        st.caption('Incorrect Username or Password.')


            with create_account_tab:
                new_email = st.text_input("New Email:")
                new_password = st.text_input("New Password:", type='password')
                confirm_password = st.text_input("Confirm Password:", type='password')
                if st.button("Create Account"):
                    if not approve_email(new_email):
                        st.caption("Invalid Email")
                        return
                    if not approve_password(new_password):
                        st.caption("Invalid Password")
                        return
                    if new_password != confirm_password:
                        st.caption("Passwords do not match")
                        return
                    add_user_to_db(email=new_email, password=new_password)
                    st.caption(f"{new_email} Successfully Added")


def main():
    st.set_page_config(page_title="OpenAI Audio to Text")
    create_users_db()
    create_transcripts_db()
    st.write(css, unsafe_allow_html=True)
    st.session_state.setdefault("audio_file_path", None)
    st.session_state.setdefault("transcript", None)
    st.session_state.setdefault("transcript_summary")
    st.session_state.setdefault("prev_file_path", None)
    st.session_state.setdefault("prev_transcript", None)
    st.session_state.setdefault("user_id", None)
    st.session_state.setdefault("user_authenticated", False)
    st.title("OpenAI Audio to Text")
    user_authentication_tab()

    if st.session_state.user_authenticated:
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

                    # Summarize Transcripts
                    insert_into_transcripts(file_name=(st.session_state.audio_file_path.split("\\")[1]),
                                            transcription=st.session_state.transcript,
                                            transcription_summary=st.session_state.transcript_summary,
                                            user_id=st.session_state.user_id)
                    

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
    
