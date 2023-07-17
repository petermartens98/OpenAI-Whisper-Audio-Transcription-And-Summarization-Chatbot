import streamlit as st
import os 
import re
from contextlib import closing
import openai
import sqlite3
import hashlib
import nltk
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from db_functions import create_db, add_user_to_db, authenticate_user, get_user_id, \
    insert_into_transcripts, get_transcript_ids_and_names, get_transcript_by_id, \
    get_summary_by_id, insert_audio, get_transcript_id, get_sentiment_by_id
from htmlTemplates import css, user_template, bot_template

    
def approve_password(password):
    if len(password) >= 8 and re.search(r"(?=.*[a-z])(?=.*[A-Z])(?=.*[0-9])(?=.*[_@$#!?&*%])", password):
        return True
    return False
    

def approve_email(email):
    email_regex = '^[a-zA-Z0-9]+[\._]?[a-zA-Z0-9]+[@]\w+[.]\w{2,3}$'
    if re.search(email_regex, email): return True
    else: return False


def user_authentication_tab():
    if st.session_state.user_authenticated:
        st.success("User Succesfully Authenticated")
        return

    with st.expander("User Authentication", expanded=True):
        login_tab, create_account_tab = st.tabs(["Login", "Create Account"])
        with login_tab: handle_login_tab()
        with create_account_tab: handle_create_account_tab()


def handle_login_tab():
    email = st.text_input("Email:") 
    password = st.text_input("Password:", type='password')
    if st.button("Login") and authenticate_user(email=email,password=password):
        st.session_state.user_authenticated = True
        st.session_state.user_id = get_user_id(email=email)
        st.experimental_rerun()


def handle_create_account_tab():
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
        st.caption("User Successfully Added")


def display_convo():
    with st.container():
        for i, message in enumerate(reversed(st.session_state.chat_history)):
            if i % 2 == 0: st.markdown(bot_template.replace("{{MSG}}", message), unsafe_allow_html=True)
            else: st.markdown(user_template.replace("{{MSG}}", message), unsafe_allow_html=True)


def display_prev_convo():
    with st.container():
        for i, message in enumerate(reversed(st.session_state.prev_chat_history)):
            if i % 2 == 0: st.markdown(bot_template.replace("{{MSG}}", message), unsafe_allow_html=True)
            else: st.markdown(user_template.replace("{{MSG}}", message), unsafe_allow_html=True)


def init_session_states():
    session_states = {
        "audio_file_path": None,
        "transcript": None,
        "transcript_summary": None,
        "sentiment_label": None,
        "prev_sentiment_label": None,
        "prev_file_path": None,
        "prev_transcript": None,
        "prev_transcript_summary": None,
        "user_id": None,
        "user_authenticated": False,
        "chat_history": [],
        "prev_chat_history": []
    }
    for state, default in session_states.items():
        st.session_state.setdefault(state, default)


def main():
    st.set_page_config(page_title="OpenAI Audio to Text")
    st.write(css, unsafe_allow_html=True)
    create_db()
    init_session_states()
    st.title("OpenAI Audio to Text")
    user_authentication_tab()
    if st.session_state.user_authenticated:
        create_tab, prev_tab = st.tabs(["Create Transcription","Previous Transctiptions"])
        with create_tab:
            uploaded_file = st.file_uploader("Upload Audio File", type=['mp3', 'mp4', 'mpeg', 'mpga', 
                                                                        'm4a', 'wav', 'webm'])
            if st.button("Generate Transcript and Summary") and uploaded_file:
                st.session_state.chat_history = []
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
                        template='Summarize this audio transcript: <transcript>{input}</transcript>'
                    )
                    sentiment_prompt = PromptTemplate(
                        input_variables=['transcript','summary'],
                        template='''
                            Return a single word sentiment of either ['positive','negative' or 'neutral'] from this transcript and summary.
                            \nTRANSCRIPT: {transcript}
                            \nTRANSCRIPT SUMMARY: {summary}
                            \nSENTIMENT LABEL HERE ('positive','negative', or 'neutral'):
                        '''
                    )
                    llm = OpenAI(temperature=0.65, model_name="gpt-4")
                    summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
                    st.session_state.transcript_summary = summary_chain.run(input=st.session_state.transcript)
                    sentiment_chain = LLMChain(llm=llm, prompt=sentiment_prompt)
                    st.session_state.sentiment_label = sentiment_chain.run(transcript=st.session_state.transcript,
                                                                           summary=st.session_state.transcript_summary)
                    insert_into_transcripts(file_name=(st.session_state.audio_file_path.split("\\")[1]),
                                            transcription=st.session_state.transcript,
                                            transcription_summary=st.session_state.transcript_summary,
                                            sentiment_label = st.session_state.sentiment_label,
                                            user_id=st.session_state.user_id
                    )
                    insert_audio(file_path=st.session_state.audio_file_path, 
                                 transcript_id=get_transcript_id(file_name=(st.session_state.audio_file_path.split("\\")[1]))
                    )
                    

            if st.session_state.audio_file_path:
                if st.session_state.transcript:
                    st.subheader(st.session_state.audio_file_path.split("\\")[1])
                    with st.expander("Transcription", expanded=True):
                        st.write(st.session_state.transcript)
                    if st.session_state.transcript_summary:
                        with st.expander("Summary", expanded=True):
                            st.write(st.session_state.transcript_summary)
                        with st.expander("Sentiment Analysis", expanded=True):
                            st.write(f"Sentiment Label: {st.session_state.sentiment_label}")
                        with st.expander("Text Statistics", expanded=True):
                            st.write(f"Transcription Word Count: {len(st.session_state.transcript.split())}")
                            st.write(f"Transcription Character Count: {len(st.session_state.transcript)}")
                        st.subheader("Chat with Transctiption")
                        user_message = st.text_input("User Message", key='unique_key1')
                        if st.button("Submit Message") and user_message:
                            with st.spinner("Generating Response..."):
                                chat_template = PromptTemplate(
                                    input_variables=['transcript','summary','chat_history','user_message'],
                                    template='''
                                        You are an AI chatbot intended to discuss about the user's audio transcription.
                                        \nTRANSCRIPT: "{transcript}"
                                        \nTRANSCIRPT SUMMARY: "{summary}"
                                        \nCHAT HISTORY: {chat_history}
                                        \nUSER MESSAGE: {user_message}
                                        \nAI RESPONSE HERE:
                                    '''
                                )
                                chat_llm = ChatOpenAI(model='gpt-4',temperature=0.7)
                                chat_llm_chain = LLMChain(llm=chat_llm, 
                                                        prompt=chat_template)
                                ai_response = chat_llm_chain.run(
                                    transcript=st.session_state.transcript,
                                    summary=st.session_state.transcript_summary,
                                    chat_history=st.session_state.chat_history,
                                    user_message=user_message
                                )
                                st.session_state.chat_history.append(f"USER: {user_message}")
                                st.session_state.chat_history.append(f"AI: {ai_response}")
                                
                        if st.session_state.chat_history:
                            display_convo()

                
        with prev_tab:
            transcript_selection = st.selectbox(label="Select Transcript", options=get_transcript_ids_and_names())
            if st.button("Render Transcript") and transcript_selection:
                transcript = get_transcript_by_id(transcript_selection)
                summary = get_summary_by_id(transcript_selection)
                sentiment_label = get_sentiment_by_id(transcript_selection)
                st.session_state.update(
                    prev_file_path=transcript_selection,
                    prev_transcript=transcript,
                    prev_transcript_summary=summary,
                    prev_chat_history=[],
                    prev_sentiment_label=sentiment_label
                )
            if st.session_state.prev_transcript:
                st.subheader(st.session_state.prev_file_path)
                with st.expander("Transcription", expanded=True):
                    st.write(st.session_state.prev_transcript)
                if st.session_state.prev_transcript_summary:
                    with st.expander("Summary", expanded=True):
                        st.write(st.session_state.prev_transcript_summary)
                    with st.expander("Sentiment Analysis", expanded=True):
                        st.write(f"Sentiment Label: {st.session_state.prev_sentiment_label}")
                    with st.expander("Text Statistics", expanded=True):
                        st.write(f"Transcription Word Count: {len(st.session_state.prev_transcript.split())}")
                        st.write(f"Transcription Character Count: {len(st.session_state.prev_transcript)}")
                    st.subheader("Chat with Transctiption")
                    pc_user_message = st.text_input("User Message", key='unique_key2')
                    if st.button("Submit Message", key="button2") and pc_user_message:
                        with st.spinner("Generating Response..."):
                            chat_template = PromptTemplate(
                                input_variables=['transcript','summary','chat_history','user_message'],
                                template='''
                                    You are an AI chatbot intended to discuss about the user's audio transcription.
                                    \nTRANSCRIPT: "{transcript}"
                                    \nTRANSCIRPT SUMMARY: "{summary}"
                                    \nCHAT HISTORY: {chat_history}
                                    \nUSER MESSAGE: {user_message}
                                    \nAI RESPONSE HERE:
                                '''
                            )
                            chat_llm = ChatOpenAI(model='gpt-4',temperature=0.7)
                            chat_llm_chain = LLMChain(llm=chat_llm, 
                                                    prompt=chat_template)
                            ai_response = chat_llm_chain.run(
                                transcript=st.session_state.prev_transcript,
                                summary=st.session_state.prev_transcript_summary,
                                chat_history=st.session_state.prev_chat_history,
                                user_message=pc_user_message
                            )
                            st.session_state.prev_chat_history.append(f"USER: {pc_user_message}")
                            st.session_state.prev_chat_history.append(f"AI: {ai_response}")
                            
                    if st.session_state.prev_chat_history:
                        display_prev_convo()


if __name__ == "__main__":
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    main()
    
