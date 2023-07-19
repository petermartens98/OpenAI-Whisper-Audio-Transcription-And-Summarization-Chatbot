import streamlit as st
from audio_recorder_streamlit import audio_recorder
from dotenv import load_dotenv
import os 
import re
import openai
import chromadb
from supabase import create_client, Client
from langchain.vectorstores import Chroma
import pandas as pd
from collections import Counter
import vecs
import supabase
import plotly
import plotly.graph_objs as go
from langchain.chains import VectorDBQA
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from db_functions import create_db, add_user_to_db, authenticate_user, get_user_id, \
    insert_into_transcripts, get_transcript_ids_and_names, get_transcript_by_id, \
    get_summary_by_id, insert_audio, get_transcript_id, get_sentiment_by_id, get_sentiment_report_by_id, \
    get_fact_check_by_id
from htmlTemplates import css, user_template, bot_template


# TODO: Implement Column Layout - transcription on left, Chat on right

# TODO: LangChain Agents for further research

# TODO: Segment Audio and Revise SQL Schema

# TODO: Insert time stamps into transcription

# TODO: time stamps to pandas df to analysis and visualizations

# TODO: Post Processing and Correction

# TODO: Word Cloud Visualization

#############################
# Experiment With Qdrant for VectorDB ?
# Experiment With PineCone for VectorDB ?
# Experiment With Chroma for VectorDB ?
#############################

    
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
        "sentiment_report": None,
        "prev_sentiment_label": None,
        "prev_file_path": None,
        "prev_transcript": None,
        "prev_transcript_summary": None,
        "user_id": None,
        "user_authenticated": False,
        "chat_history": [],
        "prev_chat_history": [],
        "embeddings_db": None,
        "prev_ai_research": "",
        "fact_check": None,
        "prev_fact_check": None
    }
    for state, default in session_states.items():
        st.session_state.setdefault(state, default)


def get_word_frequency(text):
    cleaned_text = re.sub(r'[^\w\s]', '', text.lower())
    words = cleaned_text.split()
    word_freq = Counter(words)
    df_word_freq = pd.DataFrame(list(word_freq.items()), columns=['Word', 'Frequency'])
    df_word_freq = df_word_freq.sort_values(by='Frequency', ascending=False).reset_index(drop=True)
    return df_word_freq


def main():
    st.set_page_config(page_title="Whisper Transcription ChatBot")
    st.write(css, unsafe_allow_html=True)
    embedding_function = OpenAIEmbeddings()
    create_db()
    init_session_states()
    st.title("OpenAI Transcription Tool")
    user_authentication_tab()
    if st.session_state.user_authenticated:
        chat_template = PromptTemplate(
            input_variables=['transcript','summary','chat_history','user_message', 'sentiment_report'],
            template='''
                You are an AI chatbot intended to discuss about the user's audio transcription.
                \nTRANSCRIPT: "{transcript}"
                \nTRANSCIRPT SUMMARY: "{summary}"
                \nTRANSCRIPT SENTIMENT REPORT: {sentiment_report}
                \nCHAT HISTORY: {chat_history}
                \nUSER MESSAGE: {user_message}
                \nAI RESPONSE HERE:
            '''
        )
        chat_llm = ChatOpenAI(model='gpt-4',temperature=0.7)
        chat_llm_chain = LLMChain(llm=chat_llm, 
                                prompt=chat_template)
        create_tab, prev_tab = st.tabs(["Create Transcription","Previous Transctiptions"])
        with create_tab:
            upload_dir = 'uploads'
            os.makedirs(upload_dir, exist_ok=True)
            upload_mode = st.radio("Upload Mode", options=['File Upload', 'Voice Record'],horizontal=True)
            if upload_mode == 'File Upload':
                uploaded_file = st.file_uploader("Upload Audio File", type=['mp3', 'mp4', 'mpeg', 'mpga', 
                                                                            'm4a', 'wav', 'webm'])
                if uploaded_file is not None:
                    audio_bytes = uploaded_file.read()
                    st.audio(audio_bytes, format="audio/wav")

            if upload_mode == 'Voice Record':
                audio_bytes = audio_recorder(text="Record")
                if audio_bytes:
                    file_path = os.path.join(upload_dir, 'audio_record.wav')
                    with open(file_path, 'wb') as fp:
                        fp.write(audio_bytes)
                    st.audio(audio_bytes, format="audio/wav")
                    uploaded_file = file_path
            

            if st.button("Generate Transcript and Summary") and uploaded_file:
                st.session_state.chat_history = []
                with st.spinner('Processing...'):
                    if isinstance(uploaded_file, str):
                        st.session_state.audio_file_path = uploaded_file
                    else:
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
                            Return a single word sentiment of either ['Positive','Negative' or 'Neutral'] from this transcript and summary.
                            After that single word sentiment, add a comma, then return a sentiment report, analyzing transcript sentiment.
                            \nTRANSCRIPT: {transcript}
                            \nTRANSCRIPT SUMMARY: {summary}
                            \nSENTIMENT LABEL HERE ('Positive','Negative', or 'Neutral') <comma-seperated> REPORT HERE:
                        '''
                    )
                    llm = OpenAI(temperature=0.65, model_name="gpt-4")
                    summary_chain = LLMChain(llm=llm, prompt=summary_prompt)
                    st.session_state.transcript_summary = summary_chain.run(input=st.session_state.transcript)
                    sentiment_chain = LLMChain(llm=llm, prompt=sentiment_prompt)
                    sentiment_results = sentiment_chain.run(transcript=st.session_state.transcript,
                                                                           summary=st.session_state.transcript_summary).split(",")
                    st.session_state.sentiment_label = sentiment_results[0]
                    st.session_state.sentiment_report = "".join(sentiment_results[1:])
                    if st.session_state.embeddings_db is not None:
                        qa = VectorDBQA.from_chain_type(llm=llm,
                                                        vectorstore=st.session_state.embeddings_db)
                        st.session_state.prev_ai_research = qa.run(f'''
                            \nReferring to previous results and information, 
                            write relating to this summary: <summary>{st.session_state.transcript_summary}</summary>
                        ''')
                    
                    fact_check_prompt = PromptTemplate(
                        input_variables=['transcript', 'summary'],
                        template='''
                            Fact-check this transcript for factual or logical inacurracies or inconsistencies
                            \nWrite a report on the factuality / logic of the transcirpt
                            \nTRANSCRIPT: {transcript}
                            \nTRANSCRIPT SUMMARY: {summary}
                            \nAI FACT CHECK RESPONSE HERE
                        '''
                    )
                    # TODO: 
                    fact_check_chain = LLMChain(llm=llm, prompt=fact_check_prompt)
                    st.session_state.fact_check = fact_check_chain.run(transcript=st.session_state.transcript,
                                                                       summary=st.session_state.transcript_summary)

                    # TODO: Related Research LangChain Agents
                    insert_into_transcripts(file_name=(st.session_state.audio_file_path.split("\\")[1]),
                                            transcription=st.session_state.transcript,
                                            transcription_summary=st.session_state.transcript_summary,
                                            sentiment_label = st.session_state.sentiment_label,
                                            sentiment_report = st.session_state.sentiment_report,
                                            user_id=st.session_state.user_id,
                                            prev_ai_research=st.session_state.prev_ai_research,
                                            fact_check=st.session_state.fact_check
                    )
                    insert_audio(file_path=st.session_state.audio_file_path, 
                                 transcript_id=get_transcript_id(file_name=(st.session_state.audio_file_path.split("\\")[1]))
                    )
                    transcript_texts = [st.session_state.audio_file_path.split("\\")[1],
                                           st.session_state.transcript,
                                           st.session_state.transcript_summary,
                                           st.session_state.sentiment_label,
                                           st.session_state.sentiment_report]
                    embedding_function = OpenAIEmbeddings()
                    # Insert into vectoDB?
                  
                    

            if st.session_state.audio_file_path:
                if st.session_state.transcript:
                    st.subheader(st.session_state.audio_file_path.split("\\")[1])
                    with st.expander("Transcription", expanded=True):
                        st.write(st.session_state.transcript)
                    if st.session_state.transcript_summary:
                        with st.expander("Summary", expanded=True):
                            st.write(st.session_state.transcript_summary)
                        with st.expander("Fact Check", expanded=True):
                            st.write(st.session_state.fact_check)
                        with st.expander("Sentiment Analysis", expanded=True):
                            st.write(f"Sentiment Label: {st.session_state.sentiment_label}")
                            st.write(f"Sentiment Report: {st.session_state.sentiment_report}")
                        with st.expander("Text Statistics", expanded=True):
                            st.write(f"Transcription Word Count: {len(st.session_state.transcript.split())}")
                            st.write(f"Transcription Character Count: {len(st.session_state.transcript)}")
                            st.write("Word Frequency:")
                            st.dataframe(get_word_frequency(st.session_state.transcript), height=200, width=300)
                        st.subheader("Chat with Transctiption")
                        user_message = st.text_input("User Message", key='unique_key1')
                        if st.button("Submit Message") and user_message:
                            with st.spinner("Generating Response..."):
                                ai_response = chat_llm_chain.run(
                                    transcript=st.session_state.transcript,
                                    summary=st.session_state.transcript_summary,
                                    chat_history=st.session_state.chat_history,
                                    user_message=user_message,
                                    sentiment_report=st.session_state.sentiment_report
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
                sentiment_report = get_sentiment_report_by_id(transcript_selection)
                fact_check = get_fact_check_by_id(transcript_selection)
                st.session_state.update(
                    prev_file_path=transcript_selection,
                    prev_transcript=transcript,
                    prev_transcript_summary=summary,
                    prev_chat_history=[],
                    prev_sentiment_label=sentiment_label,
                    prev_sentiment_report=sentiment_report,
                    prev_fact_check=fact_check
                )
                # TODO: Render Previous Audio
            if st.session_state.prev_transcript:
                st.subheader(st.session_state.prev_file_path)
                with st.expander("Transcription", expanded=True):
                    st.write(st.session_state.prev_transcript)
                if st.session_state.prev_transcript_summary:
                    with st.expander("Summary", expanded=True):
                        st.write(st.session_state.prev_transcript_summary)
                    
                    with st.expander("Fact Check", expanded=True):
                        st.write(st.session_state.prev_fact_check)

                    with st.expander("Sentiment Analysis", expanded=True):
                        st.write(f"Sentiment Label: {st.session_state.prev_sentiment_label}")
                        st.write(f"Sentiment Report: {st.session_state.prev_sentiment_report}")

                    with st.expander("Text Statistics", expanded=True):
                        st.write(f"Transcription Word Count: {len(st.session_state.prev_transcript.split())}")
                        st.write(f"Transcription Character Count: {len(st.session_state.prev_transcript)}")
                        st.write("Word Frequency:")
                        st.dataframe(get_word_frequency(st.session_state.prev_transcript), height=200, width=300)

                    st.subheader("Chat with Transctiption")
                    pc_user_message = st.text_input("User Message", key='unique_key2')
                    if st.button("Submit Message", key="button2") and pc_user_message:
                        with st.spinner("Generating Response..."):
                            ai_response = chat_llm_chain.run(
                                transcript=st.session_state.prev_transcript,
                                summary=st.session_state.prev_transcript_summary,
                                chat_history=st.session_state.prev_chat_history,
                                user_message=pc_user_message,
                                sentiment_report=st.session_state.prev_sentiment_report
                            )
                            st.session_state.prev_chat_history.append(f"USER: {pc_user_message}")
                            st.session_state.prev_chat_history.append(f"AI: {ai_response}")
                            
                    if st.session_state.prev_chat_history:
                        display_prev_convo()


if __name__ == "__main__":
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    main()
    
