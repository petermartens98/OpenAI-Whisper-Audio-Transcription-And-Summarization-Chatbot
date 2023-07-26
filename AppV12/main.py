import streamlit as st
from audio_recorder_streamlit import audio_recorder
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
import os 
import re
import pinecone
import openai
import pandas as pd
from collections import Counter
from pydub import AudioSegment
from langchain.tools import PubmedQueryRun
from langchain import LLMMathChain
from langchain.vectorstores import Chroma, Pinecone
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import Tool
from langchain.chains import VectorDBQA
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import YouTubeSearchTool
from langchain.memory import ConversationBufferMemory

from db_functions import create_db, add_user_to_db, authenticate_user, get_user_id, \
    insert_into_transcripts, get_transcript_ids_and_names, get_transcript_by_id, \
    get_summary_by_id, insert_audio, get_transcript_id, get_sentiment_by_id, get_sentiment_report_by_id, \
    get_fact_check_by_id, get_ai_qa_by_id
from prompts import chat_template, fact_check_prompt, sentiment_prompt
from htmlTemplates import css, user_template, bot_template


# TODO: Segment Audion: Insert time stamps into transcription

# TODO: Improve Pinecone MetaData and Organization


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
        "transcript": "",
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
        "current_ai_research": "",
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


def define_tools():
    wiki = WikipediaAPIWrapper()
    DDGsearch = DuckDuckGoSearchRun()
    YTsearch = YouTubeSearchTool()
    pubmed = PubmedQueryRun()
    llm_math_chain = LLMMathChain(llm=llm, verbose=True)

    tools = [
        Tool(
            name = "Wikipedia Research Tool",
            func=wiki.run,
            description="Useful for researching older information and checking facts on wikipedia"
        ),
        Tool(
            name = "DuckDuckGo Internet Search Tool",
            func = DDGsearch.run,
            description="Useful for researching newer information and checking facts on the internet"
        ),
        Tool(
            name = "YouTube Links Tool",
            func = YTsearch.run,
            description="Useful for gathering links on YouTube"
        ),
        Tool(
            name='Vector-Based Previous Transcript / Information Database Tool',
            func=qa.run,
            description='Provides access to previous transcripts and related data'
        ),
        Tool(
            name ='Calculator and Math Tool',
            func=llm_math_chain.run,
            description='Useful for mathematical questions and operations'
        ),
        Tool(
            name='Pubmed Science and Medical Journal Research Tool',
            func=pubmed.run,
            description='Useful for Pubmed science and medical research'
        )
    ]
    return tools

    

# Upload audio files for file or voice
def upload_audio_tab():
    global uploaded_file
    os.makedirs(upload_dir, exist_ok=True)
    upload_mode = st.radio("Upload Mode", options=['File Upload', 'Voice Record'])
    uploaded_file = None
    
    if upload_mode == 'File Upload':
        uploaded_file = st.file_uploader("Upload Audio File", type=['mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'wav', 'webm'])
        if uploaded_file is not None:
            audio_bytes = uploaded_file.read()
            st.audio(audio_bytes, format="audio/wav")

    elif upload_mode == 'Voice Record':
        audio_bytes = audio_recorder(text="Record")
        if audio_bytes:
            file_path = os.path.join(upload_dir, 'audio_record.wav')
            with open(file_path, 'wb') as fp:
                fp.write(audio_bytes)
            st.audio(audio_bytes, format="audio/wav")
            uploaded_file = file_path


# Audio File Processing
def process_file():
    with st.spinner('Processing File...'):
        if isinstance(uploaded_file, str):
            st.session_state.audio_file_path = uploaded_file
        else:
            file_path = os.path.join(upload_dir, uploaded_file.name)
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.audio_file_path = file_path


# Audio Transcription

# GPT4 Audio Post Processing
def generate_corrected_transcript(transcript):
    system_prompt = '''
        You are a helpful AI assistant, intended to fix any spelling or grammar mistakes in user audio transcript.
        \nIf words appear incorrect or there are run-on word, fix the transcript the best you can.   
    '''
    response = openai.ChatCompletion.create(
        model=MODEL,
        temperature=TEMP,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": transcript
            }
        ]
    )
    return response['choices'][0]['message']['content']


def transcribe_audio():
    with st.spinner('Transcribing Audio...'): 
        with open(st.session_state.audio_file_path, 'rb') as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)['text']
            st.session_state.transcript = generate_corrected_transcript(transcript)


def display_transcript():
    with st.expander("Transcription", expanded=True):
        transcript = st.session_state.prev_transcript if is_prev_tab else st.session_state.transcript
        st.write(transcript)


# Transcript Summarization

def map_reduce_summarize_text(input):
    try:
        text_splitter = CharacterTextSplitter()
        texts = text_splitter.split_text(input)
        docs = [Document(page_content=t) for t in texts]
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        return chain.run(docs)
    except Exception as e:
        return(f"An error occured: {e}")

def summarize_transcript():
    with st.spinner("Generting Summary..."):
        st.session_state.transcript_summary = map_reduce_summarize_text(st.session_state.transcript)


def display_summary():
    with st.expander("Summary", expanded=True):
        transcript_summary = st.session_state.prev_transcript_summary if is_prev_tab else st.session_state.transcript_summary
        st.write(transcript_summary)


# Fact Check Transcript

def fact_check_transcript():
    zsrd_agent = create_zrsd_agent()
    with st.spinner("Fact Checking..."):
        st.session_state.fact_check = zsrd_agent.run(fact_check_prompt.format(st.session_state.transcript, st.session_state.transcript_summary))
    

def display_fact_check():
    with st.expander("Fact Check", expanded=True):
        fact_check = st.session_state.prev_fact_check if is_prev_tab else st.session_state.fact_check
        st.write(fact_check)


# Sentiment Analysis

def analyze_sentiment():
    with st.spinner("Analyzing Sentiment..."):
        sentiment_chain = LLMChain(llm=llm, prompt=sentiment_prompt)
        sentiment_results = sentiment_chain.run(transcript=st.session_state.transcript, summary=st.session_state.transcript_summary).split(",")
        st.session_state.sentiment_label = sentiment_results[0]
        st.session_state.sentiment_report = "".join(sentiment_results[1:])


def display_sentiment():
    with st.expander("Sentiment Analysis", expanded=True):
        if is_prev_tab:
            st.write(f"Sentiment Label: {st.session_state.prev_sentiment_label}")
            st.write(f"Sentiment Report: {st.session_state.prev_sentiment_report}")
        else:
            st.write(f"Sentiment Label: {st.session_state.sentiment_label}")
            st.write(f"Sentiment Report: {st.session_state.sentiment_report}")


# Vector DB QA Search

def qa_search():
    with st.spinner("Refering to Previous Transcripts..."):
        st.session_state.current_ai_research = qa.run(f'''
            \nReferring to previous results and information, 
            write relating to this summary: <summary>{st.session_state.transcript_summary}</summary>
        ''')


def display_qa():
    with st.expander("Previous Related Information (Pinecone Retrieval QA)", expanded=True):
        ai_research = st.session_state.prev_ai_research if is_prev_tab else st.session_state.current_ai_research
        st.write(ai_research)


# Text Stats / Metrics

def text_stats():
    with st.expander("Text Statistics", expanded=True):
        transcript = st.session_state.prev_transcript if is_prev_tab else st.session_state.transcript
        st.write(f"Transcription Word Count: {len(transcript.split())}")
        st.write(f"Transcription Character Count: {len(transcript)}")
        st.write("Word Frequency:")
        st.dataframe(get_word_frequency(transcript), height=200, width=300)


# sidebar function
def sidebar():
    global TEMP, MODEL
    with st.sidebar:
        with st.expander("Settings", expanded=True):
            TEMP = st.slider(label='LLM Temperature', min_value=0.0, max_value=1.0, value=0.7)
            MODEL = st.selectbox(label='LLM Model', options=['gpt-4','gpt-3.5-turbo'])


def display_results():
    display_transcript()
    display_summary()
    display_fact_check()
    display_sentiment()
    display_qa()
    text_stats()


def generate_and_display_results():
    transcribe_audio()
    display_transcript()
    summarize_transcript()
    display_summary()
    fact_check_transcript()
    display_fact_check()
    analyze_sentiment()
    display_sentiment()
    qa_search()
    display_qa()
    text_stats()


def create_zrsd_agent():
    tools = define_tools()
    memory = ConversationBufferMemory(memory_key="chat_history")
    zsrd_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, memory=memory)
    return zsrd_agent


def create_qa():
    embedding_function = OpenAIEmbeddings()
    vectorstore = Pinecone.from_existing_index(index_name, embedding_function)
    return VectorDBQA.from_chain_type(llm=llm,vectorstore=vectorstore)


# Main Function

def main():
    global qa,  llm, is_prev_tab, uploaded_file, upload_dir, index_name
    upload_dir = 'uploads'
    st.set_page_config(page_title="Whisper Transcription ChatBot")
    st.write(css, unsafe_allow_html=True)
    pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv('PINECONE_ENVIORNMENT'))
    index_name='index1'
    create_db()
    init_session_states()
    st.title("OpenAI Transcription Tool")
    user_authentication_tab()
    if st.session_state.user_authenticated:
        sidebar()
        llm = OpenAI(temperature=TEMP, model_name=MODEL)
        embedding_function = OpenAIEmbeddings()
        qa =  create_qa()
        chat_llm_chain = LLMChain(llm=llm, prompt=chat_template)
        create_tab, prev_tab = st.tabs(["Create Transcription","Previous Transctiptions"])
        with create_tab:
            is_prev_tab = False
            upload_audio_tab() 
            if uploaded_file is not None:
                if st.button("Generate Transcript and Summary"):
                    st.session_state.chat_history = []
                    process_file()
                    st.subheader(st.session_state.audio_file_path.split("\\")[1])
                    generate_and_display_results()
                    insert_into_transcripts(file_name=(st.session_state.audio_file_path.split("\\")[1]),
                                            transcription=st.session_state.transcript,
                                            transcription_summary=st.session_state.transcript_summary,
                                            sentiment_label = st.session_state.sentiment_label,
                                            sentiment_report = st.session_state.sentiment_report,
                                            user_id=st.session_state.user_id,
                                            prev_ai_research=st.session_state.current_ai_research,
                                            fact_check=st.session_state.fact_check
                    )
                    insert_audio(file_path=st.session_state.audio_file_path, 
                                transcript_id=get_transcript_id(file_name=(st.session_state.audio_file_path.split("\\")[1]))
                    )
                    transcript_texts = [st.session_state.transcript,
                                        st.session_state.transcript_summary,
                                        st.session_state.sentiment_label,
                                        st.session_state.sentiment_report,
                                        st.session_state.current_ai_research,
                                        st.session_state.fact_check]
                    Pinecone.from_texts(transcript_texts, embedding_function, index_name=index_name)
                    st.experimental_rerun()
                        

                if st.session_state.audio_file_path and st.session_state.transcript:
                    st.subheader(st.session_state.audio_file_path.split("\\")[1])
                    display_results()     
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
                            # Store Messages in Sqlite and PineCone

                    if st.session_state.chat_history:
                        display_convo()

                
        with prev_tab:
            is_prev_tab = True
            transcript_selection = st.selectbox(label="Select Transcript", options=get_transcript_ids_and_names())
            if st.button("Render Transcript") and transcript_selection:
                st.session_state.update(
                    prev_file_path = transcript_selection,
                    prev_transcript=get_transcript_by_id(transcript_selection),
                    prev_transcript_summary = get_summary_by_id(transcript_selection),
                    prev_chat_history=[],
                    prev_sentiment_label = get_sentiment_by_id(transcript_selection),
                    prev_sentiment_report = get_sentiment_report_by_id(transcript_selection),
                    prev_fact_check = get_fact_check_by_id(transcript_selection),
                    prev_ai_research = get_ai_qa_by_id(transcript_selection)
                )
                # TODO: Render Previous Audio
            if st.session_state.prev_transcript:
                st.subheader(st.session_state.prev_file_path)
                display_results()
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
                        # TODO: Store Messages in Sqlite and PineCone
                        
                if st.session_state.prev_chat_history:
                    display_prev_convo()


if __name__ == "__main__":
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    main()
    
