import streamlit as st
from langchain.prompts import PromptTemplate

chat_template = PromptTemplate(
    input_variables=['transcript','summary','chat_history','user_message', 'sentiment_report'],
    template='''
        You are an AI chatbot intended to discuss about the user's audio transcription.
        \nTRANSCRIPT: "{transcript}"
        \nTRANSCIRPT SUMMARY: "{summary}"
        \nTRANSCRIPT SENTIMENT REPORT: "{sentiment_report}"
        \nCHAT HISTORY: {chat_history}
        \nUSER MESSAGE: "{user_message}"
        \nAI RESPONSE HERE:
    '''
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

fact_check_prompt = '''
        Fact-check this transcript for factual or logical inacurracies or inconsistencies
        \nWrite a report on the factuality / logic of the transcirpt
        \nTRANSCRIPT: {}
        \nTRANSCRIPT SUMMARY: {}
        \nAI FACT CHECK RESPONSE HERE:
'''