import sqlite3
import hashlib
from contextlib import closing
import streamlit as st
import os
from sqlite3 import connect


def create_db():
    with sqlite3.connect('MASTER.db') as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS Users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT,
                password TEXT
            );
            CREATE TABLE IF NOT EXISTS Transcripts (
                transcript_id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT,
                transcription TEXT,
                transcription_summary TEXT,
                sentiment_label TEXT,
                sentiment_report TEXT,
                prev_ai_research TEXT,
                fact_check TEXT,
                user_id INTEGER,
                FOREIGN KEY(user_id) REFERENCES Users(user_id)
            );
            CREATE TABLE IF NOT EXISTS AudioFiles (
                audio_id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT,
                audio_data BLOB,
                transcript_id INTEGER,
                user_id INTERGER,
                FOREIGN KEY(transcript_id) REFERENCES Transcripts(transcript_id),
                FOREIGN KEY(user_id) REFERENCES Users(user_id)
            );
            CREATE TABLE IF NOT EXISTS Messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT,
                message TEXT,
                transcript_id INTEGER,
                user_id INTEGER,
                FOREIGN KEY(transcript_id) REFERENCES Transcripts(transcript_id),
                FOREIGN KEY(user_id) REFERENCES Users(user_id)
            );
        """)


def add_user_to_db(email, password):
    password_hash = hashlib.sha256(password.encode('utf-8')).hexdigest()
    with closing(sqlite3.connect('MASTER.db')) as conn:
        with closing(conn.cursor()) as cursor:
            insert_query = """
                INSERT INTO Users (email, password)
                VALUES (?, ?)
            """
            cursor.execute(insert_query, (email, password_hash))
            conn.commit()


def authenticate_user(email, password):
    password_hash = hashlib.sha256(password.encode('utf-8')).hexdigest()
    with closing(sqlite3.connect('MASTER.db')) as conn:
        with closing(conn.cursor()) as cursor:
            select_query = """
                SELECT email FROM Users WHERE email = ? AND password = ?
            """
            cursor.execute(select_query, (email, password_hash))
            user = cursor.fetchone()
    return bool(user)


def get_user_id(email):
    with closing(connect('MASTER.db')) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT user_id FROM Users WHERE email = ?", (email,))
        result = cursor.fetchone()
        return result[0] if result else None


def insert_into_transcripts(user_id, file_name, transcription, transcription_summary, sentiment_label, sentiment_report, prev_ai_research, fact_check):
    with sqlite3.connect('MASTER.db') as conn:
        cursor = conn.cursor()
        query = """
            INSERT INTO Transcripts (user_id, file_name, transcription, transcription_summary, sentiment_label, sentiment_report, prev_ai_research, fact_check) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor.execute(query, (user_id, file_name, transcription, transcription_summary, sentiment_label, sentiment_report, prev_ai_research, fact_check))
        conn.commit()


def get_transcript_ids_and_names():
    with sqlite3.connect('MASTER.db') as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT transcript_id, file_name FROM Transcripts WHERE user_id = ?", (st.session_state.user_id,))
        results = cursor.fetchall()
        return [f"{row[0]} - {row[1]}" for row in results]


def get_summary_by_id(selection):
    id = int(selection.split(' - ')[0])
    with sqlite3.connect('MASTER.db') as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT transcription_summary FROM Transcripts WHERE transcript_id = ? AND user_id = ?", 
                       (id, st.session_state.user_id))
        result = cursor.fetchone()
        if result is not None: return result[0]
        else: return "No transcript found for the given id"


def get_transcript_id(file_name):
    with sqlite3.connect('MASTER.db') as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT transcript_id FROM Transcripts WHERE file_name = ?", (file_name,))
        result = cursor.fetchall()[0][0]
        return result


def get_transcript_by_id(selection):
    id = int(selection.split(' - ')[0])
    with sqlite3.connect('MASTER.db') as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT transcription FROM Transcripts WHERE transcript_id = ?", (id,))
        result = cursor.fetchone()
        if result is not None: return result[0]
        else: return "No transcript found for the given id"


def get_sentiment_by_id(selection):
    id = int(selection.split(' - ')[0])
    with sqlite3.connect('MASTER.db') as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT sentiment_label FROM Transcripts WHERE transcript_id = ?", (id,))
        result = cursor.fetchone()
        if result is not None: return result[0]
        else: return "No transcript found for the given id"


def get_sentiment_report_by_id(selection):
    id = int(selection.split(' - ')[0])
    with sqlite3.connect('MASTER.db') as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT sentiment_report FROM Transcripts WHERE transcript_id = ?", (id,))
        result = cursor.fetchone()
        if result is not None: return result[0]
        else: return "No transcript found for the given id"


def get_fact_check_by_id(selection):
    id = int(selection.split(' - ')[0])
    with sqlite3.connect('MASTER.db') as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT fact_check FROM Transcripts WHERE transcript_id = ?", (id,))
        result = cursor.fetchone()
        if result is not None: return result[0]
        else: return "No transcript found for the given id"


def get_ai_qa_by_id(selection):
    id = int(selection.split(' - ')[0])
    with sqlite3.connect('MASTER.db') as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT prev_ai_research FROM Transcripts WHERE transcript_id = ?", (id,))
        result = cursor.fetchone()
        if result is not None: return result[0]
        else: return "No transcript found for the given id"


def insert_audio(file_path, transcript_id):
    with sqlite3.connect('MASTER.db') as conn:
        with open(file_path, 'rb') as f:
            audio_data = f.read()
        conn.execute("""
            INSERT INTO AudioFiles (file_name, audio_data, transcript_id, user_id) VALUES (?, ?, ?, ?)
        """, (os.path.basename(file_path), audio_data, transcript_id, st.session_state.user_id))
        conn.commit()


def insert_message():
    pass
