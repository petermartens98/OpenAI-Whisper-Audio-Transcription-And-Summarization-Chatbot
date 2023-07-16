import sqlite3

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
                user_id INTEGER,
                FOREIGN KEY(user_id) REFERENCES Users(user_id)
            );
            CREATE TABLE IF NOT EXISTS AudioFiles (
                audio_id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT,
                audio_data BLOB,
                transcript_id INTEGER,
                FOREIGN KEY(transcript_id) REFERENCES Transcripts(transcript_id)
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
