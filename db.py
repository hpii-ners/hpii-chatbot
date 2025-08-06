# db.py
import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()

# open connection with autocommit=True
conn = psycopg2.connect(
    host=os.getenv("POSTGRES_HOST"),
    port=os.getenv("POSTGRES_PORT"),
    database=os.getenv("POSTGRES_DB"),
    user=os.getenv("POSTGRES_USER"),
    password=os.getenv("POSTGRES_PASSWORD")
)
conn.autocommit = True   # <- eliminates transaction poisoning

def save_chat_history(user_id, question, answer):
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat_history (user_id, question, answer) VALUES (%s, %s, %s)",
                (user_id, question, answer)
            )
    except Exception as e:
        print("[DB ERROR]", e)        # log the real error
        # no need to rollback when autocommit=True