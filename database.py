import sqlite3
from datetime import datetime

DB_PATH = 'instance/app.db'

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_data TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed_at TIMESTAMP,
                detections TEXT,
                audio_file TEXT
            )
        ''')
        conn.commit()
    print("Database initialized.")

def insert_image(image_base64):
    with get_db() as conn:
        cur = conn.execute(
            'INSERT INTO images (image_data, status) VALUES (?, ?)',
            (image_base64, 'pending')
        )
        conn.commit()
        return cur.lastrowid

def update_image_status(image_id, status, detections=None, audio_file=None):
    with get_db() as conn:
        conn.execute(
            '''UPDATE images
               SET status = ?, detections = ?, audio_file = ?, processed_at = CURRENT_TIMESTAMP
               WHERE id = ?''',
            (status, detections, audio_file, image_id)
        )
        conn.commit()

def get_image(image_id):
    with get_db() as conn:
        row = conn.execute('SELECT * FROM images WHERE id = ?', (image_id,)).fetchone()
        return row

def get_pending_images():
    with get_db() as conn:
        rows = conn.execute('SELECT * FROM images WHERE status = "pending" ORDER BY id ASC').fetchall()
        return rows