import sqlite3

DB_PATH = 'student_voice_track.db'

def get_all_student_embeddings():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT roll_no, embedding_path, time FROM students')
        return cursor.fetchall()

def get_all_teacher_embeddings():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT teacher_id, embedding_path FROM teachers')
        return cursor.fetchall()

def update_student_time(roll_no, delta_time):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('UPDATE students SET time = time + ? WHERE roll_no = ?', (delta_time, roll_no))
        conn.commit()

def add_student(roll_no, embedding_path):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('INSERT OR IGNORE INTO students (roll_no, embedding_path) VALUES (?, ?)', (roll_no, embedding_path))
        conn.commit()

def add_teacher(teacher_id, embedding_path):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('INSERT OR IGNORE INTO teachers (teacher_id, embedding_path) VALUES (?, ?)', (teacher_id, embedding_path))
        conn.commit()

def get_student_by_roll_no(roll_no):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT roll_no, embedding_path, time FROM students WHERE roll_no = ?', (roll_no,))
        return cursor.fetchone()

def get_teacher_by_teacher_id(teacher_id):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT teacher_id, embedding_path FROM teachers WHERE teacher_id = ?', (teacher_id,))
        return cursor.fetchone()

# Connect to SQLite database (creates file if it doesn't exist)
conn = sqlite3.connect('student_voice_track.db')
cursor = conn.cursor()

# Create students table (roll_no, embedding_path, time)
cursor.execute('''
CREATE TABLE IF NOT EXISTS students (
    roll_no TEXT PRIMARY KEY,
    embedding_path TEXT NOT NULL,
    time REAL DEFAULT 0
)
''')

# Create teachers table (teacher_id, embedding_path)
cursor.execute('''
CREATE TABLE IF NOT EXISTS teachers (
    teacher_id TEXT PRIMARY KEY,
    embedding_path TEXT NOT NULL
)
''')

conn.commit()
conn.close()
