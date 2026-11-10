import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app import app
from database.database import get_db_connection

app.config['TESTING'] = True
client = app.test_client()

conn = get_db_connection()
cur = conn.cursor(dictionary=True)
cur.execute('SELECT id, username FROM users LIMIT 1')
user = cur.fetchone()
cur.close()
conn.close()

with client.session_transaction() as sess:
    sess['user_id'] = user['id']
    sess['username'] = user['username']

routes = [
    ('/', 'Home Page'),
    ('/dashboard', 'Dashboard (Consolidated DB)'),
    ('/history', 'History Page (In-Memory Aggregate)'),
    ('/profile', 'Profile Page (Fast Stats)'),
    ('/image_detection', 'Image Upload Page'),
    ('/video_detection', 'Video Upload Page'),
    ('/audio_detection', 'Audio Upload Page'),
    ('/about', 'About Page'),
    ('/contact', 'Contact Page'),
    ('/login', 'Login Page'),
    ('/register', 'Register Page'),
]

header = f"{'Route':<25} | {'Min (ms)':<10} | {'Avg (ms)':<10} | {'Max (ms)':<10}"
print(header)
print("-" * len(header))
for path, label in routes:
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        resp = client.get(path)
        times.append((time.perf_counter() - t0) * 1000)
    print(f"{path:<25} | {min(times):>8.2f} ms | {sum(times)/len(times):>8.2f} ms | {max(times):>8.2f} ms")
