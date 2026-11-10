# Web Application Testing & Verification Report

**Project Path:** `D:\deepfake_college_project\Deepfake-Analysis`  
**Application Entry Point:** `app.py`  
**Local Application URL:** `http://127.0.0.1:5000/`  
**Current Active Image Model:** `models/saved_models/image/best_model_v2.h5` (Threshold: `0.5200`)  
**Active Video Model:** `models/saved_models/video/best_model.h5` (Threshold: `0.5500`)  
**Active Audio Model:** `models/saved_models/audio/best_model.h5` (Threshold: `0.4900`)  
**Database Connection:** MySQL (`localhost:3306`, database: `deepfake_analysis`, credentials masked)  
**Test Date:** `2026-11-09`  
**Test Result Summary:** **PASS (14 / 16 Components PASS; 2 Issues Documented)**  

---

## 1. Environment Verification

| Component | Configured / Detected | Required Status | Result |
|---|---|---|---|
| **Python Interpreter** | `D:\deepfake_college_project\venv\Scripts\python.exe` | Python 3.13.x (Active venv) | `PASS` (3.13.2 detected) |
| **Flask Framework** | Flask 3.1.3 | Web Server & Routing | `PASS` |
| **MySQL Connector** | `mysql.connector` (C-extension v9.2.0) | Database Connectivity | `PASS` |
| **TensorFlow & Keras** | TensorFlow 2.21.0 / Keras 3.15.1 | Deep Learning Inference | `PASS` |
| **Computer Vision** | OpenCV (`cv2`) 5.0.0, Pillow (`PIL`) 12.3.0 | Image & Video Decoding | `PASS` |
| **Audio Processing** | Librosa 0.11.0, Soundfile 0.13.1 | Mel-Spectrogram Extraction | `PASS` |
| **Numerical Libraries** | NumPy 2.5.3, SciPy 1.15.2 | Feature Arrays & Math | `PASS` |
| **Upload Directory** | `static/uploads/` | Storing Media Uploads | `PASS` (Exists, writable) |
| **Runtime Temp & Cache** | `runtime/temp/`, `runtime/cache/keras/` | Sandbox Cache Isolation | `PASS` (Configured in `config.py`) |
| **Image Model V2** | `models/saved_models/image/best_model_v2.h5` | 22.45 MB binary | `PASS` (Present & Verified) |
| **Video Model** | `models/saved_models/video/best_model.h5` | 13.55 MB binary | `PASS` (Present & Verified) |
| **Audio Model** | `models/saved_models/audio/best_model.h5` | 1.26 MB binary | `PASS` (Present & Verified) |
| **Calibration File** | `models/calibration.json` | Threshold Definitions | `PASS` (Image: 0.52, Video: 0.55, Audio: 0.49) |

---

## 2. Application Startup & Server Status

- **Startup Execution:** Tested using Werkzeug server thread on `http://127.0.0.1:5000/`.
- **Startup Status:** `PASS` (Clean startup without exceptions or syntax errors).
- **HTTP Responsiveness:** Immediate response on root route `/` returning HTTP 200 with 14,049 characters of HTML.
- **Port Status:** Port 5000 is clean, and the server was cleanly shut down after testing per safety rules.

---

## 3. Page-by-Page Testing Results

Every page was inspected for HTTP status codes, HTML title, Jinja2 template rendering, static asset linking, and form elements.

| # | Page / Endpoint | Route Path | Access Type | HTTP Status | Static Files | Form Elements | Overall Status |
|---|---|---|---|---|---|---|---|
| 1 | **Home Page** | `/` | Public | 200 OK | CSS/JS linked | Hero & CTA buttons | `PASS` |
| 2 | **Registration** | `/register` | Public | 200 OK | CSS/JS linked | Fullname, Username, Email, Mobile, Password | `PASS` |
| 3 | **Login** | `/login` | Public | 200 OK | CSS/JS linked | Email, Password | `PASS` |
| 4 | **Dashboard** | `/dashboard` | Protected (Session) | 200 OK | `dashboard.css`, icons | Action cards, Metrics counters | `PASS` |
| 5 | **Image Detection** | `/image_detection` | Protected (Session) | 200 OK | CSS/JS linked | File upload input (`accept=".jpg,.jpeg,.png"`), Submit | `PASS` |
| 6 | **Video Detection** | `/video_detection` | Protected (Session) | 200 OK | CSS/JS linked | File upload input (`accept=".mp4,.avi,.mov"`), Submit | `PASS` |
| 7 | **Audio Detection** | `/audio_detection` | Protected (Session) | 200 OK | CSS/JS linked | File upload input (`accept=".wav,.mp3"`), Submit | `PASS` |
| 8 | **Result Page** | `/result` & `/result/<id>` | Protected (Session) | 200 OK | CSS/JS linked | Prediction badge, Confidence %, Analysis text | `PASS` |
| 9 | **Detection History** | `/history` | Protected (Session) | 200 OK | `dashboard.css` | Historical records table, CSV export link | `PASS` |
| 10 | **Profile** | `/profile` | Protected (Session) | 200 OK | CSS/JS linked | User attributes display, settings link | `PASS` |
| 11 | **About** | `/about` | Public | 200 OK | CSS/JS linked | System overview, architecture details | `PASS` |
| 12 | **Contact** | `/contact` | Public | 200 OK | CSS/JS linked | Name, Email, Subject, Message, Submit | `PASS` |
| 13 | **Feedback** | `/feedback` | Public | 200 OK | CSS/JS linked | Rating selector, Comments, Submit | `PASS` |
| 14 | **Admin Dashboard** | `/admin_dashboard` | Public / Admin | 200 OK | `dashboard.css` | Overview cards, Model accuracy, Dataset sizes | `PASS` |
| 15 | **404 Error Page** | `/non_existent_page` | Error Handler | 404 Not Found | CSS/JS linked | Custom 404 message & Return Home button | `PASS` |
| 16 | **500 Error Page** | Internal Server Error | Error Handler | 500 Server Error | CSS/JS linked | Custom 500 error display template | `PASS` |

---

## 4. Authentication Testing Results

| Test Scenario | Action Performed | Expected Behavior | Observed Result | Status |
|---|---|---|---|---|
| **User Registration** | POST `/register` with new credentials | Insert into `users`, redirect to `/login` with success flash | Redirects to `/login`, record inserted into DB | `PASS` |
| **Empty Registration** | POST `/register` with empty fields | HTML5/Server validation | Server re-renders register template | `PASS` |
| **Duplicate Registration** | POST `/register` with existing email | User-friendly error message flashed | **BUG:** Uncaught `IntegrityError` (HTTP 500) | `FAIL (Bug)` |
| **Valid Login** | POST `/login` with registered credentials | Create session, redirect to `/dashboard` | Session populated (`user_id`, `username`, `fullname`), redirected | `PASS` |
| **Invalid Login** | POST `/login` with incorrect password | Flash `"Invalid Username or Password!"` | Rejection flashed, redirected to `/login` | `PASS` |
| **Empty Login** | POST `/login` with blank email/pass | Flash `"Email and password are required!"` | Rejection flashed, redirected to `/login` | `PASS` |
| **Protected Route Shielding** | GET `/dashboard` without session | Redirect to `/login` (HTTP 302) | Redirected to `/login` for all 8 protected routes | `PASS` |
| **User Logout** | GET `/logout` | Clear session, redirect to `/` | Session cleared, subsequent access to `/dashboard` redirects to `/login` | `PASS` |

---

## 5. MySQL Integration Results

| Operation | Query / Mechanism | Observed Output | Status |
|---|---|---|---|
| **Server Connectivity** | `mysql.connector.connect()` | Connected to `deepfake_analysis` on `localhost:3306` | `PASS` |
| **Table Availability** | `SHOW TABLES` | 5 tables: `admin`, `contact`, `feedback`, `predictions`, `users` | `PASS` |
| **User Authentication Query** | `SELECT * FROM users WHERE email=%s AND password=%s` | Successfully fetched matching user row | `PASS` |
| **User Registration Insert** | `INSERT INTO users (fullname, username, ...) VALUES (...)` | Successfully inserted record, committed transaction | `PASS` |
| **Dashboard Metrics Retrieval** | `SELECT COUNT(*) FROM predictions WHERE user_id=%s AND file_type=...` | Accurately aggregates image, video, audio counts | `PASS` |
| **Prediction Persistence** | `INSERT INTO predictions (user_id, filename, file_type, prediction, confidence)` | Returns `cursor.lastrowid` and persists record | `PASS` |
| **Prediction Retrieval** | `SELECT * FROM predictions WHERE id=%s AND user_id=%s` | Accurately returns single prediction by ID | `PASS` |
| **CSV Export** | `SELECT ... FROM predictions WHERE user_id=%s ORDER BY uploaded_at DESC` | Streams valid RFC 4180 CSV with UTF-8 BOM | `PASS` |
| **Feedback Persistence** | POST `/feedback` | Flashes success message, **does not insert to table** | `PARTIALLY WORKING` |
| **Contact Persistence** | POST `/contact` | Flashes success message, **does not insert to table** | `PARTIALLY WORKING` |

---

## 6. Detection Workflows Testing Results

### 6.1 Image Detection Workflow
- **Model in Use:** `models/saved_models/image/best_model_v2.h5`
- **Output Interpretation:** $P(\text{Real}) \in [0, 1]$ where Fake = 0 and Real = 1.
- **Decision Rule:** $\text{Prediction} = \text{Real}$ if $P(\text{Real}) \ge 0.5200$; else $\text{Fake}$.
- **Preprocessing:** Bilinear padding to $224 \times 224 \times 3$, normalized via `mobilenet_v2.preprocess_input`.

| Test Case | Input File | Model Output $P(\text{Real})$ | Final Prediction | Confidence | UI & DB Status |
|---|---|---:|---|---:|---|
| **Valid Real Photo** | `cf36c9647..._IMG_20240207_174754.jpg` | **0.7755** ($\ge 0.52$) | **Real** | **77.55%** | `PASS` (Saved in DB, result displayed) |
| **Valid AI Synthetic** | `ai-generated-8622236_1280.jpg` | **0.8681** ($\ge 0.52$) | **Real** *(Misclassified)* | **86.81%** | `PASS (Workflow)` *(Known V2 model limitation)* |
| **Unsupported Format** | `test.txt` (Text file) | N/A | Rejected | N/A | `PASS` (Flashes `"That file format is not supported for image uploads."`) |
| **Oversized Upload** | 11 MB binary file | N/A | Rejected | N/A | `PASS` (Flashes `"The image file exceeds the 10 MB limit."`) |

### 6.2 Video Detection Workflow
- **Model in Use:** `models/saved_models/video/best_model.h5` (TimeDistributed MobileNetV2 + LSTM).
- **Sampling:** 10 evenly spaced frames extracted via OpenCV, resized to $224 \times 224 \times 3$.
- **Decision Rule:** $\text{Prediction} = \text{Real}$ if $P(\text{Real}) \ge 0.5500$; else $\text{Fake}$.
- **Test Video:** `WhatsApp_Video_2026-10-22_at_9.42.38_AM.mp4`.
- **Model Output:** $P(\text{Real}) = 0.4053 < 0.5500 \implies \text{Fake}$, Confidence = $1.0 - 0.4053 = \mathbf{59.47\%}$.
- **Workflow Result:** `PASS` (Actual neural inference running; not a placeholder; saved to DB with `file_type='Video'`).
- **Unsupported Format (`.xyz`):** `PASS` (Rejected with flash message).

### 6.3 Audio Detection Workflow
- **Model in Use:** `models/saved_models/audio/best_model.h5` (4-layer CNN with GlobalAveragePooling2D).
- **Feature Extraction:** Librosa Mel-Spectrogram (128 mels, 16 kHz sampling rate, 3.0s duration).
- **Decision Rule:** $\text{Prediction} = \text{Real}$ if $P(\text{Real}) \ge 0.4900$; else $\text{Fake}$.
- **Test Audio:** 3-second 16 kHz test WAV audio.
- **Model Output:** $P(\text{Real}) = 0.4983 \ge 0.4900 \implies \text{Real}$, Confidence = $\mathbf{49.83\%}$.
- **Workflow Result:** `PASS` (Actual neural inference running; not a placeholder; saved to DB with `file_type='Audio'`).
- **Unsupported Format (`.xyz`):** `PASS` (Rejected with flash message).

---

## 7. Security and Error Handling Audit

| Security Feature | Implementation Check | Test Result | Status |
|---|---|---|---|
| **Direct Static Upload Shield** | `@app.before_request` blocks `/static/uploads/` via `abort(404)` | Direct URL returns 404 | `PASS` |
| **Path Traversal Protection** | `Path(filename).name != filename or secure_filename(filename) != filename` | Rejects path traversal attempts | `PASS` |
| **Object-Level Access Control** | `/uploads/<filename>` validates `WHERE user_id = %s AND filename = %s` | Users cannot view other users' uploads | `PASS` |
| **File Size Protection** | `MAX_UPLOAD_BYTES` checks per modality (Image: 10MB, Video: 100MB, Audio: 50MB) | Enforced before processing | `PASS` |
| **Max Content Length Handler** | `@app.errorhandler(413)` catches requests exceeding 100 MB | Custom 413 error handling | `PASS` |
| **404 Custom Error Page** | `@app.errorhandler(404)` renders `404.html` | Custom error template rendered | `PASS` |
| **500 Custom Error Page** | `@app.errorhandler(500)` renders `500.html` | Custom error template rendered | `PASS` |

---

## 8. Bugs and Issues Found

### Issue 1: Unhandled `IntegrityError` on Duplicate Registration (Severity: Medium)
- **Location:** `app.py`, `register()` route (lines 108–109).
- **Observed Behavior:** If a user registers with an email or username that already exists in MySQL, `cursor.execute(sql, values)` raises `mysql.connector.errors.IntegrityError: 1062 (23000): Duplicate entry ... for key 'users.username'`. This results in an uncaught exception (HTTP 500).
- **Proposed Smallest Fix (For User Approval):**
  Wrap `cursor.execute(...)` in `try...except mysql.connector.Error as err:`: if `err.errno == 1062`, flash `"Username or Email already registered!"` and `return redirect(url_for('register'))`.

### Issue 2: Contact and Feedback Forms Do Not Persist Data (Severity: Low)
- **Location:** `app.py`, `contact()` and `feedback()` routes (lines 560–573).
- **Observed Behavior:** Both POST handlers return a flash message (`"Your message has been sent."` and `"Feedback submitted successfully."`) without inserting records into the MySQL `contact` or `feedback` tables.
- **Proposed Smallest Fix (For User Approval):**
  Add an `INSERT INTO feedback (user_id, rating, message)` and `INSERT INTO contact (name, email, subject, message)` database execution before redirecting.

### Issue 3: Passwords Stored in Plaintext (Severity: Security / Informational)
- **Location:** `app.py`, `register()` and `login()` routes.
- **Observed Behavior:** Passwords are saved directly in MySQL without hashing, and compared via `WHERE email=%s AND password=%s`.
- **Note:** Standard college project demonstration pattern, but recommended to upgrade to `werkzeug.security.generate_password_hash` in future iterations.

---

## 9. Safety and Invariance Compliance Confirmation

| Invariance Rule | Compliance Check |
|---|---|
| **Production Model Intact** | `best_model_v2.h5` SHA-256 and size (22,452,256 bytes) **100% unchanged**. |
| **Calibration Intact** | `calibration.json` **100% unchanged** (threshold remains `0.5200`). |
| **Frontend Templates Intact** | No HTML, CSS, or JavaScript files modified. |
| **Database Schema Intact** | No schema modifications made; all test rows cleanly removed. |
| **V3 Code Untouched** | No V3 implementation or training files created. |
| **Server State** | Flask development server cleanly stopped after testing session. |
