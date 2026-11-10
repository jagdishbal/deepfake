# Backend Audit Report — Deepfake Analysis Web Server

**Audit Date:** November 10, 2026  
**Auditor:** Senior Backend Systems & Python Engineer  
**Core Files Audited:** `app.py`, `config.py`, `predict.py`, `services/prediction_service.py`, `services/audit_service.py`  

---

## 1. Flask Application Structure & Initialization

The backend application is orchestrated in `app.py`:
- **App Instance:** `Flask(__name__)` configured with `MAX_CONTENT_LENGTH = 100 * 1024 * 1024` (100 MB).
- **Session Security:** `app.secret_key = "deepfake-analysis-secret"` providing signed client session cookies.
- **Cache Control:** `@app.after_request` injects `Cache-Control: public, max-age=43200` on static assets to optimize browser performance.
- **Directory Protection:** `@app.before_request` intercepts direct requests to `/static/uploads/` returning HTTP 404, ensuring uploaded files are only accessible to authenticated owners via `/uploads/<filename>`.

---

## 2. Request Handling & Upload Lifecycle

```
[Client POST /<modality>_detection]
                 │
                 ▼
       [Session Auth Check]  ──(No session)──▶  [Redirect /login]
                 │
            (Authenticated)
                 ▼
       [_save_upload()]
         ├── secure_filename() sanitization
         ├── Extension whitelist check
         ├── MIME-type validation
         ├── Size quota check (10MB Image, 100MB Video, 50MB Audio)
         └── Unique file path generation: <uuid>_<filename>
                 │
                 ▼
       [predict_media()] (services/prediction_service.py)
         └── predict_file() (predict.py)
                 │
                 ▼
       [_record_prediction()]
         └── INSERT INTO predictions (user_id, filename, file_type, prediction, confidence)
                 │
                 ▼
       [log_prediction_audit()]
         └── Writes detailed dual probabilities, threshold, routing reason
                 │
                 ▼
       [Redirect /result/<prediction_id>]
                 │
                 ▼
       [view_result()] renders result.html with dual probabilities
```

---

## 3. Modality Prediction Orchestration in `predict.py`

### Thread-Safe Model Management
- Models are cached in memory using `@lru_cache(maxsize=4)`.
- Concurrent thread access is synchronized with `_MODEL_LOAD_LOCK = threading.Lock()`.
- Warmup model preloading runs in a background daemon thread at server launch so that users experience instant inference on their initial upload.

### Unified Modality Dispatcher
- `predict_file(file_path, modality=None)`:
  - If `modality == "image"`: dispatches to `predict_image(file_path)` (executes Resolution-Gated Dual Pipeline between Model V3 and Model V2).
  - If `modality == "video"`: dispatches to `predict_video(file_path)` (executes 3-window 5-frame sampling and weighted min-mean aggregation).
  - If `modality == "audio"`: dispatches to `predict_audio(file_path)` (extracts 140 acoustic features and executes MLP inference).

---

## 4. Error Handling & Exception Management

The application features three tiers of defensive error handling:
1. **Application-Level Error Handlers:**
   - `@app.errorhandler(404)`: Renders custom `404.html` with navigation back to Home.
   - `@app.errorhandler(413)`: Gracefully flashes `"The upload exceeds the 100 MB server limit."` and redirects to the upload page.
   - `@app.errorhandler(500)`: Catches unexpected crashes and renders user-friendly `500.html`.
2. **Domain-Specific Prediction Exceptions:**
   - `UnsupportedMediaError`: Raised when unsupported formats are submitted; flashes user-friendly alert.
   - `MediaDecodeError`: Raised on corrupted video bitstreams, silent MP4s, or truncated audio; flashes specific failure reason and cleans up temporary files.
   - `ModelUnavailableError`: Raised if model files are missing; logs error to administrator and falls back safely.
3. **Database Transaction Rollbacks:**
   - Every database modification wraps operations in `try ... except ... finally` with explicit `connection.rollback()` on exception and guaranteed cursor/connection closing.
