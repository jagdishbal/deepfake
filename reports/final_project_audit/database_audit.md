# MySQL Database Audit Report — Deepfake Analysis System

**Audit Date:** November 10, 2026  
**Auditor:** Database Administrator & Backend Specialist  
**Database Name:** `deepfake_analysis`  
**Database Engine:** MySQL 8.0.x (InnoDB, `utf8mb4_0900_ai_ci`)  
**Host & Port:** `localhost:3306`  

---

## 1. Database Connection Management

The database layer in `database/database.py` manages connections via MySQL Connector Pooling:
- **Pool Configuration:**
  - Name: `deepfake_db_pool`
  - Size: 8 connections
  - `pool_reset_session=True`
- **Fallback Resilience:**
  - If the connection pool is exhausted or uninitialized, `get_db_connection()` automatically falls back to an immediate direct connection to ensure zero dropped user requests.

---

## 2. Table Schemas & Relational Integrity

The database architecture consists of 5 relational tables:

### 1. `users` Table
```sql
CREATE TABLE `users` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `fullname` VARCHAR(100) NOT NULL,
  `username` VARCHAR(50) NOT NULL,
  `email` VARCHAR(100) NOT NULL,
  `mobile` VARCHAR(15) DEFAULT NULL,
  `password` VARCHAR(255) NOT NULL,
  `role` ENUM('user', 'admin') DEFAULT 'user',
  `created_at` TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `username` (`username`),
  UNIQUE KEY `email` (`email`)
) ENGINE=InnoDB;
```
- **Audited Row Count:** 13 registered users.
- **Constraints:** Enforces unique email and username; supports role-based administration.

### 2. `admin` Table
```sql
CREATE TABLE `admin` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `username` VARCHAR(50) NOT NULL,
  `password` VARCHAR(255) NOT NULL,
  `created_at` TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `username` (`username`)
) ENGINE=InnoDB;
```
- **Audited Row Count:** 1 system administrator (`username: admin`).

### 3. `predictions` Table
```sql
CREATE TABLE `predictions` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `user_id` INT NOT NULL,
  `filename` VARCHAR(255) NOT NULL,
  `file_type` ENUM('Image', 'Video', 'Audio') NOT NULL,
  `prediction` ENUM('Real', 'Deepfake') NOT NULL,
  `confidence` DECIMAL(5,2) NOT NULL,
  `uploaded_at` TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `user_id` (`user_id`),
  CONSTRAINT `predictions_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB;
```
- **Audited Row Count:** 156+ verified prediction records across Image, Video, and Audio.
- **Relational Integrity:** Foreign key `user_id` references `users(id)` with `ON DELETE CASCADE`.

### 4. `contact` Table
```sql
CREATE TABLE `contact` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `name` VARCHAR(100) NOT NULL,
  `email` VARCHAR(100) NOT NULL,
  `subject` VARCHAR(200) DEFAULT NULL,
  `message` TEXT NOT NULL,
  `created_at` TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB;
```
- Stores user inquiries submitted through `/contact`.

### 5. `feedback` Table
```sql
CREATE TABLE `feedback` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `user_id` INT DEFAULT NULL,
  `rating` INT DEFAULT NULL,
  `message` TEXT,
  `created_at` TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  KEY `user_id` (`user_id`),
  CONSTRAINT `feedback_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE,
  CONSTRAINT `feedback_chk_1` CHECK ((`rating` BETWEEN 1 AND 5))
) ENGINE=InnoDB;
```
- Enforces check constraint ensuring ratings remain between 1 and 5 stars.

---

## 3. Database Persistence Verification

During the live multi-modality integration test, predictions were inserted and queried across all three media formats:
- **Record ID 173:** `[Image]` -> Prediction: `Real`, Confidence: `85.21%`
- **Record ID 174:** `[Video]` -> Prediction: `Deepfake`, Confidence: `99.35%`
- **Record ID 175:** `[Audio]` -> Prediction: `Deepfake`, Confidence: `99.96%`

All records were successfully retrieved by `GET /history` and `GET /dashboard`, confirming full relational integrity.
