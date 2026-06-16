# Crowd Count System

AI-powered crowd detection and alert system using YOLOv8n, FastAPI, and Docker — deployed on AWS EC2 with automated CI/CD and MLflow experiment tracking.



![Crowd Count System Demo](screenshot.png)



## Features
- Upload an image and detect the number of people using YOLOv8n
- Set a custom crowd threshold
- Get instant visual alerts (Normal / ALERT) based on threshold
- Tracks all predictions via MLflow

## Tech Stack
- **Model:** YOLOv8n (Ultralytics)
- **Backend:** FastAPI, Uvicorn
- **Frontend:** HTML, CSS, JavaScript
- **Containerization:** Docker
- **CI/CD:** GitHub Actions
- **Cloud:** AWS EC2
- **Experiment Tracking:** MLflow

## How It Works
1. User uploads an image and sets a threshold
2. FastAPI backend runs YOLOv8n to detect and count people
3. System compares count to threshold and returns an alert status
4. Frontend displays result with color-coded alert

## Setup (Local)
\`\`\`bash
git clone https://github.com/aiswaryabiju30/Crowd-Count-System.git
cd Crowd-Count-System
pip install -r requirements.txt
uvicorn main:app --reload
\`\`\`

## Limitations
- Accuracy decreases on dense, overlapping crowds (YOLOv8n limitation)
- Currently supports single-image analysis, not live video streams
