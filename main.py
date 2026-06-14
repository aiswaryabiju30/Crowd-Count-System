from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import shutil
import mlflow
mlflow.set_experiment("crowd_count_system")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("yolov8n.pt")

mlflow.set_experiment("crowd_count_system")


@app.post("/predict")
async def predict(file: UploadFile = File(...), threshold: int = Form(5)):
    temp_path = "uploaded_image.jpg"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model(temp_path)

    person_count = 0
    for box in results[0].boxes:
        if int(box.cls[0]) == 0:
            person_count += 1

    alert = person_count > threshold

    with mlflow.start_run():
        print("MLFLOW RUN STARTED")
        print("Threshold:", threshold)
        print("Person Count:", person_count)
        print("Alert:", alert)

        mlflow.log_param("threshold", threshold)
        mlflow.log_metric("person_count", person_count)
        mlflow.log_metric("alert", int(alert))

        print("MLFLOW LOGGING COMPLETED")

    return JSONResponse({
        "person_count": person_count,
        "threshold": threshold,
        "alert": alert
    })


@app.get("/")
def home():
    return {"message": "Crowd Count System API is running"}