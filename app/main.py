import os
import sys
import base64
import cv2
import numpy as np
from dotenv import load_dotenv
# Добавляем путь к модулю app в sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, Request, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.detect import CarDetectionModel


# Загрузка переменных окружения из файла .env
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path, override=True)

# Вывод значений переменных окружения для проверки
print(f"DEVICE: {os.getenv('DEVICE')}")
print(f"HOST: {os.getenv('HOST')}")
print(f"PORT: {os.getenv('PORT')}")


app = FastAPI()
templates = Jinja2Templates(directory=os.path.join("app", "templates"))

# Инициализируем модель
model_path = os.path.join("app", "models", "model_l_yolo_v11_custom.pt")
car_detection_model = CarDetectionModel(model_path=model_path)

app.mount("/static", StaticFiles(directory=os.path.join("app", "static")), name="static")

@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, file: UploadFile = File(...)):
    try:
        # Указываем путь для сохранения загружаемого файла
        upload_folder = os.path.join("app", "static", "uploads")
        os.makedirs(upload_folder, exist_ok=True)  # Создаём папку, если её нет
        file_path = os.path.join(upload_folder, 'img_original.jpg')

        # Сохраняем файл на диск
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Вызываем метод детекции, передавая путь к файлу
        result_path, detections = car_detection_model.detect_cars(file_path)
        
        # Преобразуем детекции в удобный формат
        detections = [(box, float(confidence)) for box, confidence in detections]

        # Формируем ответ
        return templates.TemplateResponse("result.html", {
            "request": request,
            "detected_image_path": f"/static/{os.path.basename(result_path)}",
            "detections": detections
        })

    except Exception as e:
        return JSONResponse(content={"error": f"Произошла ошибка: {str(e)}"}, status_code=500)

class ImageRequest(BaseModel):
    image_base64: str

@app.post("/api/detect")
async def api_detect(image_request: ImageRequest):
    """
    API для обработки изображения в формате base64.
    Возвращает изображение с наложенными bbox (в формате base64) и массив координат bbox.
    """
    try:
        # Декодируем изображение из base64
        image_data = base64.b64decode(image_request.image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            print("Не удалось прочитать изображение!!!!")
        # Указываем путь для сохранения изображения
        image_path = os.path.join("app", "static", "uploads", "image_api.jpg")
       
        # Сохраняем изображение на диск 
        cv2.imwrite(image_path, image)

        # Передаём путь к изображению в метод детекции
        result_path, detections = car_detection_model.detect_cars(image_path)

        # Читаем изображение с bbox и конвертируем его в base64
        with open(result_path, "rb") as result_image_file:
            result_image_base64 = base64.b64encode(result_image_file.read()).decode('utf-8')

        # Формируем ответ
        response = {
            "image_with_bbox_base64": result_image_base64,
            "detections": [
                {"box": [int(coord) for coord in box], "confidence": float(confidence)}
                for box, confidence in detections
            ]
        }

        return JSONResponse(content=response)
    except Exception as e:
        return JSONResponse(content={"error": f"Произошла ошибка: {str(e)}"}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    # Чтение IP и порта из переменных окружения
    host = os.getenv('HOST', '127.0.0.1')
    port = int(os.getenv('PORT', 8000))
    uvicorn.run(app, host=host, port=port)        