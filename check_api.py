import requests
import base64
from PIL import Image
import io

# URL вашего локального сервера
url = "http://127.0.0.1:8000/api/detect"

# получаем изображение в формате base64_строка
image_path = "vid_4_720.jpg"
with open(image_path, "rb") as image_file:
    image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

# Заголовки запроса
headers = {'Content-Type': 'application/json'}

# Данные запроса
data = {"image_base64": image_base64}

# Отправка POST-запроса
response = requests.post(url, json=data, headers=headers)

# Обработка ответа
if response.status_code == 200:
    response_data = response.json()
    
    # Декодируем изображение с bbox из base64
    image_with_bbox_base64 = response_data["image_with_bbox_base64"]
    image_with_bbox_data = base64.b64decode(image_with_bbox_base64)
    image_with_bbox = Image.open(io.BytesIO(image_with_bbox_data))

    # Сохраняем изображение с bbox в корень проекта с именем out.jpg
    image_with_bbox.save("out.jpg")
    print("Успешный ответ. Изображение с детекцией, записано в корень проекта: 'out.jpg'")
    
    # Выводим только координаты bbox и уровень уверенности
    for detection in response_data["detections"]:
        print({"box": detection["box"], "confidence": detection["confidence"]})
else:
    print("Ошибка:", response.status_code, response.text)