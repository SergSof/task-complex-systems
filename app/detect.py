import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from dotenv import load_dotenv

# Загрузка переменных окружения из файла .env
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path, override=True)

class CarDetectionModel:
    def __init__(self, model_path):
        # Определяем устройство (GPU или CPU) из переменной окружения
        device = os.getenv('DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_path).to(device)
        
    def detect_cars(self, image_path: str, target_classes=[2]):
        """
        Выполняет детекцию объектов на изображении.
        Из документации: "Устройство (по умолчанию: device=None): Выбор CPU или GPU."
        Автоматически использует GPU, если он доступен, и переключается на CPU, 
        если GPU недоступен.

        Args:
            image_input (разные типы): Путь к изображению, 
            объект OpenCV, NumPy массив или Torch тензор.
            target_classes (list): Список целевых классов 
            для фильтрации. По умолчанию [2] (автомобили).

        Returns:
            output_path (str): Путь к изображению с детекциями.
            car_detections (list): Список детекций автомобилей.
        """
        
        # Детекция
        results = self.model.predict(image_path, conf=0.5, iou=0.3)[0] #, classes=target_classes

        # Извлекаем bounding boxes и уверенности
        boxes = results.boxes.xyxy.cpu().numpy().astype(int)  # Координаты [x1, y1, x2, y2]
        scores = (results.boxes.conf * 100).cpu().numpy().astype(int)  # Уверенность

        # Сохраняем детекции автомобилей
        car_detections = [(box, score) for box, score in zip(boxes, scores)]
       
        # Визуализация результатов с использованием встроенных функций ultralytics
        annotated_image = results.plot()  # Получаем изображение с нарисованными bounding boxes

        # Сохранение изображения с детекциями
        output_path = os.path.join('app', 
                                   'static', 
                                   'detected_cars.jpg')  # Путь для сохранения результата в папке static
        cv2.imwrite(output_path, annotated_image)  # Сохранение изображения в формате RGB

        return output_path, car_detections  # Возвращаем путь к изображению и детекции
