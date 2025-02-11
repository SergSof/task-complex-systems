# Тестовое задание

Задача состоит в построении модели, определяющей автомобили на изображениях. 
Результат сохраняется в виде bbox. 
Датасет (передается по запросу в виде архива) состоит из папки с изображениями, которые нужно обрабатывать и файла с разметкой для изображений.
Задача – создать простой веб-сервис (можно на fastapi), который содержит:
1. HTML-страницу с формой загрузки и отображением результатов обработки.
2. API, принимающий изображение в виде base64-строки и возвращающий результат его обработки: 
   a. Маску наложенную на изображение (изображения).
   b. Массив с bbox.
3. Приложение должно работать в контейнере docker.
4. НС должна уметь работать на GPU и CPU.
5. С помощью переменных окружения настраивается устройство, на котором работает нейросеть, IP и порт приложения.

Сервис должен быть с инструкцией для запуска (README) и Makefile. Выбор метода решения свободный полностью. Однако, желательно для избранного метода проанализировать точность на всей выборке или ее части, дать рекомендации по повышению точности. 
Предоставить ipynb с обучением модели.

Системные требования:
1. Использование python3.10+.
2. Система будет тестироваться на Linux Ubuntu.
3. Задание отправляется ссылкой на GitHub или в архиве zip.

# Развернуть проект на своем устройстве

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/SergSof/task-complex-systems
   cd task-complex-systems
   ```

2. Создайте и активируйте виртуальное окружение:
```bash
python -m venv venv
```
### На macOS/Linux:
```bash
source venv/bin/activate
```  
### Для Windows: 
```bash
.\venv\Scripts\activate
```


3. Установите зависимости:
```bash
pip install -r requirements.txt
```

5. Запустите приложение:
```bash
python app/main.py
```

# Использование переменных окружения
Для настройки переменных окружения используйте файл .env. Пример содержимого файла .env:
```python
DEVICE=cpu
HOST=127.0.0.1
PORT=8000
```
Описание переменных окружения
- DEVICE: Устройство, на котором будет работать нейросеть (cpu или gpu).
- HOST: IP-адрес, на котором будет запущено приложение.
- PORT: Порт, на котором будет запущено приложение.

# Использование API

API принимает изображение в формате base64 и возвращает координаты и изображение с наложенными bounding boxes.

URL
http://127.0.0.1:8000/api/detect

Метод:
POST

Тело запроса: 
строка, представляющая изображение в формате base64.

Ответ API:
API возвращает JSON-объект, содержащий следующие поля:

- `image_with_bbox_base64`: строка, представляющая изображение с наложенными bounding boxes в формате base64.
- `detections`: массив объектов, каждый из которых содержит:
- `box`: массив координат bounding box в формате `[x1, y1, x2, y2]`.
- `confidence`: уровень уверенности для данного bounding box.

Пример ответа:
```python
{
  "image_with_bbox_base64": "",
  "detections": [
    {
      "box": [100, 150, 200, 250],
      "confidence": 0.95
    },
    {
      "box": [300, 350, 400, 450],
      "confidence": 0.90
    }
  ]
}
```
Для проверки API можно использовать файл check_api.py. На вход подаем картинку vid_4_720.jpg (в корне проекта), получаем результат out.jpg (тоже в корне проекта).



# Ссылка на блокнот Google Colab с обучением
<a href="https://colab.research.google.com/drive/13CAaRRoUbgyOGs_QMqul9o4DwFN_rYWk?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

YOLO v11

Обученная модель: app\models\model_l_yolo_v11_custom.pt

- Precision: 0.996
- Recall: 1.0
- mAP50: 0.995
