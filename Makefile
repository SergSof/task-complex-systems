# Makefile

# Переменные
VENV_DIR = venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip

# Цели
all: install run

# Создание виртуального окружения
$(VENV_DIR):
    python -m venv $(VENV_DIR)

# Установка зависимостей
install: $(VENV_DIR)
    $(PIP) install -r requirements.txt

# Запуск приложения
run:
    $(PYTHON) app/main.py

# Развертывание через Docker
docker-build:
    docker-compose up --build

# Очистка
clean:
    rm -rf $(VENV_DIR)
    rm -f out.jpg

.PHONY: all install run docker-build clean