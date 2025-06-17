import pandas as pd
import hashlib
import re
from cryptography.fernet import Fernet

# 1. Проверка CSV на уязвимости (CSV Injection)
# Опасные символы могут использоваться для атак в Excel и других программах
# Функция проверяет, начинаются ли значения в текстовых столбцах с этих символов
def check_csv_injection(df):
    dangerous_chars = ('=', '+', '-', '@')
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].astype(str).apply(lambda x: x.lstrip(' ').startswith(dangerous_chars)).any():
            print(f"Обнаружены потенциальные CSV-инъекции в столбце {col}!")
        else:
            print(f"Столбец {col} безопасен.")
    return df

# 2. Фильтрация данных от SQL-инъекций и XSS-атак
# SQL-инъекции используются для взлома баз данных, а XSS – для атак на пользователей
# Функция заменяет опасные конструкции на строку '[BLOCKED]'
def clean_input(value):
    sql_keywords = ["SELECT", "DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "UNION", "--"]  # SQL-команды
    xss_patterns = [r'<script.*>.*?</script>', r'javascript:.*', r'onerror=.*']  # XSS-скрипты

    for keyword in sql_keywords:
        if keyword.lower() in value.lower():
            return "[BLOCKED]"
    return value

# 3. Хеширование столбца с ценами (SHA-256)
# Хеширование используется для защиты конфиденциальных данных, таких как пароли
# В данном случае мы хешируем цену ноутбука, чтобы скрыть оригинальные значения
def hash_price(price):
    return hashlib.sha256(str(price).encode()).hexdigest()

# 4. Шифрование данных (например, цены ноутбуков)
# Шифрование позволяет скрыть данные, но при этом их можно расшифровать при наличии ключа
# Генерируем ключ и шифруем цену ноутбука
cipher = Fernet(Fernet.generate_key())

def encrypt_price(price):
    return cipher.encrypt(str(price).encode()).decode()

def decrypt_price(encrypted_price):
    return cipher.decrypt(encrypted_price.encode()).decode()

def decrypt_ram(encrypted_ram):
    return cipher.decrypt(encrypted_ram.encode()).decode()
