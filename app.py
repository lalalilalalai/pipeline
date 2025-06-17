from fastapi import FastAPI, File, UploadFile
import pandas as pd
import joblib
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from contextlib import asynccontextmanager
import os
from sanitize import check_csv_injection, cipher, clean_input, hash_price, encrypt_price, decrypt_price, decrypt_ram

model = None
model_path = './laptop_price_model.pkl'

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if os.path.exists(model_path):
        print("Loading existing model...")
        model = joblib.load(model_path)
    else:
        print("Training new model...")
        file_path = './Laptop_price.csv'
        df = pd.read_csv(file_path)
        X = df.drop(columns=['Price'])
        y = df['Price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        cat_features = X.select_dtypes(include=['object']).columns.tolist()

        num_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        cat_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])
        preprocessor = ColumnTransformer([
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ])
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5))
        ])
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, model_path)
        model = pipeline
        print("Model trained and saved.")
    
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global model 
    content = await file.read()
    df = pd.read_csv(BytesIO(content))

    check_csv_injection(df)
    df = df.applymap(lambda x: clean_input(str(x)) if isinstance(x, str) else x)
    print("Фильтрация данных завершена.")
    
    df['Price_hashed'] = df['Price'].apply(hash_price)
    print("Столбец с хешированными ценами добавлен.")

    df['Price_Encrypted'] = df['Price'].apply(encrypt_price)
    df['RAM_Size_Encrypted'] = df['RAM_Size'].apply(encrypt_price)
    print("Столбец с зашифрованными ценами добавлен.")

    # Выводим 5 расшифрованных значений RAM
    print("\nПримеры расшифрованных значений RAM:")
    for encrypted_value in df['RAM_Size_Encrypted'].head(5):
        decrypted_value = decrypt_ram(encrypted_value)
        print(f"Зашифровано: {encrypted_value[:30]}... → Расшифровано: {decrypted_value}")

    # Сохранение обработанных данных
    output_path = "Laptop_price_secured.csv"
    df.to_csv(output_path, index=False)
    print(f"Обработанный файл сохранен: {output_path}")

    predictions = model.predict(df)
    return {"predictions": predictions.tolist()}

@app.get("/")
def read_root():
    return {"message": "Model is ready for predictions"}
