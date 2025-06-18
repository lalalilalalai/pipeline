from fastapi import FastAPI, File, UploadFile, HTTPException
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
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        content = await file.read()
        df = pd.read_csv(BytesIO(content))
    except pd.errors.ParserError:
        raise HTTPException(status_code=400, detail="Invalid CSV file")
    finally:
        await file.close()
    predictions = model.predict(df)
    return {"predictions": predictions.tolist()}

@app.get("/")
def read_root():
    return {"message": "Model is ready for predictions"}
