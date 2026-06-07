import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

def train_model():
    print("Loading dataset...")
    # Load dataset
    df = pd.read_csv("house_prices_8000.csv")
    
    # Drop duplicates and handle missing values if any
    df = df.drop_duplicates()
    df = df.dropna()
    
    # Separate features and target
    # Features we want to use (must match what app.py provides)
    numeric_features = [
        'sqft_living', 'sqft_lot', 'bedrooms', 'bathrooms', 'floors', 
        'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 
        'yr_built', 'yr_renovated', 'garage', 'parking', 'hoa_monthly',
        'lat', 'long', 'zipcode'
    ]
    categorical_features = ['city', 'neighborhood']
    
    # Check if target exists
    if 'price' not in df.columns:
        raise ValueError("Target column 'price' not found in dataset.")
        
    X = df[numeric_features + categorical_features + ['waterfront']]
    y = df['price']
    
    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # We will treat 'waterfront' as numeric or passthrough since it's already 0/1
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('passthrough', 'passthrough', ['waterfront'])
        ])
        
    # Append regressor to preprocessing pipeline
    # Using RandomForest for robust, powerful predictions out of the box
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training model (this might take a few seconds)...")
    model_pipeline.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model_pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"R2 Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:,.2f}")
    
    print("Saving model to model.pkl...")
    with open('model.pkl', 'wb') as file:
        pickle.dump(model_pipeline, file)
        
    print("✅ Model trained and saved successfully!")

if __name__ == "__main__":
    train_model()
