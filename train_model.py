import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def generate_telco_data(n_samples=3000):
    """Generates synthetic but highly realistic telecom customer data."""
    print("Generating synthetic telco data...")
    np.random.seed(42)
    
    # Generate features
    tenure = np.random.randint(1, 72, n_samples)
    monthly_charges = np.round(np.random.uniform(20.0, 120.0, n_samples), 2)
    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.5, 0.3, 0.2])
    tech_support = np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2])
    
    # Create logic for the target variable (Churn)
    # Customers are more likely to churn if they are on month-to-month, have low tenure, or no tech support
    churn_prob = np.zeros(n_samples)
    churn_prob += np.where(contract == 'Month-to-month', 0.4, 0.0)
    churn_prob += np.where(tenure < 12, 0.3, 0.0)
    churn_prob += np.where(tech_support == 'No', 0.2, 0.0)
    churn_prob += np.where(monthly_charges > 80, 0.1, 0.0)
    
    # Normalize probabilities and generate binary outcome
    churn_prob = np.clip(churn_prob, 0, 1)
    churn = np.random.binomial(1, churn_prob)
    
    df = pd.DataFrame({
        'Tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'Contract': contract,
        'TechSupport': tech_support,
        'Churn': churn
    })
    
    return df

def build_and_train_pipeline():
    # 1. Get Data
    df = generate_telco_data()
    
    # Save a sample for the app to use later
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/churn_data.csv", index=False)
    
    # 2. Split Features (X) and Target (y)
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Define Preprocessing Steps
    # Numerical data gets scaled (standardized)
    numeric_features = ['Tenure', 'MonthlyCharges']
    numeric_transformer = StandardScaler()
    
    # Categorical data gets One-Hot Encoded (turned into 1s and 0s)
    categorical_features = ['Contract', 'TechSupport']
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # 4. Create the final ML Pipeline
    print("Building and training the Random Forest Pipeline...")
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42))
    ])
    
    # 5. Train the model
    pipeline.fit(X_train, y_train)
    
    # 6. Evaluate the model
    y_pred = pipeline.predict(X_test)
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 7. Export the trained pipeline
    os.makedirs("model", exist_ok=True)
    joblib.dump(pipeline, 'model/churn_pipeline.pkl')
    print("\nSuccess! Pipeline saved to model/churn_pipeline.pkl")

if __name__ == "__main__":
    build_and_train_pipeline()