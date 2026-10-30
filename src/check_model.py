import joblib
import pandas as pd 
#1. load trained model
model = joblib.load("models/best_churn_model.joblib")

print("Model:", model)
print("Pipeline steps:", model.named_steps.keys())

print("\nPreprocessor:")
print(model.named_steps["preprocessor"])

print("\nTrained classifier:")
print(model.named_steps["model"])  

print("Model loaded successfully!")

# 2. Give new customer details
new_customer = pd.DataFrame([{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 75.5,
    "TotalCharges": 906.0
}])

# 3. Predict
prediction = model.predict(new_customer)

# 4. Display result
if prediction[0] == 1:
    print("Customer will Churn")
else:
    print("Customer will NOT Churn")

