🛡️ Insurance Fraud Detection API

This project is an end-to-end machine learning pipeline designed to detect fraudulent insurance claims using a FastAPI-based backend. The model is trained on historical data, enhanced with SMOTE for class balancing, and deployed via an API that returns prediction results including fraud probability and risk label.

🚀 Features

✅ Real-time fraud prediction using JSON-based FastAPI endpoint

✅ Preprocessing pipeline with cleaning and feature engineering

✅ Balanced dataset using SMOTE for improved fraud detection

✅ RandomForestClassifier model for robust classification

✅ Probability scoring with fraud risk labels (SAFE, REVIEW, FRAUD)

🤖 Machine Learning Approach

This project uses Supervised Machine Learning - Classification to detect fraud.

Algorithms Used:

Random Forest Classifier: A tree-based ensemble learning model for binary classification

SMOTE (Synthetic Minority Over-sampling Technique): A data balancing technique to handle class imbalance

📁 Project Structure

![insurance_project_structure](https://github.com/user-attachments/assets/fbe130cf-565d-48f3-b47b-73a8a05362b0)


insurance-fraud-detection/
├── app/                 # FastAPI app
├── scripts/             # Data cleaning and model training
├── models/              # Trained model (fraud_model.pkl)
├── data/                # Raw and processed datasets
├── requirements.txt     # Dependency list
└── README.md            # Project overview

🧠 Model Performance


![output](https://github.com/user-attachments/assets/5a161214-2a8f-4c8f-9e4c-39679ede5ad4)

![model_performance_table](https://github.com/user-attachments/assets/74629e02-e871-447d-9908-bf6030d5d2cf)


Overall Accuracy: ~79%

🧱️ Architecture Diagram

![fraud_architecture_vertical](https://github.com/user-attachments/assets/dcda7e39-25e5-41e5-b07a-33eba6610a7a)


The model is trained using cleaned.csv with SMOTE and RandomForest

Trained model is saved as fraud_model.pkl

FastAPI loads this model on request

Users POST JSON data to /predict and receive prediction + probability

📊 Prediction Example

![Screenshot 2025-05-27 211201](https://github.com/user-attachments/assets/2f902543-bd95-4f84-a1de-dbd50103df81)


{
  "fraud_prediction": 0,
  "fraud_probability": 41,
  "risk_label": "⚠️ MANUAL REVIEW"
}

🚪 How to Run Locally

# Clone the repo
https://github.com/Gireeshboddu/insurance-fraud-detection.git
cd insurance-fraud-detection

# Create virtual env and install dependencies
python -m venv venv
venv\Scripts\activate  # For Windows
pip install -r requirements.txt

# Run FastAPI
cd app
uvicorn main:app --reload

# Open browser
http://127.0.0.1:8000/docs

🔹 Future Enhancements

📅 Scheduled model retraining with Airflow

🌐 Deploy to Render/AWS EC2

📈 Streamlit dashboard for manual review insights

🔒 Authentication with OAuth2 for API access

📖 License

MIT License — feel free to use, fork, and contribute.

Built with ❤️ by Gireesh Boddu
