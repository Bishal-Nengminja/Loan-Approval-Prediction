# 🏦 Loan Approval Prediction System

![License: MIT][(https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Bishal-Nengminja/Loan-Approval-Prediction/blob/main/LICENSE)

## 🌟 Project Overview

This project provides an end-to-end solution for predicting loan approval status using a machine learning model. It includes a comprehensive Jupyter Notebook for data analysis, model training, and evaluation, as well as an interactive web application built with Streamlit for real-time predictions. The system is designed to showcase the complete workflow of a data science project — from data exploration to deployment.

## ✨ Key Features & Technologies

- **Machine Learning Model:** A Logistic Regression model is trained to predict loan approval with high accuracy.
- **Comprehensive Analysis:** The provided Jupyter Notebook details every step of the process, including exploratory data analysis, data cleaning, and feature engineering.
- **Interactive Web Application:** A user-friendly Streamlit app allows users to input various financial and personal details to get an instant prediction.
- **Model Deployment:** The trained model and data preprocessing tools (like the StandardScaler) are saved using `joblib`, making them easily loadable for the web application.

### 🧰 Libraries Used

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `streamlit`
- `joblib`
- `plotly`

## ⚙️ How It Works

1. **Data Exploration & Preprocessing:**  
   The `Loan_Approval_Prediction_Complete_Analysis.ipynb` notebook explores the dataset, handles categorical features using one-hot encoding, and scales numerical data.

2. **Model Training:**  
   A Logistic Regression model is trained on the processed data.

3. **Model Saving:**  
   The trained model and scaler are saved as `loan_approval_model.joblib` and `scaler.joblib`.

4. **Web Application Interface:**  
   The `app.py` script loads the saved model and scaler, then creates a Streamlit interface for user input.

5. **Real-Time Prediction:**  
   The web app takes user input, preprocesses it, and uses the trained model to generate a prediction instantly.

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- Jupyter Notebook (for analysis)
- Streamlit (for web application)

### Installation

1. **Clone the repository:**

```bash
[git clone https://github.com/YOUR_GITHUB_USERNAME/loan-approval-prediction.git](https://github.com/Bishal-Nengminja/Loan-Approval-Prediction)
cd loan-approval-prediction
````

2. **Install the required dependencies:**
```bash
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.1.post1
matplotlib==3.8.4
seaborn==0.13.2
streamlit==1.35.0
joblib==1.4.2
plotly==5.22.0
```

## 🧪 Usage

### To view the data analysis:

```bash
jupyter notebook Loan_Approval_Prediction_Complete_Analysis.ipynb
```

### To run the web app:

```bash
streamlit run app.py
```

## 📈 Results and Performance

The model’s performance (accuracy, confusion matrix, etc.) is detailed within the Jupyter Notebook. These metrics show its effectiveness in predicting loan approvals.

## 🤝 Contributing

Contributions are welcome!
Feel free to fork the repository, create a branch, and submit a pull request.

## 📄 License

This project is licensed under the [MIT License](LICENSE).
---

## 📞 Contact

* GitHub: Bishal Nengminja [(https://github.com/Bishal-Nengminja)]
* Email: bishalnengminja61@gmail.com

---

⭐️ If you found this helpful, consider giving it a ⭐️ on GitHub!
