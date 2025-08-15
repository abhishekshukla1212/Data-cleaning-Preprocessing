# Car Classification using Support Vector Machine (SVM)

## 📌 Project Overview
This project uses a Support Vector Machine (SVM) model to classify cars based on their specifications.  
The dataset contains details like engine capacity, horsepower, torque, speed, and more.  
We preprocess the data, encode categorical features, tune hyperparameters, and evaluate model performance using cross-validation.

---

## 📂 Project Structure
SVM.ipynb # Main Jupyter Notebook with SVM implementation
│── requirements.txt # Python dependencies
│── README.md # Project documentation
│── dataset.csv # Car dataset (optional)

## ⚙️ Features
- Data cleaning & preprocessing (handling units, removing symbols, type conversion)
- Encoding categorical variables using `LabelEncoder`
- Train-Test split for model training
- SVM classifier with `rbf` kernel
- Hyperparameter tuning using `GridSearchCV`
- Cross-validation for model evaluation
- Visualization of decision boundaries (2D)

## 📊 Dataset
The dataset contains the following columns:
- **Company Names**
- **Cars Names**
- **Engines**
- **CC/Battery Capacity**
- **HorsePower**
- **Total Speed**
- **Performance(0 - 100) KM/H**
- **Cars Prices**
- **Fuel Types**
- **Seats**
- **Torque**

