# 🚗 Car Price Prediction – Data Cleaning & Preprocessing

This project demonstrates how to clean, preprocess, and visualize a raw dataset of car specifications to prepare it for machine learning models.

---

## 📁 Dataset

- **Source:** Cars Datasets 2025.csv
- **Size:** 1,218 rows × 11 columns
- **Target Variable:** `Cars Prices`

---

## 🔧 Data Preprocessing Steps

1. Handled missing values using **median/mode**.
2. Converted **object columns to numeric** (removed 'HP', 'Nm', etc.).
3. Encoded categorical features using **One-Hot Encoding**.
4. Scaled numeric features with **StandardScaler**.
5. Removed outliers using **IQR method**.
6. Split into **training and test sets** (80:20 ratio).

---

## 📊 Visualizations

### Boxplot of Car Prices

![Boxplot](price_boxplot.png)

---

## 📄 Output Files

- [`cleaned_cars_dataset.csv`](cleaned_cars_dataset.csv): Final cleaned dataset
- [`model_output.csv`](model_output.csv): Model predictions vs actual prices
- [`results.txt`](results.txt): Model performance and details

---

## 💻 Technologies Used

- Python 🐍
- Pandas & NumPy
- Scikit-learn
- Matplotlib
- Git & GitHub

---

## 🧠 Author

**Abhishek Shukla**  
[GitHub](https://github.com/abhishekshukla1212)

---
