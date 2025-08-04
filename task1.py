import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("Cars Datasets 2025.csv", encoding='ISO-8859-1')
print(df.head())
print(df.info())
print(df.isnull().sum())  
print(df.describe())        

X = df.drop("Cars Prices", axis=1)
y = df["Cars Prices"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


num_cols = ['CC/Battery Capacity', 'HorsePower', 'Total Speed',             # Handle missing values
            'Performance(0 - 100 )KM/H', 'Cars Prices', 'Torque', 'Seats']


for col in num_cols:
    df[col] = df[col].str.extract(r'([\d.]+)').astype(float)  # non-numeric values (strip strings like " Nm", " HP", etc.)


for col in num_cols:                                     # (missing numeric values with median)
    df[col] = df[col].fillna(df[col].median())


cat_cols = df.select_dtypes(include='object').columns   # (Fill remaining categorical missing values )
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

df = pd.get_dummies(df, drop_first=True)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

import matplotlib.pyplot as plt

for col in num_cols:
    plt.figure()
    df.boxplot(column=[col])
    plt.title(f"Boxplot of {col}")
    plt.show()


for col in num_cols:                        # Remove outliers using IQR
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]





