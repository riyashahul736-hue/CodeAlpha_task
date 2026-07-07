import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

data = {
    'age': [
        25, 35, 45, 50, 30,
        22, 20, 24, 28, 26,
        40, 55, 32, 60, 38
    ],

    'income': [
        50000, 80000, 100000, 90000, 60000,
        15000, 10000, 18000, 20000, 22000,
        75000, 120000, 65000, 150000, 70000
    ],

    'loan': [
        50000, 100000, 120000, 150000, 70000,
        300000, 250000, 350000, 400000, 280000,
        100000, 150000, 80000, 200000, 90000
    ],

    'debt': [
        5000, 10000, 15000, 20000, 8000,
        80000, 70000, 90000, 100000, 75000,
        15000, 20000, 10000, 25000, 12000
    ],

    # 1 = Low Risk, 0 = High Risk
    'score': [
        1, 1, 1, 1, 1,
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1
    ]
}

df = pd.DataFrame(data)

X = df[['age', 'income', 'loan', 'debt']]
y = df['score']

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

with open('model/credit_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("New AI Credit Model Trained Successfully!")



