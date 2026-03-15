
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

# Sample dataset
data = {
    "area":[1000,1500,2000,2500,3000],
    "price":[200000,300000,400000,500000,600000]
}

df = pd.DataFrame(data)

X = df[['area']]
y = df['price']

model = LinearRegression()
model.fit(X,y)

os.makedirs("saved_model", exist_ok=True)
joblib.dump(model,"saved_model/linear_model.pkl")

print("Model trained and saved to saved_model/linear_model.pkl")
