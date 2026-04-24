import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    'area': [1000, 1500, 2000, 2500, 3000],
    'price': [200000, 300000, 400000, 500000, 600000]
}

df = pd.DataFrame(data)

X = df[['area']]
y = df['price']

model = LinearRegression()
model.fit(X, y)

print(model.predict([[2200]]))
