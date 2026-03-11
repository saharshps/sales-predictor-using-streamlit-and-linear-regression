import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import joblib

df = pd.read_csv(r"C:\Users\sahar\OneDrive\Documents\Advertising.csv", index_col=0)

X = df.drop('Sales', axis=1)
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.8, random_state=1)

model = LinearRegression()
model.fit(X_train, y_train)

predict = model.predict(X_test)   
print("Predicted:", predict)
print("Actual:", y_test)
print(model.coef_,model.intercept_)

accuracy=model.score(X_test,y_test)
print(accuracy)

print(model.predict([[3,4,1]]))

print(sns.pairplot(df))
plt.show()

cross_val=cross_val_score(model,X,y,cv=5)
print(cross_val.mean())
 
joblib.dump(model,"multilinear_regression_model")
print("'multilinear_regression_model' saved successfully")
