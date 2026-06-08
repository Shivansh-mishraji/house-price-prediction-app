import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
dataset=pd.read_csv("C:\\Users\\91727\\Desktop\\house price prediction\\house_prices_8000.csv")
print(dataset.head())
dataset.isnull().sum()
dataset.nunique()
print("Duplicate rows:", dataset.duplicated().sum())
dataset=dataset.drop_duplicates()
# Feature Selection and Data Preparation
X= dataset.drop('price',axis=1)
X=pd.get_dummies(X)
Y=dataset['price']

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(X_train,Y_train)

Y_pred=model.predict(X_test)
print("R2 Score:",r2_score(Y_test , Y_pred))
print("Mean Squared Error:",mean_squared_error(Y_test , Y_pred))


plt.scatter(Y_test , Y_pred)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=2, color='red')  
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()  
# If you want to include categorical columns in correlation, encode them first:
corr_data = pd.get_dummies(dataset)
print(corr_data.head())

# Save the trained model
import pickle

# Save model to file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("✅ Model saved as model.pkl")
