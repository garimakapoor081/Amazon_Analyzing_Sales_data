import pandas as pd
import numpy as np
# Load datasets
products =("C:\\Users\\GARIMA KAPOOR\\Downloads\\amazon.csv")
df=pd.read_csv(products)
print(df)
print(df.shape)

# header
print(df.head(5))
# Coloumn
print(df.columns)


# Display basic info and check for missing values
print(df.info())
print(df.isnull().sum)

# Basic statistics
print(df.describe())

pd.set_option('display.Max_Columns',None)

# Fill missing values or drop rows/columns as necessary
ismissing_cols = [col for col in df.columns if 'ismissing' in col]
df['total_missing'] = df[ismissing_cols].sum(axis=1)


import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df, x="actual_price",bins=30,color="red")
df.hist(figsize=(15,10))
plt.title("Histogram data distribution")
plt.xlabel('actual_price  ')
plt.ylabel('category')
plt.show()

# Top-selling products
top_products = df.groupby(['discounted_price'],as_index=False).sum().sort_values(by=['actual_price'],
ascending=False).head(10)
sns.barplot(x=top_products.index , y=top_products['actual_price'])
plt.xlabel('actual_price')
plt.ylabel('discounted_price')
plt.title("Top Product")
plt.show()

# Actual Pricing vs rating
plt.scatter(df['actual_price'],df['discounted_price'])
plt.xlabel('actual_price')
plt.ylabel('discounted_price')
plt.show()





#Train test split
from sklearn.model_selection import train_test_split
X = df['actual_price']
y =df['discounted_price']
print(X)
print(y)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=1)
print("X_train before Scalar:\n",X_train)
print("X_test before Scalar:\n",X_test)
from sklearn.preprocessing import StandardScaler

#Linear Regrashion Train Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Coefficient
print("Coefficient/Slope: ",regressor.coef_)
print("Constant / Intercept: ",regressor.intercept_)


#prediction
y_pred = regressor.predict(X_test)
result_df = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print(result_df)










