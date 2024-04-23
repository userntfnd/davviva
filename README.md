# davviva

# expt 1 linear regression
```
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
dataset = pd.read_csv("salary_data.csv")
dataset.head()
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 1/3, random_state = 1)
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
plt.scatter(X_test,y_test, color = "red")
plt.plot(X_test,y_pred, color = "blue")
plt.title("salary vs experience")
plt.xlabel("Years of exp")
plt.ylabel("salary")
plt.show()
```

# expt 2 multiple regression
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("Student_Performance.csv")

dataset.head()

X = dataset.drop(columns=['Performance Index'])
y = dataset['Performance Index']
X_encoded = pd.get_dummies(X, columns=['Extracurricular Activities', 'Sample Question Papers Practiced'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plt.scatter(X_test['Hours Studied'], y_test, color='blue', label='Actual')
plt.scatter(X_test['Hours Studied'], y_pred, color='red', label='Predicted')
plt.xlabel('Hours Studied')
plt.ylabel('Performance Index')
plt.title('Scatterplot for Hours Studied vs Performance Index')
plt.legend()
plt.show()

plt.plot(X_test['Previous Scores'], y_test, 'o', color='blue', label='Actual')
plt.plot(X_test['Previous Scores'], y_pred, 'o', color='red', label='Predicted')
plt.xlabel('Previous Scores')
plt.ylabel('Performance Index')
plt.title('Line chart for Previous Scores vs Performance Index')
plt.legend()
plt.show()

plt.hist(y_test, color='blue', alpha=0.5, label='Actual' )
plt.hist(y_pred, color='red', alpha=0.5, label='Predicted')
plt.xlabel('Performance Index')
plt.ylabel('Frequency')
plt.title('Histogram of Performance Index')
plt.legend()
plt.show()
```
