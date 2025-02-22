from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score

lr = LinearRegression()
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
reg = lr.fit(X_train, y_train)
print(r2_score(y_test, reg.predict(X_test)))


