from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

lr = LinearRegression()
X, y = load_diabetes(return_X_y=True)
reg = lr.fit(X, y)
print(reg.score(X, y))


