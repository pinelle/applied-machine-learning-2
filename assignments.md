## Introduction to Python Assignments

These assignments are included in my Introduction to Python workshop.


## Assignment 1

```markdown
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

#create data
X, y = make_regression(n_samples = 300, n_features=1, noise=8, bias=2)
y2 = y**2

#create model
poly_features = PolynomialFeatures(degree =15)  
X_poly = poly_features.fit_transform(X)
poly_model = LinearRegression()  
poly_model.fit(X_poly, y2)
y_pred = poly_model.predict(X_poly)

#print score, plot results
new_X, new_y = zip(*sorted(zip(X, y_pred))) # sort values for plotting
plt.plot(new_X, new_y)
plt.scatter(X,y2)
print(r2_score(y2, y_pred))
```



