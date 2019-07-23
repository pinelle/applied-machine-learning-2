## Applied Machine Learning II Code Samples

These are the code samples covered in the slides for my Applied Machine Learning II workshop.

## Simple Linear Regression (Ordinary Least Squares)

```markdown
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import r2_score

# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# R-squared score: 1 is perfect prediction
print('R-squared: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='blue')
plt.plot(diabetes_X_test, diabetes_y_pred, color='red', linewidth=1)
plt.xticks(())
plt.yticks(())
plt.show()
```




## Polynomial Regression

```markdown
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score

#Pressure
y_train = np.array([
     [0.002],
     [0.0012],
     [0.0060],
     [0.0300],
     [0.0900],
     [0.2799]])

#Temperature
X_train = np.array([
    [0],	
    [20],	
    [40],	
    [60],	
    [80],	
    [100]])

y_test = np.array([
     [0.001],
     [0.0092],
     [0.0060],
     [0.0300],
     [0.0900],
     [0.2799]])

#Temperature
X_test = np.array([
    [20],	
    [80],	
    [40],	
    [60],	
    [80],	
    [100]])

# Fitting Linear Regression to the dataset 
lin = LinearRegression()
lin.fit(X_train, y_train) 
y_pred = lin.predict(X_test)
print('\n\nR-squared: %.2f' % r2_score(y_test, y_pred))

# Fitting Polynomial Regression to the dataset 
poly = PolynomialFeatures(degree = 4) 
X_poly = poly.fit_transform(X_train) 
poly.fit(X_poly, y_train) 
lin2 = LinearRegression() 
lin2.fit(X_poly, y_train) 

# Visualising the Linear Regression results 
plt.scatter(X_train, y_train, color = 'blue') 
plt.plot(X_train, lin.predict(X_train), color = 'red') 
plt.title('Linear Regression') 
plt.xlabel('Temperature') 
plt.ylabel('Pressure') 
  
plt.show() 

# Visualising the Polynomial Regression results
y_pred = lin2.predict(poly.fit_transform(X_test)) 
print('R-squared: %.2f' % r2_score(y_test, y_pred))

plt.scatter(X_train, y_train, color = 'blue') 
plt.plot(X_train, lin2.predict(poly.fit_transform(X_train)), color = 'red') 
#Plot titles  
plt.show() 
```

## Logistic Regression

```markdown
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

#x contains bid price, y contains win (1) / loss (0)
x = np.array([100,120,150,170,200,200,202,203,205,210,215,250,270,300,305,310])
y = np.array([1,1,1,1,1,1,1,0,1,0,0,0,0,0,0,0])

x_test = np.array([120,155,174, 200,202,203,215, 250 ,400, 510, 660, 529, 660, 710, 888, 900])
y_test = np.array([1,1,1,1,1,0,1,0,0,0,0,0,0,0,0,0])

#Convert a 1D array to a 2D array in numpy
X = x.reshape(-1,1)
X_test = x_test.reshape(-1,1)

logreg = LogisticRegression(C=1.0, solver='lbfgs', multi_class='ovr')
logreg.fit(X, y)

predictions = logreg.predict(X_test)
score = logreg.score(X_test, y_test)
print(score)

prices = np.arange(100, 310, 0.5) #create values, step .5, for plot
probabilities= []
for i in prices:
    p_loss, p_win = logreg.predict_proba([[i]])[0]
    probabilities.append(p_win)
    
plt.scatter(prices,probabilities) #display linear regression model    
plt.title("Logistic Regression Model")
plt.xlabel('Price')
plt.ylabel('Status (1:Won, 0:Lost)')
```

## Multiclass Logistic Regression

```markdown
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits

digits = load_digits()
print('Image Data Shape' , digits.data.shape) # 1797 images, 8 x 8 pixels = 64
print("Label Data Shape", digits.target.shape) #1797 labels (integers from 0–9)

#plot the first five images in the dataset
plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
	plt.subplot(1, 5, index + 1)
	plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
	plt.title('Training: %i\n' % label, fontsize = 20)

x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)

logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)
predictions = logisticRegr.predict(x_test)

score = logisticRegr.score(x_test, y_test)
print(score)
```

## Decision Trees

```markdown
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

#569 instances, 30 features
print("\nTarget names:", cancer['target_names'])
print("\nFeature names:\n", cancer['feature_names'])

X_train, X_test, y_train, y_test = train_test_split(
cancer.data, cancer.target, stratify=cancer.target, random_state=42)

#create tree without any pre-pruning
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)
print("\nTree with no pre-pruning")
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}\n".format(tree.score(X_test, y_test)))

#create tree by limiting it to max-depth of 4
print("Tree with max depth of 4")
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)
print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}\n".format(tree.score(X_test, y_test)))
```

## Random Forests

```markdown
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)

forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(forest.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(forest.score(X_test, y_test)))
```

## K-fold Cross Validation

```markdown
from numpy import array
from sklearn.model_selection import KFold

data = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
# prepare cross validation
kfold = KFold(3, True, 1)
# enumerate splits
for train, test in kfold.split(data):
	print('train: %s, test: %s' % (data[train], data[test]))
```

