## Applied Machine Learning II Code Samples

These assignments are included in my Introduction to Python workshop.


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



## Assignment 2
Create a program that prompts the user for a numeric test grade 
The numeric grade should be used to determine if a student has earned an A, B, C, D or F in the class, using the rules on this table :

![useful image](https://user-images.githubusercontent.com/52934249/61403530-b5c83800-a892-11e9-9e7f-b94c8b941aa4.png)


Use a print() statement to output the letter grade.


Sample code to get you started:

```markdown
score = input(“What is your exam score? ")
score = float(score)
if (score >=90) :
	…
```
## Assignment 3
Write a function that calculates whether or not a given year is a leap year and has 366 days instead of 365.

Use this algorithm:
- A year will be a leap year if it is divisible by 4 but not by 100. 
- If a year is divisible by 4 and by 100, it is not a leap year unless it is also divisible by 400
- 1996, 1992, 1988 are leap years because they are divisible by 4 but not by 100

The function should print :"The year XXXX is a leap year." or "The year XXXX is NOT a leap year."
It should accept one argument: a non-negative number representing a year to evaluate

Sample code to get you started:

```markdown
def  leapYear (year):
	isLeapYear = false
	#determine if this is a leap year
	print(..)
  
leapYear(1996)
leapYear(2000)
leapYear(2002)
```


