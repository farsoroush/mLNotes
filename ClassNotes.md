# Machine Learning Notes

## Disclosure:
The notes here are personal notes on various machine learning courses (ML courses) and can be used by others. If you find any mistakes in understanding or in writing (or in any of the content) please feel free to comment or contact me via serpoush [at] gmail. 

## Course 1: Machine Learning Fundamentals by Andrew Ng
### Week 1
The basic definition of ML can come from Arthur Samuel in 1959: "Field of study that gives computers the ability to learn without being explicitly programmed."
The first event was Sameul's program to predict the outcome of the checkers game. 
The definition is more of an informal one. 

Anyhow: there are two main types of learning: 
- **supervised learning**
  -   The first type is `regression` in which the algorithm refers X (input) to Y (output), that is, the algorithm learns from the given "right answers" to make predictions. various examples of this could be:
    -   audio-to-text conversion
    -   house size to price prediction
    -   image+sensor data to driving decision (self-driving car)
    -   etc.
  -   The second type of supervised learning problem is `the classification` problem. One example could be cancer detection between benign and malignant.
    - The main difference is here the model predicts 0 and 1 ( i.e. categories OR classes), as opposed to the regression model which reflects a range of values.   
- **unsupervised learning**
  - here we train the algorithm with data that is not labeled, with benign/malignant labels in the case of cancer detection. We just try to learn or detect any pattern or any structure in the dataset. The algorithm cannot tell us whether the tumor, for example, is benign or malignant, however, it can find out which data points have similar properties, aka clustering.
  - Another example can be grouping the customers based on various variables they entered into the website.
  - A more formal definition of unsupervised learning is "data has input X labels and no output y labels and the algorithm needs to find *structure* in the data."
  - `Types`:
    - Clustering/Grouping
    - Anomaly Detection: finding unusual data points. 
    - Dimensionality Reduction
There are two more types but those are more specialized: 
- recommender systems
- Reinforcement learning

## Linear Regression with one Variable:
Simple problem can be the prediction of house prices based on the house size. One might be able to come up with a linear equation like `y = ax +b`. 
This is a supervised learning example as we are training the model with "right answers" first and then the model predicts the numebrs. 

The similar idea in classification would be training the models with pictures of cats vs dogs and then ask for picture detection. 

One main difference between the two, classification vs regression, is the outcome of the first can be limited to the number of classes and the later would be infinite number of possible results as the outcome is a real number. 

***Terminology:***
  - Training Set: the data used to train the model (Da! :D )
  - `x` is the input variable and called `feature`
  - `y` is the output variable and called the `target`
  - `(x,y)` single training example
  - `m` number of training examples
  - $`(x^{(i)},y^{(i)})`$ shows the $i^{th}$ training example

Traditionally the training set would be introduced to a learning algorithm and the outcome would be a **hypothesis** $*f*$, which will be called **function** or **the model** from now on. The function will take $`x`$ as input and put out **prediction or estimate** $`\hat{y}`$. So, be aware of the distinction:

- $y$ is the true target
- $`\hat{y}`$ is the estimate or the prediction of the model.

The proper representation, mathematically, will be:
```math
f_{w,b} = w x + b            
```
where w and b are generate by the model and called **parameters**. Same concept can be expanded to the non-linear equations to create **non linear regression model**.

> The lab note uses the codes below:
  <details>
      <summary>The snippets of python code used</summary>
      The code is written in Python. 
    
  ```python
  import numpy as np
  import matplotlib.pyplot as plt
  plt.style.use('./deelplearning.mplstyle')
  
  #x_train is the input variable (size in 1000 square feet)
  #y_train is the target variable (price in 1000s of dollars)
  x_train = np.array([1.0,2.0])
  y_train = np.array([300.0,500.0])
  print(f"x_train = {x_train}")
  print(f"y_train = {y_train}")
  #m is the number of training examples:
  print(f"x_train.shape: {x_train.shape}")
  m=x_train.shape[0]
  print(f"Number of training example is: {m}")
  
  #Plot the data points
  plt.scatter(x_train, y_train, marker='x', c='r')   # showing the data points using red crosses
  #Set the title
  plt.title("Housing Prices")
  #Set the y-axis label
  plt.ylabel('Price (in 1000s of dollars)')
  #Set the x-axis label
  plt.xlabel('Size (1000 sqft)')
  plt.show()
  
  #modeling
  w = 100
  b = 100
  print(f"w: {w}")
  print(f"b: {b}")
  
  def compute_model_output(x, w, b):
      """
      Computes the prediction of a linear model
      Args:
        x (ndarray (m,)): Data, m examples 
        w,b (scalar)    : model parameters  
    Returns
      f_wb (ndarray (m,)): model prediction
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb

#now lets compute the model output function:
tmp_f_wb = compute_model_output(x_train, w, b,)

# Plot our model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()
```   
  </details>

### Week 2
#### Cost Function

This function is our measure for checking how well our predictions are. The aim in the regression is to get a function that gets predictions as close as possible to the targets. Cost function measures the **Mean Squared Errors** or 

$$J(w,b) = \frac{1}{2m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})^2 = \frac{1}{2m} \sum_{i=1}^m (f_{w,b}(x^{(i)}) - y^{(i)})^2$$

Where parameters, w and b in the case of the linear regression example, can be adjusted to improve the model. So the goal is to minimize the function $J(w,b)$. Simplified model can use a $b=0$ to only play with $w$.

So, the goal is to minimize the cost function. One of the proposed solutions is the **Gradient Descent** and it can be used to minimize any function. 

For example, when we have a cost function $`J(w_1, w_2, w_3, w_4, ..., w_n, b)`$ and the aim is to find $min_{w_1, w_2, w_3, w_4, ..., w_n, b} J(w_1, w_2, w_3, w_4, ..., b)$. The initial guess can be $w=0 \text{  and   } b=0$. and then we keep chaning w and b to reduce the final value for J. 

> Note that the minimum can be unique or non-unique - i.e. local minimum - depending on the complexity of regression function and J.

Here is the actual equation for the mimization:

$`w = w - \alpha \frac{\partial}{\partial w} J(w,b) \text{          Note that the new value of w is replaced by the new calculation on the RHS of the Eqn.} `$

$` b = b - \alpha \frac{\partial}{\partial b} J(w,b) `$

$\alpha$ is defined to be the **learning rate** or how fast/slow the algorithm would take its steps. The derivative term decides in which direction we are taking steps of $\alpha$.

We take these steps until the **algorithm converges**. It is importan to SIMULTANEOUSLY update w and b. So, the correct implementation would be:

$` temp-w = w - \alpha \frac{\partial}{\partial w} J(w,b) `$

$` temp-b = b - \alpha \frac{\partial}{\partial b} J(w,b) \text{      where the old value of w is used in the  calculation.} `$

$` w = temp-w `$

$` b = temp-b `$

> Note that **Batch** gradient descent refers to the case that user uses all the datasets to minimize the cost function. 

### Multiple Linear Regression:

In the case of the linear regression we only had one parameter/feature, which is far from reality. A more practical case is when you have multiple features, i.e. $x_1, x_2, x_3, x_4, ..., x_n$. Example for the house pricing could be sqft, # of bdrm, # of bathrm, Building age, and target would be the price. $n$ will be used as the indicator of the number of features. $\overrightarrow{x^{(i)}}$ shows the vector of features and it would be equal to $\overrightarrow{x^{(i)}} = (x_1, x_2, x_3, x_4, ..., x_n)$. 

In this case the regression model becomes:
```math
\overrightarrow{x^{(i)}} = [w_1  w_2  \cdots w_n] \times [x_1 \\\ x_2 \\\ x_3 \\\ x_4, ..., x_n]^T + b
```

The vecotization format in python using the `numpy` package would be: `f = np.dot(w,x)+b`. without vecortization it would be:
```python
for j in range (0,n):
  f = f + w[j] * x[j]
```
> Note that vectorized format, using the numpy `np.dot` will perform faster since it multiplies the vector elements in parallel.

At this point the cost function and minimization of it would be changed, or expanded, to adopt the fact that w, b, and x are all now vectors or matrices. The rest of the concepts for the gradient decend would be the same. 

### Feature Scaling:
The main goal of this part is to find a way to run the gradient descent faster when we are dealing with multiple linear regression. In the example of the house price, imagine the impact of parameter for the size of the house  vs the parameter for the number of bedrooms, two features are considerable different in value range and as such their parameters are vastly different. The reflection on the J contour plots would be long concentric oval shapes showing which one of the parameters will change more drastically. in this case, the gradient descent formula will bounce quite a few rounds before full convergence. The solution to this problem is to **re-scale** the data so the features will have similar variations and J contour plots will be more like concentric circles. The other solution is to perform **mean normalization**, that is, finding the average of each feature, suppose $\mu_i$, and recalculate each feature based off of the following equation:
```math
x_{i,new} = \frac{x_i - \mu_i}{x_{i,max} - x_{i, min}}
```
The outcome of this method is features realignment in a $[-1,1]$ range.

The other solution is **Z-score normalization** where one would use the mean and standard deviation of the data and uses the following for normalization:
```math
x_{i,new} = \frac{x_i - \mu_i}{\sigma_i}
```
The outcome this time would be a normalzied in a symmetrical range of $[-a,a]$, where a is any given number. This would be the ideal case, where minor deviations are acceptable too. 

The last one can happen in python using the code belwo from `sklearn` library:

```python
from sklearn.preprocessing import scale
scale(X_orig, axis=0, with_mean=True, with_std=True, copy=True)
```

> Note: One could check the range of each feature using the following line of code in pythong:
> ```python
> np.ptp("Dataset", axis=0)
> ```

#### Practical Tips:

1. Make sure the learning curve is working properly, that is, J is decreasing constantly and make sure there is a convergence test to stop where the decrease is not significant anymore.
2. Make sure the learning rate, $\alpha$, is selected properly. One example could be when J is jumping up and down as iterations progress. 
3. With small enough learning rate, J must be decreasing. If not, it means most probable there is a mistake in the written code.
4. One could use a small learning rate of 0.0001 and then $\times 3$ to get larger and observe the iterations and J progression. 

### Feature Engineering and Polynomial Regression
There are cases where multiple features could be combined into one new feature, such as the case where you have width and height of a lot as two features. In the case of the example given, one could multiply the two features and come up with a new feature, aka area. Such actions would result in the creation of polynomial expressions for the regression and would mean a more complex model to handle. The previous concept for J still applies, only require more algebraic works. 

### SciKit practical code lines:
ScikitLearn (SKL) has a gradient descent regression model which works the best on the normalized data. User can use the SKL built-in scaler, StandardScaler(), which uses z-score normalization. Below is the code snippet example:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
# Import data (below is from a previously ran example):
X_train, y_train = load_house_data()
X_features = ['size(sqft)','bedrooms','floors','age']

#Using Scaler:
scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)

#Using Regression Model:
sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_norm, y_train)
print(sgdr)
print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")
```

The outcome looks like:

```
SGDRegressor(alpha=0.0001, average=False, early_stopping=False, epsilon=0.1,
             eta0=0.01, fit_intercept=True, l1_ratio=0.15,
             learning_rate='invscaling', loss='squared_loss', max_iter=1000,
             n_iter_no_change=5, penalty='l2', power_t=0.25, random_state=None,
             shuffle=True, tol=0.001, validation_fraction=0.1, verbose=0,
             warm_start=False)
number of iterations completed: 138, number of weight updates: 13663.0
```
Viewing the parameters:
```python
b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameters:                   w: {w_norm}, b:{b_norm}")
```

Using the model and predict:
```python
# make a prediction using sgdr.predict()
y_pred_sgd = sgdr.predict(X_norm)
# make a prediction using w,b. 
y_pred = np.dot(X_norm, w_norm) + b_norm  
print(f"prediction using np.dot() and sgdr.predict match: {(y_pred == y_pred_sgd).all()}")

print(f"Prediction on training set:\n{y_pred[:4]}" )
print(f"Target values \n{y_train[:4]}")
```


### Week 3
#### Classification:
The output variable $\hat{y}$ here will have discrete values as opposed to regression where the output $\hat{y}$ had a continuous value. 

Typical examples of this problem could be:

|Question|Answer "yes or no"|
| :----------- | :-----------: |
|Is this email **spam**?|yes or no|
|Is the transaction **fraudulent**?|yes or no|
|Is the tumor **malignant**?|yes or no|

The above examples are known as **binary** classification as the output only can be one of the two values listed. As such, we could replace the yes/no terms with True/False or 1/0 instead. When 0 and 1 used, one could call them negative and positive class and it must be noted that it does not mean "bad" vs "good" necessary. This is just to categorize/distinguish them.   

> Note: The terms **Class** and **Category** may be used interchangably.

> Note 2: One could say that the classification problem can be modeled using linear regression where a function of $f_{w,b} (x) = wx+b$ predicts the outcome. Then a **decision boundary** is designated and assign $f_{w,b} (x)<0.5  ==> \hat{y} = 0$ and $f_{w,b} (x)>0.5  ==> \hat{y} = 1$. $0.5$ can be called the threshold. Note that additional data points, suppose another malignant point with high $x$ value, will change $(w,b)$ such that the new decision boundary is moved; with the new boundary some of the previously "malignant/yes" cases are now "benign/no."

**Logistic Regression** is what can be used instead of the linear regression models. 

#### Logistic Regression:
This regression fits a "S" shaped curve, aka **Sigmoid Function**, to the data. The upper range for the function is 1 and the lower range is 0 with the following function:

```math
g(z) = \frac {1} {1+e^{-z}}

\text{ to apply this function and create a logistic regression:}
z = \overrightarrow{w} \cdot \overrightarrow{x} + b

\text{ \ So:}
f_{\overrightarrow{w},b} (\overrightarrow{x}) = g(\overrightarrow{w} \cdot \overrightarrow{x} + b) = \frac {1} {1+e^{-\overrightarrow{w} \cdot \overrightarrow{x} + b}}
```

The outcome of the logistic regression is a *probability* of the expectation. So, the outcome of 0.7 means that $\hat{y}$ has a 70% chance of being 1 and 30% chance of being 0; or written mathematically: $= P(y=1 | x;\overrightarrow{w},b)$ which is read the probability of y=1 for x given values of w and b. 

Here is how the Sigmoid function is created in python:

```python
def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
         
    """

    g = 1/(1+np.exp(-z))
   
    return g
```

##### Decision boundary:
Ultimately we need to define a threshold where the model returns 1 or 0, instead of a probability. 

**Simplified Cost function for the logistic regression model:**

```math
J(\mathbf{w},b) = \frac{1}{m} \sum_{i=0}^{m-1} \left[ loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) \right] \tag{1}
```

where

* $loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)})$ is the cost for a single data point, which is:

    $loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) = -y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) \tag{2}$
    
*  where m is the number of training examples in the data set and:
```math
\begin{align}
  f_{\mathbf{w},b}(\mathbf{x^{(i)}}) &= g(z^{(i)})\tag{3} \\
  z^{(i)} &= \mathbf{w} \cdot \mathbf{x}^{(i)}+ b\tag{4} \\
  g(z^{(i)}) &= \frac{1}{1+e^{-z^{(i)}}}\tag{5} 
\end{align}
```

And here is a simple function for the cost function in python:

```python
def compute_cost_logistic(X, y, w, b):
    """
    Computes cost

    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """

    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i],w) + b
        f_wb_i = sigmoid(z_i)
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
             
    cost = cost / m
    return cost
```
