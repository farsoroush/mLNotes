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
  f_{\mathbf{w},b}(\mathbf{x^{(i)}}) = g(z^{(i)}) \tag{3} 
  z^{(i)} = \mathbf{w} \cdot \mathbf{x}^{(i)}+ b \tag{4} 
  g(z^{(i)}) = \frac{1}{1+e^{-z^{(i)}}} \tag{5} 
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


## Course 2: Advanced Learning Algorithms by Andrew Ng

### Neural Network Model:

### TensorFlow (TF) Implementation:
Exmaple discussed is coffee roasting. if X is the input array of temperatures and layer 1 activation function is a^[1], and output or the second layer activation is a^[2], then the code for inference in TF will be:
``` python
x = np.array([[200.0,17.0]]) # temp and duration for coffee roasting
layer_1 = Dense(units=3, activation = 'sigmoid')
a1 = layer_1(x)
layer_2 = Dense(units=1, activation = 'sigmoid')
a2 = layer_2(a1)
# if we want a threshold value, suppose a2>0, then we can apply the following:
if a2>=0.5:
  yhat = 1
else:
 yhat = 0
```

More details can be found in the lab notes. 

### Data in TensorFlow (TF):
As TF and Numpy were developed in parallel, the data structure is different between the two libraries. We will discuss the differences here. 
Example 1 is Featuring vectors:
Suppose you have a 2X3 matrix, 2 rows and 3 columns, so the array would be `np.array([[1,2,3],[4,5,6]])` in numpy. Notice to show a 2D array we are using [[ and ]] showing start and end of the array. So the difference between [[200,17]] and [200,17] is the first one is a 2D array while the second one is a 1D vector.
TF is designed with the purpose of efficiently processing large datasets. So a 2D vector will show up as a Tensor rather than a matrix. In the code example above, `a2` is a Tensor object in tf and will be shown as `tf.Tensor([[0.8]],shape(1,1), dtype=float32)`. If one convert `a2` using `a2.numpy()` to a numpy object, the ourcome will be `array([[0.8]], dtype = float32)`.

### Building a Neural Network:
Earlier we saw how to create a neural net. Here is another way:

```python
layer_1 = Dense(units = 3, activation "sigmoid")
layer_2 = Dense(units = 1, activation "sigmoid")
model = Sequential([layer_1,layer_2]) # This is asking TF to create a neural net from the two layers created sequentially
```

This can also be achieved using the following line, rather than having explicit assignments:

```python
model = Sequential([Dense(units = 3, activation ="sigmoid"),
                    Dense(units = 1, activation ="sigmoid")])
model.compile(...)
```

suppose you have a training set as below:

|     	|    	| y 	|
|-----	|----	|---	|
| 200 	| 17 	| 1 	|
| 120 	| 5  	| 0 	|
| 425 	| 20 	| 0 	|
| 212 	| 18 	| 1 	|

in this model, the trainig set will be:
```python
x = np.array([[200.0,17.0],
              [120.0,5.0],
              [425.0,20.0],
              [212.0,18.0]])
 y = np.array([1,0,0,1])
```

And we can create a model using the lines below:
```python
model.compule(...) # sequentially put together the two layers
model.fit(x,y) #fitting the training
model.predict(x_new) #carries the inference for you
```

### Neural Net Implementation in Python:

The idea is to review how the code is written in `Python` using the coffee roasting model with two layers (3 nodes and 1 node). 
Essentially, the algebraic model will be:
```python
X = np.array([200,17]) # which is 200F for 17minutes is the input. 
```

The node activations will be calculated using the following template function
```math
a_1^{[1]} = g(\overrightarrow{w}_1^{[1]} \cdot \overrightarrow{x} + b_1^{[1]})
```
```python
#calculating the first layer first node
w1_1 = np.array([1,2])
b1_1 = np.array([-1])
z1_1=np.dot(w1_1,x)+b1_1
a1_1= sigmoid(z1_1)
#calculating the first layer second node
w1_2 = np.array([-3,4])
b1_2 = np.array([1])
z1_2=np.dot(w1_2,x)+b1_2
a1_2= sigmoid(z1_2)
#calculating the first layer third node
w1_3 = np.array([5,-6])
b1_3 = np.array([2])
z1_3=np.dot(w1_3,x)+b1_3
a1_3= sigmoid(z1_3)

#grouping them together:
a1 = np.array([a1_1,a1_2,a1_3]


#calculating the second layer first node
w2_1 = np.array([-7,8,9])
b2_1 = np.array([3])
z2_1=np.dot(w2_1,x)+b2_1
a2_1= sigmoid(z2_1)
```
Given the following w arrays:
```math
\overrightarrow{w}_1^{[1]} = \begin{bmatrix} 1 \\\ 2 \end{bmatrix}   \text{  and  }
\overrightarrow{w}_2^{[1]} = \begin{bmatrix} -3 \\\ 4 \end{bmatrix}   \text{  and  }
\overrightarrow{w}_3^{[1]} = \begin{bmatrix} 5 \\\ -6 \end{bmatrix}   \text{  and  }
   \text{
and
  }
\overrightarrow{a}^{[0]} = \overrightarrow{X}

```
Then the python formatted arrays should be:
```python
W = np.array[[1,-3,5],[2,4,-6]] # first row will be the first component of each w and second row is the second component
b = np.array([-1,1,2])
a_in = np.array([-2,4])


def dense (a_in,W,b)
    """
    Computes dense layer
    Args:
      a_in (ndarray (n, )) : Data, 1 example 
      W    (ndarray (n,j)) : Weight matrix, n features per unit, j units
      b    (ndarray (j, )) : bias vector, j units  
    Returns
      a_out (ndarray (j,))  : j units|
    """

  units = W.shape[1]
  a_out = np.zeros(units)
  for j in range(units):
    w=W[:,j]  #[pulling j th column
    z = np.dot(w,a_in)+b[j]
    a_out [j] = g(z)
  return a_out

def sequential(x):
  a_1 = dense(x,W1,b1)
  a_2 = dense(a_1,W2,b2)
  a_3 = dense(a_2,W3,b3)
  a_4 = dense(a_3,W4,b4)
  f_x = a4
return f_x
```
Prediction function will work as follows:
```python
def my_predict(X, W1, b1, W2, b2):
    m = X.shape[0]
    p = np.zeros((m,1))
    for i in range(m):
        p[i,0] = my_sequential(X[i], W1, b1, W2, b2)
    return(p)

X_tst = np.array([
    [200,13.9],  # postive example
    [200,17]])   # negative example
X_tstn = norm_l(X_tst)  # remember to normalize
predictions = my_predict(X_tstn, W1_tmp, b1_tmp, W2_tmp, b2_tmp)

yhat = np.zeros_like(predictions)
for i in range(len(predictions)):
    if predictions[i] >= 0.5:
        yhat[i] = 1
    else:
        yhat[i] = 0
print(f"decisions = \n{yhat}")

yhat = (predictions >= 0.5).astype(int)
print(f"decisions = \n{yhat}")
```

### Neural Net Vectorization OR Efficient Implementation
To efficiently implement or code for neural network, one could either optimize the algorithms or (maybe and!) optimize the way the code is written. Notice the code reflectd before using x, w, and b, followed by dense function definition and use of sigmoid function, can be rewritten as follows:
```python
X = np.array([[]])
W = np.array([[],[]])
B = np.array([[]])
#the lines above created 2D arrays.
def dense(A_in,W, B):
  Z=np.matmul(A_in,W)+b #matmul performs matrix multiplication function
  A_out = g(Z)
  return A_out
```

## Course 2 Week 2:
### Neural Net Training: TF implementation
The typical example for this trainign is to train on a set of handwriten 0s and 1s. 
Last week the following script was written to handle the 3 layer neural network (NN from now on). 
```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
model = Sequential([
                    Dense(units=25, activation = 'sigmoid'),
                    Dense(units=15, activation = 'sigmoid'),
                    Dense(units=1, activation = 'sigmoid'),
                    ])
# to compile the model and find the loss function:
from tensorflow.keras.losses import BinaryCrossentropy
model.compile(loss=BinaryCrossentropy())
# and now fit the model
model.fit(X,Y,epochs = 100)
```
Here we want to initially establish the details behind the training of the model.

Very similar to the logistic regression training, where we wanted to find the proper coefficients for the wx+b equation to estimate/predict the data trend, here we follow the same steps and then find the loss and the cost function for the training set. Lastly we tried to minimize the cost function. 

Step 1: Specify how to compute the output for a given input: 
```python
model = Sequential ([Dense(...)], Dense(...),...,Dense(...))
```
Step 2: create the model and the loss function to use: 
the equation for the binary cross entropy function isis:
```math
L(f(\vv{x}),y) = - y \log{f(\vv{x})} - (1-y)\log{(1-f(\vv{x}))}
```
```python
model.compile(loss=BinaryCrossentropy()) 
```
Not that the Binary function is being used for a classification problem. If the problem was a regression problem, we could use Mean Squared Error function instead. 

Step 3: Specify how to minimize the cost function: 
```python
model.fit(X,y, epochs = 100) # uses the gradient descent to compute the minimum value for w and b. 
```

### Neural Net Training: Activation Functions:

So far all examples included using sigmoid function for the activation function. However, if we are doing some variation of categorization (e.g. multiple ways to categorize that shows in sruverys). One option is Rectified Linear Unit (ReLU). Another option is to just linear activation function. 
The choice of the activation function is highly depends on what sort of output we are expecting. Example could be the case of being able to have negative output or not, where ReLU can play a clear role. 

### Multiclass Classification:
The previous examples of detecting the 0 and 1 is the simplest form of categorization, meaning the algorithm needs to understand if the input is 0 or 1. Expanding the same example to detect all digits from 0 to 9, will open the field to new problems and requires new modalities to handle multiclass classification problem. 

Recall that Logestic Regression is used for 2 possible output values and we used sigmoid function for calculating the possibility. **Softmax** Regression is the answer o multiple output possibilities. Here is an example for 4 output conditions:
```math
z_1 = \overrightarrow{w}_1 \cdot \overrightarrow{x}_1 + b_1

z_2 = \overrightarrow{w}_2 \cdot \overrightarrow{x}_2 + b_2

z_3 = \overrightarrow{w}_3 \cdot \overrightarrow{x}_3 + b_3

z_4 = \overrightarrow{w}_4 \cdot \overrightarrow{x}_4 + b_4
```

And the Activations will be:
```math
a_1 = \frac{e^{z_1}}{e^{z_1}+e^{z_2}+e^{z_3}+e^{z_4}} = P(y=1|\overrightarrow{x})
```
and similarly a_2, a_3, and a_4 can be calculated. ** Remember that the sum of probabilities will stay 1 as usual. **

As such the Softmax Regression for N possible outputs calculates using the following:
``` math
\displaylines{z_j = \overrightarrow{w}_j \cdot \overrightarrow{x}_j + b_j \text{     where    } j = 1,2,3,...,N
\\
and:
\\
a_j = \frac{e^{z_j}}{\Sigma_{k=1}^{N}{e^{z_k}} = P(y=j|\overrightarrow{x})}
}
```
Now the cost function should be, similar to the logistic Regression:
```math
loss(a_1,a_2,...,a_N,y) = \left\{\begin{matrix} -log(a_1)  \text{    if    } y=1
\\ -log(a_2)  \text{    if    } y=2
\\ \vdots 
\\ -log(a_N)  \text{    if    } y=N
\end{matrix}\right.
```
#### Implementation of Softmax:
Previously, for the handwritten classification, the archtecture of the Neural Net ended in 1 neuron last layer. In the softmax categorization, the last layer is changed to N neuron, depending on the number of categories, which is called the Softmax layer. Notice that a_i is a function of z_i and other z_j s that are i not equal j, unlike the previous categorization that a_i was only a function of z_i. 

Here is how the code implementation would look like for Softmax:

```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Specify the model (f_w,b (x) =?)
model = Sequential([
        Dense(unit = 25, activation = 'relu',
        Dense(unit = 15, activation = 'relu',
        Dense(unit = 10, activation = 'linear', 
])
##Note: instead of linear in the last activation, we originally used "softmax," however, if we use linear and add "from_logits=True" in the complile section, TF will rearrange terms to form a more numerically accurate model. 
# Specify loss and cost
from tensorflow.keras.losses import SparseCategoricalCrossEntropy

model.compile(loss = SparseCategoricalCrossEntropy(from_logits=True) )

# Train on Data to minimize J
model.fit(X,Y,epoch = 100)

#Predict
logit = model(X)
f_x = tf.nn.softmax(logit)
```
Note that the same idea regarding the numerical stability also applies to the Logistic Regression models too. 


Here is a practical example:

```python
def my_softmax(z):
    ez = np.exp(z)              #element-wise exponenial
    sm = ez/np.sum(ez)
    return(sm)
# make  dataset for example
centers = [[-5, 2], [-2, -2], [1, 2], [5, -2]]
X_train, y_train = make_blobs(n_samples=2000, centers=centers, cluster_std=1.0,random_state=30)

#Creating the model
model = Sequential(
    [ 
        Dense(25, activation = 'relu'),
        Dense(15, activation = 'relu'),
        Dense(4, activation = 'softmax')    # < softmax activation here
    ]
)
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(0.001),
)

model.fit(
    X_train,y_train,
    epochs=10
)
#Predict
p_nonpreferred = model.predict(X_train)
print(p_nonpreferred [:2])
print("largest value", np.max(p_nonpreferred), "smallest value", np.min(p_nonpreferred))
```

However, there is a preferred modeling way:

```python
preferred_model = Sequential(
    [ 
        Dense(25, activation = 'relu'),
        Dense(15, activation = 'relu'),
        Dense(4, activation = 'linear')   #<-- Note
    ]
)
preferred_model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  #<-- Note
    optimizer=tf.keras.optimizers.Adam(0.001),
)

preferred_model.fit(
    X_train,y_train,
    epochs=10
)      
```
NOTE: The output predictions are not probabilities! If the desired output are probabilities, the output should be be processed by a softmax.

### Advanced optimization:
Previously we reviewed the gradient descent for improving the model. Depending on the needs of the model, some times the approach needs to be modified and the rate of optimization needs to be adjusted. The procedure to do so is through, **Adam (Adaptive Movement estimation**.The algorithm uses different values for $\alpha$ for each $w_j$ to modifiy the rate at which optimization is happening and result in faster more accurate outcome. Belwo is how it is written as acode:

```python
model = Sequential([
                    tf.keras.layers.Dense(units=25,activation='sigmoid',
                    tf.keras.layers.Dense(units=15,activation='sigmoid',
                    tf.keras.layers.Dense(units=10,activation='linear',
                    ])
model.compilte(optimization=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)) #initial learning rate of 1e-3

model.fit(X,Y,epochs=100)
```

#### Convolutional Layer
If each neuran instead of looking at one and only one specific neuron, they could look at a region, made up of multiple pixels, called convolution layer. This makes the creation of the model faster. 

## Week 3: Debugging a Learning Algorithm
The aim is to find out ways to carry on diagnostics to find out how we can improve the model or find the issues with the model. 

1- Evaluating the model: this is the defaul break down of the training dataset into "Trainig" set and the "Test" set. 70-30 or 80-20 would be a typical split ratio. This way we can calculate the Error of the Test sample using the actual data. 
As such, now we have two Error functions, $J_{test}$ and $J_{train}$ to evaluate our model. So, if $J_{test} >> J_{train}$ then we know that the model overfit on the training dataset and can be thrown off when faced with the test dataset.

2- The other way is to measure the fraction of test and train that the algorithm mis-classified. 

The next question is how to choose the model? how do one chooses the degree of the complexity of the model? That is where we use **Cross Validation** where we split the dataset to three datasets: Training (~60%), cross validation or CV set or development set or dev set (~20%), and test (~20%). 

Scikit-learn provides a [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) function to split your data into the parts mentioned above. In the code cell below, you will split the entire dataset into 60% training, 20% cross validation, and 20% test.

Side Note:
#### Feature scaling

In the previous course of this specialization, you saw that it is usually a good idea to perform feature scaling to help your model converge faster. This is especially true if your input features have widely different ranges of values. Later in this lab, you will be adding polynomial terms so your input features will indeed have different ranges. For example, $x$ runs from around 1600 to 3600, while $x^2$ will run from 2.56 million to 12.96 million. 

You will only use $x$ for this first model but it's good to practice feature scaling now so you can apply it later. For that, you will use the [`StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) class from scikit-learn. This computes the z-score of your inputs. As a refresher, the z-score is given by the equation:

$$ z = \frac{x - \mu}{\sigma} $$

where $\mu$ is the mean of the feature values and $\sigma$ is the standard deviation. The code below shows how to prepare the training set using the said class. You can plot the results again to inspect if it still follows the same pattern as before. The new graph should have a reduced range of values for `x`.

To evaluate the performance of your model, you will measure the error for the training and cross validation sets. For the training error, recall the equation for calculating the mean squared error (MSE):

```math
J_{train}(\vec{w}, b) = \frac{1}{2m_{train}}\left[\sum_{i=1}^{m_{train}}(f_{\vec{w},b}(\vec{x}_{train}^{(i)}) - y_{train}^{(i)})^2\right]
```

Scikit-learn also has a built-in [`mean_squared_error()`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html) function that you can use. Take note though that [as per the documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-error), scikit-learn's implementation only divides by `m` and not `2*m`, where `m` is the number of examples. As mentioned in Course 1 of this Specialization (cost function lectures), dividing by `2m` is a convention we will follow but the calculations should still work whether or not you include it. Thus, to match the equation above, you can use the scikit-learn function then divide by 2 as shown below. We also included a for-loop implementation so you can check that it's equal. 

Another thing to take note: Since you trained the model on scaled values (i.e. using the z-score), you should also feed in the scaled training set instead of its raw values.

You will scale the cross validation set below by using the same `StandardScaler` you used earlier but only calling its [`transform()`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler.transform) method instead of [`fit_transform()`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler.fit_transform).

First, you will generate the polynomial features from your training set. The code below demonstrates how to do this using the [`PolynomialFeatures`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html) class. It will create a new input feature which has the squared values of the input `x` (i.e. degree=2).

Here is an example of running the models and debugging:

```python
 for array computations and loading data
import numpy as np

# for building linear regression models and preparing data
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# for building and training neural networks
import tensorflow as tf

# custom functions
import utils

#Regression model building:
# Load the dataset from the text file
data = np.loadtxt('./data/data_w3_ex1.csv', delimiter=',')

# Split the inputs and outputs into separate arrays
x = data[:,0]
y = data[:,1]

# Convert 1-D arrays into 2-D because the commands later will require it
x = np.expand_dims(x, axis=1)
y = np.expand_dims(y, axis=1)

print(f"the shape of the inputs x is: {x.shape}")
print(f"the shape of the targets y is: {y.shape}")

# Initialize the class
scaler_linear = StandardScaler()

# Compute the mean and standard deviation of the training set then transform it
X_train_scaled = scaler_linear.fit_transform(x_train)

print(f"Computed mean of the training set: {scaler_linear.mean_.squeeze():.2f}")
print(f"Computed standard deviation of the training set: {scaler_linear.scale_.squeeze():.2f}")

# Initialize the class
linear_model = LinearRegression()

# Train the model
linear_model.fit(X_train_scaled, y_train )
# Feed the scaled training set and get the predictions
yhat = linear_model.predict(X_train_scaled)

# Use scikit-learn's utility function and divide by 2
print(f"training MSE (using sklearn function): {mean_squared_error(y_train, yhat) / 2}")

# for-loop implementation
total_squared_error = 0

for i in range(len(yhat)):
    squared_error_i  = (yhat[i] - y_train[i])**2
    total_squared_error += squared_error_i                                              

mse = total_squared_error / (2*len(yhat))

print(f"training MSE (for-loop implementation): {mse.squeeze()}")

# Scale the cross validation set using the mean and standard deviation of the training set
X_cv_scaled = scaler_linear.transform(x_cv)

print(f"Mean used to scale the CV set: {scaler_linear.mean_.squeeze():.2f}")
print(f"Standard deviation used to scale the CV set: {scaler_linear.scale_.squeeze():.2f}")

# Feed the scaled cross validation set
yhat = linear_model.predict(X_cv_scaled)

# Use scikit-learn's utility function and divide by 2
print(f"Cross validation MSE: {mean_squared_error(y_cv, yhat) / 2}")

# Instantiate the class
scaler_poly = StandardScaler()

# Compute the mean and standard deviation of the training set then transform it
X_train_mapped_scaled = scaler_poly.fit_transform(X_train_mapped)

# Preview the first 5 elements of the scaled training set.
print(X_train_mapped_scaled[:5])

# Initialize the class
model = LinearRegression()

# Train the model
model.fit(X_train_mapped_scaled, y_train )

# Compute the training MSE
yhat = model.predict(X_train_mapped_scaled)
print(f"Training MSE: {mean_squared_error(y_train, yhat) / 2}")

# Add the polynomial features to the cross validation set
X_cv_mapped = poly.transform(x_cv)

# Scale the cross validation set using the mean and standard deviation of the training set
X_cv_mapped_scaled = scaler_poly.transform(X_cv_mapped)

# Compute the cross validation MSE
yhat = model.predict(X_cv_mapped_scaled)
print(f"Cross validation MSE: {mean_squared_error(y_cv, yhat) / 2}")

```

### High Variance vs High Bias?
The problem of finding which one is the source will help finding out a better strategy to resolve the issue. Higher Bias means that the model is not properly capturing the features of the dataset and need to implement more complexities (higher orders or more date). On the other hand High Variance means that the model is capturing significantly more than needed details and to resolve the issue we can reduce the size of sample set or reduce higher order terms from the model making the model simpler. 

![Andrew Ng's slide on Bias and Variance](https://github.com/farsoroush/mLNotes/blob/aede484acfe92cf37731299164977977992d39aa/Bias%20vs%20Variance_MLNotes.png)

However, if you train your large neural network, you must be aware that they are **low bias machines"". 

![Large Neural Networks Bias](https://github.com/farsoroush/mLNotes/blob/897c9fde0d83850173c2ba1f6de6aaa656a93fdb/LNN%20models%20bias.png)


## Machine Learning Model Development:
1- Choose Architecture: model, data, etc.
2- Train model
3- Diagnostics
4- repeat!

### Error Analysis:
Error analysis is essentially going through the errors made when the training dataset was used to train the model and the outcome was not properly categorized. 
Some times after analyzing the data, we find out the error is coming from the way the model comprehended the data and maybe categorized it. As a result, we may find the need to add more data. One of the solutions is **Data Augmentation** which is essentially taking the same dataset, and then modify or change the same set and create new examples. Simple examples for training an OCR algorithm would be: rotating, enlarging, changing the intensity of images, warp/distort the images, or flipping the images; still same data is used. 

Another example is data synthesis, which is creating a new dataset and essentially you would write a code to create the training the algorithm. This is more like a data engineering. 

Distinction: 
model-centric approach where the focus is more on improving the model and introduce new datasets. 
Data-centric approach where the training dataset is transformed and engineered to create larger training sets. 

### Transfer Learning:
Transfering the w and b matrix from an initially trained model (on a large dataset) to a new model or when adding new applications. Similar to GPT model, where it was trained on a significantly large dataset and then fine tuned with a smalled traing set. 

### Full cycle of ML modeling:

