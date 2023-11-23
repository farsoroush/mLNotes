# Machine Learning Notes

## Disclosure:
The notes here are personal notes on various machine learning courses (ML courses) and can be used by others. If you find any mistakes in understanding or in writing (or in any of the content) please feel free to comment or contact me via serpoush [at] gmail. 

## Course 1: Machine Learning Fundamentals by Andrew Ng
### Week 1 Section 1 
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


### Cost Function

This function is our measure for checking how well our predictions are. The aim in the regression is to get a function that gets predictions as close as possible to the targets. Cost function measures the **Mean Squared Errors** or 

$$J(w,b) = \frac{1}{2m} \sum_{i=1}^m (\hat{y}^{(i)} - y^{(i)})^2 = \frac{1}{2m} \sum_{i=1}^m (f_{w,b}(x^{(i)}) - y^{(i)})^2$$

Where parameters, w and b in the case of the linear regression example, can be adjusted to improve the model. So the goal is to minimize the function $J(w,b)$. Simplified model can use a $b=0$ to only play with $w$.

