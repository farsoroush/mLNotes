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
