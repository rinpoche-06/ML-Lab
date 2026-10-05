# ML-Lab
## Contents
 
| Folder | Task | Type |
|---|---|---|
| [auto-mpg](./auto-mpg) | Predict fuel efficiency | Regression |
| [california-housing](./california-housing) | Predict house prices | Regression |
| [decison-tree](./decison-tree) | Decision tree classifier | Classification |
| [knn](./knn) | K-Nearest Neighbors | Classification |
| [linear-svm](./linear-svm) | Linear SVM | Classification |
| [naive-bayes](./naive-bayes) | Naive Bayes classifier | Classification |
| [pima-dataset](./pima-dataset) | Diabetes prediction | Classification |
| [ridge-lasso-diabetes](./ridge-lasso-diabetes) | Regularized regression | Regression |
| [svm-kernels](./svm-kernels) | SVM kernel comparison | Classification |
 
---
 
## auto-mpg
 
Predicts miles per gallon from engine size, horsepower, weight, and model year.
 
- Regression problem; the target is a continuous number
- Contains missing values (horsepower), which need handling before modeling
- Heavier cars with larger engines generally get worse mileage
- Evaluated with MSE, RMSE, and R²
  
## california-housing
 
Predicts median house prices for California districts.
 
- Regression on a larger dataset than the other experiments
- Median income is the strongest predictor of price
- Latitude and longitude carry meaningful location signal
- Evaluated with R² and RMSE
## decison-tree
 
Decision tree that splits data through successive feature-based conditions.
 
- Splits chosen using Gini impurity or entropy
- Easy to visualize and interpret
- Overfits if depth is unrestricted; depth limits and pruning control this
- Folder name contains a typo. Left as is.
## knn
 
Classifies a point by majority vote among its K nearest neighbors.
 
- No training phase; the model stores the data (lazy learner)
- Value of K determines the result: small K is noisy, large K oversmooths
- Distance-based, so feature scaling is required
- Slow on large datasets because every prediction compares against all stored points
## linear-svm
 
Support Vector Machine that finds the hyperplane separating classes with the maximum margin.
 
- Maximizes the margin between classes
- Only support vectors determine the boundary
- Parameter C controls the tradeoff between margin width and misclassification
- Suited to data that is linearly separable or close to it
## naive-bayes
 
Probabilistic classifier based on Bayes' theorem.
 
- Selects the class with the highest computed probability
- Assumes features are independent of each other, which is rarely true in practice
- Fast, needs little data, and serves as a baseline
- Gaussian variant is used for continuous features
## pima-dataset
 
Predicts diabetes from glucose, BMI, age, blood pressure, and related measurements.
 
- Binary classification on the Pima Indians Diabetes dataset
- Several zero values are actually missing data and need to be treated as such
- Imputation of these values changes model performance
- Glucose is typically the most influential feature
## ridge-lasso-diabetes
 
Compares Ridge and Lasso regression on the diabetes dataset.
 
- Ridge (L2) shrinks coefficients but retains all features
- Lasso (L1) can reduce coefficients to zero, performing feature selection
- Alpha sets the strength of the penalty
- Results compared against ordinary linear regression
## svm-kernels
 
SVM with linear, polynomial, and RBF kernels for data that is not linearly separable.
 
- Kernel trick maps data to a higher dimension without computing the transformation explicitly
- Polynomial kernel produces curved boundaries; RBF produces flexible, localized ones
- C and gamma determine model behavior and require tuning
- Kernels compared on the same data to see which performs best
 
