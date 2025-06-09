# Wine Classification with Min-Max Scaling

This project demonstrates the application of **Min-Max Scaling**, a common data normalization technique, on the Wine dataset. The goal was to preprocess the features (**Alcohol** and **Malic acid**) to improve the performance of machine learning models by ensuring that all features contribute equally, regardless of their original scale.

## Project Overview

Here’s a breakdown of what I did in this project:

### 1. Data Loading and Exploration

I began by loading the Wine dataset and exploring its features — particularly focusing on **Alcohol** and **Malic acid**. I used descriptive statistics and visualizations to understand their distributions.

### 2. Data Splitting

I split the dataset into training and testing sets to ensure that the impact of scaling could be fairly evaluated on unseen data.

### 3. Min-Max Scaling Implementation

I applied **Min-Max Scaling** using `MinMaxScaler` from `sklearn.preprocessing`. This technique transforms each feature individually such that it is in the range [0, 1].

### 4. Impact Visualization

To understand the effects of scaling, I visualized the transformed data using **KDE plots** and **scatter plots**, showing both before and after comparisons.

## Key Findings

- **Normalization Effect:** Min-Max Scaling successfully scaled the features to the [0, 1] range. The `describe()` output and KDE plots confirmed this, showing the minimum and maximum values as 0.0 and 1.0 respectively.

- **Distribution Preservation:** One important insight was that Min-Max Scaling preserved the original distribution shapes of the features. It did not distort the data — just rescaled it.

- **Feature Relationship Preservation:** The scatter plots showed that the **relative positions and relationships** between data points (especially in relation to the Class label) were maintained after scaling. This is important for many ML algorithms.

- **Benefits for Models:** Scaling helped ensure that no feature dominated others simply due to its magnitude. This is especially beneficial for models like **K-Nearest Neighbors (KNN)**, **Support Vector Machines (SVM)**, and those using **Gradient Descent**.

## Tools and Libraries Used

- Python  
- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  

## Next Steps

In the future, I could explore the effects of other normalization techniques like **Z-score scaling**, **RobustScaler**, or even apply this to more complex datasets with more features and classes.
