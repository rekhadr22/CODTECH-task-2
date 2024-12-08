# CODTECH-task-2
## **NAME:Rekha D**
## **Company:CODTECH IT SOLUTIONS**
## **ID:CTO8DS9846**
## **Domain:DATA ANALYTICS**
## **Duration:10th Nov 2024 to 10th Dec 2024**
## **Mentor name:  **

## **OVERVIEW OF THE PROJECT**

  ### Task 1: House Price Prediction using Linear Regression

## **Objective**:

The objective of this project is to develop a **Linear Regression** model to predict house prices based on various features such as the number of rooms, crime rates, proximity to highways, and more. The project also involves evaluating the model's performance and interpreting the results.

## **Key Activities**:

1. **Dataset Collection and Exploration**:
   - Loaded the **Boston Housing Dataset** and explored the features such as crime rate, average number of rooms, and property tax rate.
   - Analyzed the dataset to understand the relationships between features and house prices.

2. **Data Preprocessing**:
   - Handled missing data by removing or imputing missing values.
   - Split the data into **training** and **testing** sets.
   - Scaled the features if necessary to normalize the data for better model performance.

3. **Model Development**:
   - Built a **Linear Regression** model using **Scikit-learn**.
   - Trained the model on the training dataset to learn the relationship between the features and house prices.

4. **Model Evaluation**:
   - Evaluated the model using metrics such as **Mean Squared Error (MSE)** and **R-squared (R²)** to assess the model's prediction accuracy.
   - Analyzed residuals and other evaluation metrics to determine model performance.

5. **Model Interpretation**:
   - Interpreted the model's coefficients to identify which features had the most significant impact on predicting house prices.
   - Used **matplotlib** and **seaborn** to visualize the results and evaluate the model's assumptions.

6. **Prediction**:
   - Used the trained model to predict house prices on new data points (new houses).
   - 
## **Technologies Used**:

- **Programming Language**: Python
- **Libraries**:
  - **Pandas**: Data manipulation and exploration
  - **NumPy**: Numerical operations
  - **Scikit-learn**: Machine learning models, data splitting, and evaluation
  - **Matplotlib** & **Seaborn**: Data visualization
  - **Statsmodels** (optional): Statistical analysis and model interpretation
- **Development Tools**:
  - **Jupyter Notebook** / **Google Colab**: Interactive development and visualization
  - **VS Code** / **PyCharm**: For structured code development

---

## **Dataset**:

The project uses the **Boston Housing Dataset** (available in Scikit-learn) with features like:

- `crim`: Crime rate per capita
- `zn`: Proportion of residential land zoned for large lots
- `indus`: Proportion of industrial land
- `chas`: Binary feature indicating proximity to the Charles River
- `nox`: Nitrogen oxide concentration
- `rm`: Average number of rooms per dwelling
- `age`: Proportion of houses built before 1940
- `dis`: Distance to employment centers
- `rad`: Accessibility to highways
- `tax`: Property tax rate
- `ptratio`: Pupil-teacher ratio
- `b`: Proportion of African American residents
- `lstat`: Percentage of lower status population
- `medv`: Median value of owner-occupied homes (target variable)

---

## **Model Evaluation**:

- **Mean Squared Error (MSE)**: A metric for how far the predicted house prices are from the actual values.
- **R-squared (R²)**: A measure of how well the model explains the variance in the house prices.
Model Evaluation:
Mean Squared Error (MSE): 46.176936738355245
R-squared (R²): 0.3703183614923722

## OUTPUT:
-**Model Evaluation:**
-**Mean Squared Error (MSE): 46.176936738355245**
**R-squared (R²): 0.3703183614923722**

**Model Coefficients:**
**Intercept: -36.28556028565263**
**Coefficient for 'rm': 9.353307193414203**
**Predicted Median Home Value for 6.5 rooms: $24510.94**

​
