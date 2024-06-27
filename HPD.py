import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.metrics import r2_score

#train_path = "C:\Users\jorda\OneDrive\Desktop\Nxt24\House price detection\CSV\train.csv"
#test_path = "C:\Users\jorda\OneDrive\Desktop\Nxt24\House price detection\CSV\test.csv"

train_path = 'C:/Users/jorda/OneDrive/Desktop/Nxt24/1 House price detection/CSV/train.csv'
test_path = 'C:/Users/jorda/OneDrive/Desktop/Nxt24/1 House price detection/CSV/test.csv'

# Load the CSV files into DataFrames
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# Display the first few rows of the dataframes
print(df_train.head())
print(df_test.head())



pd.set_option('display.max_rows', None)

# Calculate and display the percentage of missing values in each column of train_df
missing_values_percentage = df_train.isnull().sum().sort_values(ascending=False) * 100 / len(df_train)
print("\nPercentage of missing values in each column of train_df:")
print(missing_values_percentage)

pd.set_option('display.max_rows', None) 
# display maximum number of rows
df_train.isnull().sum().sort_values(ascending=False)*100/len(df_train)


# Print the column names of the DataFrame
print("Columns in train_df:")
print(df_train.columns.tolist())

print("\nColumns in test_df:")
print(df_test.columns.tolist())

# To drop columns with more than 60% missing values
threshold = 0.6
df_train_clean = df_train.loc[:, df_train.isnull().mean() <= threshold]
df_test_clean = df_test.loc[:, df_test.isnull().mean() <= threshold]

# To plot heatmap of missing values for the train DataFrame
plt.figure(figsize=(12, 8))
sns.heatmap(df_train.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap - Train DataFrame')
plt.show()

# To plot heatmap of missing values for the test DataFrame
plt.figure(figsize=(12, 8))
sns.heatmap(df_test.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap - Test DataFrame')
plt.show()

# To check missing values in training data and display the percentage of missing values
missing_values_percentage_train = df_train.isnull().sum().sort_values(ascending=False) * 100 / len(df_train)
print("\nPercentage of missing values in each column of train data:")
print(missing_values_percentage_train)

# To check missing values in testing data and display the percentage of missing values
missing_values_percentage_test = df_test.isnull().sum().sort_values(ascending=False) * 100 / len(df_test)
print("\nPercentage of missing values in each column of test data:")
print(missing_values_percentage_test)

# To re-check missing values in training data
missing_values_train = df_train.isnull().sum()
print("\nMissing values in each column of train data:")
print(missing_values_train)

# To re-check missing values in testing data
missing_values_test = df_test.isnull().sum()
print("\nMissing values in each column of test data:")
print(missing_values_test)

# Re-checking the dataframes
df_train.info()
df_test.info()

# Get concise information about the training DataFrame
print("Concise information about train_df:")
print(df_train.info())

# Get concise information about the testing DataFrame
print("\nConcise information about test_df:")
print(df_test.info())

# Concatenate the training and testing data to ensure consistent label encoding
combined_data = pd.concat([df_train, df_test], axis=0)

# Loop through each column in the combined data
for column in combined_data.columns:
    # Check if the column contains categorical data
    if combined_data[column].dtype == 'object':
        # Initialize LabelEncoder
        label_encoder = LabelEncoder()
        # Fit LabelEncoder on combined data to ensure consistency
        label_encoder.fit(combined_data[column].astype(str))
        # Transform both training and testing data
        df_train[column] = label_encoder.transform(df_train[column].astype(str))
        df_test[column] = label_encoder.transform(df_test[column].astype(str))

# Let's print the first few rows of the encoded data to verify the transformation
print("Encoded Training Data:")
print(df_train.head())
print("\nEncoded Testing Data:")
print(df_test.head())

# The shape of the training DataFrame
print("Shape of train_df:", df_train.shape)

# The shape of the testing DataFrame
print("Shape of test_df:", df_test.shape)

# To remove duplicated rows
df_train.drop_duplicates(inplace=True)
df_test.drop_duplicates(inplace=True)

# To remove duplicated columns
df_train = df_train.loc[:, ~df_train.columns.duplicated()]
df_test = df_test.loc[:, ~df_test.columns.duplicated()]

# To remove duplicate indices from the training DataFrame
df_train = df_train[~df_train.index.duplicated()]

# To remove duplicate indices from the testing DataFrame
df_test = df_test[~df_test.index.duplicated()]



# To identify categorical columns in the training DataFrame
df_train = pd.read_csv(train_path)
categorical_cols_train = df_train.select_dtypes(include=['object']).columns.tolist()
print("Categorical columns in the training DataFrame:")
print(categorical_cols_train)

# List of categorical columns
categorical_cols = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']

# Iterate through each categorical column and print unique values
for col in categorical_cols:
    unique_values = df_train[col].unique()
    print(f"Unique values in {col}: {unique_values}")
 

# Function for imputing categorical missing data
def impute_categorical_missing_data(df, passed_col, bool_cols, missing_data_cols):
    # Separate rows with missing values and without missing values
    df_null = df[df[passed_col].isnull()]
    df_not_null = df[df[passed_col].notnull()]

    X = df_not_null.drop(passed_col, axis=1)
    y = df_not_null[passed_col]
    
    other_missing_cols = [col for col in missing_data_cols if col != passed_col]
    
    label_encoder = LabelEncoder()

    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            X[col] = label_encoder.fit_transform(X[col])

    if passed_col in bool_cols:
        y = label_encoder.fit_transform(y)
        
    iterative_imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=42), add_indicator=True)

    for col in other_missing_cols:
        if X[col].isnull().sum() > 0:
            col_with_missing_values = X[col].values.reshape(-1, 1)
            imputed_values = iterative_imputer.fit_transform(col_with_missing_values)
            X[col] = imputed_values[:, 0]
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train RandomForestRegressor
    rf_classifier = RandomForestRegressor()
    rf_classifier.fit(X_train, y_train)

    # Predict missing values
    y_pred = rf_classifier.predict(X_test)

    # Calculate accuracy
    acc_score = r2_score(y_test, y_pred)

    print("The feature '"+ passed_col+ "' has been imputed with", round((acc_score * 100), 2), "accuracy\n")

    # Impute missing values in the original DataFrame
    X = df_null.drop(passed_col, axis=1)

    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            X[col] = label_encoder.fit_transform(X[col])

    for col in other_missing_cols:
        if X[col].isnull().sum() > 0:
            col_with_missing_values = X[col].values.reshape(-1, 1)
            imputed_values = iterative_imputer.fit_transform(col_with_missing_values)
            X[col] = imputed_values[:, 0]
                
    if len(df_null) > 0: 
        df_null[passed_col] = rf_classifier.predict(X)
        if passed_col in bool_cols:
            df_null[passed_col] = df_null[passed_col].map({0: False, 1: True})
    else:
        pass

    df_combined = pd.concat([df_not_null, df_null])
    
    return df_combined[passed_col]

# Function for imputing continuous missing data
def impute_continuous_missing_data(df, passed_col, missing_data_cols):
    df_null = df[df[passed_col].isnull()]
    df_not_null = df[df[passed_col].notnull()]

    X = df_not_null.drop(passed_col, axis=1)
    y = df_not_null[passed_col]
    
    other_missing_cols = [col for col in missing_data_cols if col != passed_col]
    
    label_encoder = LabelEncoder()

    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            X[col] = label_encoder.fit_transform(X[col])
    
    iterative_imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=42), add_indicator=True)

    for col in other_missing_cols:
        if X[col].isnull().sum() > 0:
            col_with_missing_values = X[col].values.reshape(-1, 1)
            imputed_values = iterative_imputer.fit_transform(col_with_missing_values)
            X[col] = imputed_values[:, 0]
    
    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train RandomForestRegressor
    rf_regressor = RandomForestRegressor()
    rf_regressor.fit(X_train, y_train)

    # Predict missing values
    y_pred = rf_regressor.predict(X_test)

    print("MAE =", mean_absolute_error(y_test, y_pred), "\n")
    print("RMSE =", mean_squared_error(y_test, y_pred, squared=False), "\n")
    print("R2 =", r2_score(y_test, y_pred), "\n")

    # Impute missing values in the original DataFrame
    X = df_null.drop(passed_col, axis=1)

    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'category':
            X[col] = label_encoder.fit_transform(X[col])

    for col in other_missing_cols:
        if X[col].isnull().sum() > 0:
            col_with_missing_values = X[col].values.reshape(-1, 1)
            imputed_values = iterative_imputer.fit_transform(col_with_missing_values)
            X[col] = imputed_values[:, 0]
                
    if len(df_null) > 0: 
        df_null[passed_col] = rf_regressor.predict(X)
    else:
        pass

    df_combined = pd.concat([df_not_null, df_null])
    
    return df_combined[passed_col]

# Filter and sort columns with missing values in the DataFrame
missing_values = df_train.isnull().sum()[df_train.isnull().sum() > 0].sort_values(ascending=False)
print(missing_values)

import warnings
warnings.filterwarnings('ignore')

# List of columns with missing data in training set
missing_data_cols_train = df_train.columns[df_train.isnull().any()].tolist()

# List of columns with missing data in test set
missing_data_cols_test = df_test.columns[df_test.isnull().any()].tolist()

# Define bool_cols to include all columns with boolean data type
bool_cols = [col for col in df_train.columns if df_train[col].dtype == 'bool']

# Ensure all categorical variables are encoded properly
label_encoder = LabelEncoder()

for col in df_train.columns:
    if df_train[col].dtype == 'object' or col in categorical_cols_train:
        df_train[col] = label_encoder.fit_transform(df_train[col].astype(str))

for col in df_test.columns:
    if df_test[col].dtype == 'object' or col in categorical_cols_train:
        df_test[col] = label_encoder.fit_transform(df_test[col].astype(str))

# Impute missing values using our functions for the training set
for col in missing_data_cols_train:
    print("Missing Values", col, ":", str(round((df_train[col].isnull().sum() / len(df_train)) * 100, 2))+"%")
    if col in categorical_cols_train:
        df_train[col] = impute_categorical_missing_data(df_train, col, bool_cols, missing_data_cols_train)
    else:
        df_train[col] = impute_continuous_missing_data(df_train, col, missing_data_cols_train)

# Impute missing values using our functions for the test set
for col in missing_data_cols_test:
    print("Missing Values", col, ":", str(round((df_test[col].isnull().sum() / len(df_test)) * 100, 2))+"%")
    if col in categorical_cols_train:  # Assuming you want to use the same categorical_cols_train for the test set
        df_test[col] = impute_categorical_missing_data(df_test, col, bool_cols, missing_data_cols_test)
    else:
        df_test[col] = impute_continuous_missing_data(df_test, col, missing_data_cols_test)

# Function to print missing values in the dataframe
def print_missing_values(df, dataset_name):
    missing_values = df.isnull().sum()[df.isnull().sum() > 0]
    if missing_values.empty:
        print(f"No missing values in {dataset_name}.")
    else:
        print(f"Missing values in {dataset_name}:\n{missing_values}\n")

# Print missing values before imputation
print("Before Imputation:")
print_missing_values(df_train, "training set")
print_missing_values(df_test, "test set")

# Impute missing values using our functions for the training set
for col in missing_data_cols_train:
    print("Imputing Missing Values for column:", col, "(", str(round((df_train[col].isnull().sum() / len(df_train)) * 100, 2))+"%)")
    if col in categorical_cols_train:
        df_train[col] = impute_categorical_missing_data(df_train, col, bool_cols, missing_data_cols_train)
    else:
        df_train[col] = impute_continuous_missing_data(df_train, col, missing_data_cols_train)

# Impute missing values using our functions for the test set
for col in missing_data_cols_test:
    print("Imputing Missing Values for column:", col, "(", str(round((df_test[col].isnull().sum() / len(df_test)) * 100, 2))+"%)")
    if col in categorical_cols_train:  # Assuming you want to use the same categorical_cols_train for the test set
        df_test[col] = impute_categorical_missing_data(df_test, col, bool_cols, missing_data_cols_test)
    else:
        df_test[col] = impute_continuous_missing_data(df_test, col, missing_data_cols_test)

# Print missing values after imputation
print("After Imputation:")
print_missing_values(df_train, "training set")
print_missing_values(df_test, "test set")


# Descriptive statistics for the training dataset
descriptive_stats_train = df_train.describe(include='all')

# Display the descriptive statistics for the training dataset
print("Descriptive Statistics for Training Dataset:")
print(descriptive_stats_train.T)  # Transpose for better readability

# Descriptive statistics for the test dataset
descriptive_stats_test = df_test.describe(include='all')

# Display the descriptive statistics for the test dataset
print("\nDescriptive Statistics for Test Dataset:")
print(descriptive_stats_test.T)  # Transpose for better readability

# Setting up the aesthetic style of the plots
sns.set(style="whitegrid")

# Plotting distributions of key numerical variables
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Distribution of SalePrice
sns.histplot(df_train['SalePrice'], bins=30, ax=axes[0], kde=True)
axes[0].set_title('Distribution of SalePrice')

# Distribution of LotArea
sns.histplot(df_train['LotArea'], bins=30, ax=axes[1], kde=True)
axes[1].set_title('Distribution of LotArea')

# Distribution of LotFrontage
sns.histplot(df_train['LotFrontage'], bins=30, ax=axes[2], kde=True)
axes[2].set_title('Distribution of LotFrontage')

plt.tight_layout()
plt.show()


# Correlation analysis
correlation_matrix_train = df_train.corr()

# Plotting the heatmap of the correlation matrix for the training dataset
plt.figure(figsize=(15, 12))
sns.heatmap(correlation_matrix_train, annot=False, cmap='coolwarm')
plt.title('Correlation Matrix of Variables in Training Dataset')
plt.show()

# Displaying correlation values with SalePrice in descending order for the training dataset
correlation_with_saleprice_train = correlation_matrix_train['SalePrice'].sort_values(ascending=False)
print("Correlation with SalePrice (Training Dataset):")
print(correlation_with_saleprice_train)


# Plotting count plots for MSSubClass and MSZoning

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Count plot for MSSubClass
sns.countplot(x='MSSubClass', data=df_train, ax=axes[0])
axes[0].set_title('Count of Properties by MSSubClass')
axes[0].set_xlabel('MSSubClass')
axes[0].set_ylabel('Count')

# Count plot for MSZoning
sns.countplot(x='MSZoning', data=df_train, ax=axes[1])
axes[1].set_title('Count of Properties by MSZoning')
axes[1].set_xlabel('MSZoning')
axes[1].set_ylabel('Count')

plt.tight_layout()
plt.show()

# Scatter plots with regression lines for OverallQual and GrLivArea against SalePrice

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Scatter plot for OverallQual vs SalePrice
sns.regplot(x='OverallQual', y='SalePrice', data=df_train, ax=axes[0], scatter_kws={'alpha':0.5})
axes[0].set_title('OverallQual vs SalePrice')
axes[0].set_xlabel('Overall Quality')
axes[0].set_ylabel('Sale Price')

# Scatter plot for GrLivArea vs SalePrice
sns.regplot(x='GrLivArea', y='SalePrice', data=df_train, ax=axes[1], scatter_kws={'alpha':0.5})
axes[1].set_title('GrLivArea vs SalePrice')
axes[1].set_xlabel('Above Grade Living Area (sq ft)')
axes[1].set_ylabel('Sale Price')

plt.tight_layout()
plt.show()


# Box plots for SalePrice, OverallQual, and GrLivArea

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Box plot for SalePrice
sns.boxplot(y=df_train['SalePrice'], ax=axes[0])
axes[0].set_title('Box Plot of SalePrice')
axes[0].set_ylabel('Sale Price')

# Box plot for OverallQual
sns.boxplot(y=df_train['OverallQual'], ax=axes[1])
axes[1].set_title('Box Plot of OverallQual')
axes[1].set_ylabel('Overall Quality')

# Box plot for GrLivArea
sns.boxplot(y=df_train['GrLivArea'], ax=axes[2])
axes[2].set_title('Box Plot of GrLivArea')
axes[2].set_ylabel('Above Grade Living Area (sq ft)')

plt.tight_layout()
plt.show()

# Preprocessing steps
numerical_cols = [col for col in df_train.columns if df_train[col].dtype in ['int64', 'float64'] and col != 'SalePrice']
categorical_cols = df_train.select_dtypes(include=['object']).columns

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define models
model_dict = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Regression': DecisionTreeRegressor(),
    'Random Forest Regression': RandomForestRegressor(),
    'Support Vector Machine Regression': SVR(),
    'XGBoost Regression': xgb.XGBRegressor(),
    'GB Regressor': GradientBoostingRegressor(),
    'Ada Boost Regression': AdaBoostRegressor()
}

# Split the dataset
X = df_train.drop('SalePrice', axis=1)
y = df_train['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Creating pipelines
pipelines = {name: Pipeline(steps=[('preprocessor', preprocessor), ('model', model)]) for name, model in model_dict.items()}

# Training and evaluating models
rmse_results = {}

for name, pipeline in pipelines.items():
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    rmse_results[name] = rmse

# Display RMSE results
rmse_results_sorted = dict(sorted(rmse_results.items(), key=lambda item: item[1]))
print(rmse_results_sorted)

# Splitting the Data
df_train = df_train[~df_train['Id'].isin(df_test['Id'])]
df_test = df_test[df_test['Id'].isin(df_test['Id'])]

# Preparing X_train and y_train
X_train = df_train.drop(['SalePrice'], axis=1)  # Features
y_train = df_train['SalePrice']  # Target variable

# Train the XGBoost regressor
xgb_model = xgb.XGBRegressor()
xgb_model.fit(X_train, y_train)

# Predict the test data using the trained model
y_pred = xgb_model.predict(df_test)  # No need to drop 'SalePrice' since it's not present

# Create a DataFrame for submission
final_file = pd.DataFrame({
    'Id': df_test['Id'],  # Assuming 'Id' is the identifier column in your test data
    'SalePrice': y_pred
})

# Save the submission DataFrame to a CSV file
final_file.to_csv('final_file.csv', index=False)

# Preprocessing steps
numerical_cols = [col for col in df_train.columns if df_train[col].dtype in ['int64', 'float64'] and col != 'SalePrice']
categorical_cols = df_train.select_dtypes(include=['object']).columns

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define models
model_dict = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Regression': DecisionTreeRegressor(),
    'Random Forest Regression': RandomForestRegressor(),
    'Support Vector Machine Regression': SVR(),
    'XGBoost Regression': xgb.XGBRegressor(),
    'GB Regressor': GradientBoostingRegressor(),
    'Ada Boost Regression': AdaBoostRegressor()
}

# Split the dataset
X = df_train.drop('SalePrice', axis=1)
y = df_train['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Creating pipelines
pipelines = {name: Pipeline(steps=[('preprocessor', preprocessor), ('model', model)]) for name, model in model_dict.items()}

# Training and evaluating models
r2_results = {}

for name, pipeline in pipelines.items():
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    r2 = r2_score(y_test, predictions)
    r2_results[name] = r2

# Display R² results
r2_results_sorted = dict(sorted(r2_results.items(), key=lambda item: item[1], reverse=True))
print(r2_results_sorted)