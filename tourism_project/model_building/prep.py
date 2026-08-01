
# ==========================
# Import Required Libraries
# ==========================

# For data manipulation
import pandas as pd

# For splitting the dataset
from sklearn.model_selection import train_test_split


# ==========================
# Load the Registered Dataset
# ==========================

df = pd.read_csv("tourism_project/data/tourism.csv")

print("Dataset loaded successfully.")


# ==========================
# Data Cleaning
# ==========================

# Remove unnecessary column

if "CustomerID" in df.columns:
    df.drop(columns=["CustomerID"], inplace=True)

# Fill missing values in numerical columns using median

numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

for col in numeric_cols:
    if col != "ProdTaken":
        df[col] = df[col].fillna(df[col].median())

# Fill missing values in categorical columns using mode

categorical_cols = df.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("Data cleaning completed successfully.")


# ==========================
# Define Target Variable
# ==========================

target = "ProdTaken"


# ==========================
# List of Numerical Features
# ==========================

numeric_features = [

    "Age",
    "CityTier",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
    "PitchSatisfactionScore",
    "NumberOfFollowups",
    "DurationOfPitch"

]


# ==========================
# List of Categorical Features
# ==========================

categorical_features = [

    "TypeofContact",
    "Occupation",
    "Gender",
    "MaritalStatus",
    "Designation",
    "ProductPitched"

]
