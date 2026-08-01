
import pandas as pd

# Path to the raw dataset
RAW_PATH = "tourism_project/data/tourism.csv"

# Load the raw dataset
df = pd.read_csv(RAW_PATH)

# Validate that the expected columns are present before registering it
expected_columns = [
    "ProdTaken",
    "Age",
    "TypeofContact",
    "CityTier",
    "Occupation",
    "Gender",
    "NumberOfPersonVisiting",
    "PreferredPropertyStar",
    "MaritalStatus",
    "NumberOfTrips",
    "Passport",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",
    "MonthlyIncome",
    "PitchSatisfactionScore",
    "ProductPitched",
    "NumberOfFollowups",
    "DurationOfPitch"
]

missing = [col for col in expected_columns if col not in df.columns]

if missing:
    raise ValueError(
        f"Dataset is missing expected columns: {missing}"
    )

print("Dataset registered successfully.")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\nColumns:")
print(list(df.columns))

print("\nTarget Variable Distribution (ProdTaken):")
print(df["ProdTaken"].value_counts())

print("\nMissing Values:")
print(df.isnull().sum())
