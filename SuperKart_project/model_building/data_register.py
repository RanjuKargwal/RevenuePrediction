# Load the raw dataset
import pandas as pd

RAW_PATH = "SuperKart_project/data/SuperKart.csv"
data = pd.read_csv(RAW_PATH)

# Validate that the expected columns are present before registering it
expected_columns = [
    "Product_Id", "Product_Weight", "Product_Sugar_Content", "Product_Allocated_Area", "Product_Type",
    "Product_MRP", "Store_Id", "Store_Establishment_Year", "Store_Size",
     "Store_Location_City_Type", "Store_Type", "Product_Store_Sales_Total",
]
missing = [c for c in expected_columns if c not in df.columns]
if missing:
    raise ValueError(f"Dataset is missing expected columns: {missing}")

print("Dataset registered successfully.")
print(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
print("Columns:", list(data.columns))
print("Product_Store_Sales_Total distribution:")
print(data["Product_Store_Sales_Total"].value_counts())
