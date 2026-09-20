import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("SuperKart_project/data/SuperKart.csv")
df.drop(columns=["Product_Id"], inplace=True)
df.drop(columns=["Store_Id"], inplace=True)


# NOTE: categorical columns are intentionally left as raw strings.
# The training pipeline one-hot-encodes them, and the Streamlit app also sends
# raw category values. Encoding them here (e.g. LabelEncoder) would make training
# and serving use different representations, silently breaking predictions.

target = "Product_Store_Sales_Total"
X = df.drop(columns=[target])
y = df[target]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Data prepared: train/test splits written.")
print("Product_Store_Sales_Total distribution in train:")
print(ytrain.value_counts())
