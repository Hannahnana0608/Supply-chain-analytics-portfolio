import pandas as pd
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"

# Load raw files
train = pd.read_csv(RAW_DIR / "train.csv")
test = pd.read_csv(RAW_DIR / "test.csv")
stores = pd.read_csv(RAW_DIR / "stores.csv")
oil = pd.read_csv(RAW_DIR / "oil.csv")
holidays = pd.read_csv(RAW_DIR / "holidays_events.csv")
transactions = pd.read_csv(RAW_DIR / "transactions.csv")

print("Files loaded successfully!")
print("-" * 50)

# Check shape
print("train:", train.shape)
print("test:", test.shape)
print("stores:", stores.shape)
print("oil:", oil.shape)
print("holidays:", holidays.shape)
print("transactions:", transactions.shape)

print("-" * 50)

# Check columns
print("Train columns:")
print(train.columns)

print("-" * 50)

# Check date range
train["date"] = pd.to_datetime(train["date"])

print("Train date range:")
print(train["date"].min(), "to", train["date"].max())

print("-" * 50)

# Basic sales check
print("Sales summary:")
print(train["sales"].describe())

print("-" * 50)

# Product family count
print("Number of product families:")
print(train["family"].nunique())

print("-" * 50)

# Store count
print("Number of stores:")
print(train["store_nbr"].nunique())

print("-" * 50)

# Top product families by sales
top_families = (
    train.groupby("family")["sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("Top 10 product families by total sales:")
print(top_families)

print("-" * 50)

# Top stores by sales
top_stores = (
    train.groupby("store_nbr")["sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("Top 10 stores by total sales:")
print(top_stores)