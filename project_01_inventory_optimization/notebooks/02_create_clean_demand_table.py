import pandas as pd
import numpy as np
from pathlib import Path

pd.options.display.float_format = "{:,.2f}".format

# ------------------------------------
# 1. Set project paths
# ------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("Base directory:", BASE_DIR)
print("Raw directory:", RAW_DIR)
print("Processed directory:", PROCESSED_DIR)

# ------------------------------------
# 2. Load raw data
# ------------------------------------
train = pd.read_csv(RAW_DIR / "train.csv")
stores = pd.read_csv(RAW_DIR / "stores.csv")
oil = pd.read_csv(RAW_DIR / "oil.csv")
holidays = pd.read_csv(RAW_DIR / "holidays_events.csv")
transactions = pd.read_csv(RAW_DIR / "transactions.csv")

print("\nFiles loaded successfully.")
print("train:", train.shape)
print("stores:", stores.shape)
print("oil:", oil.shape)
print("holidays:", holidays.shape)
print("transactions:", transactions.shape)

# ------------------------------------
# 3. Convert date columns
# ------------------------------------
train["date"] = pd.to_datetime(train["date"])
oil["date"] = pd.to_datetime(oil["date"])
holidays["date"] = pd.to_datetime(holidays["date"])
transactions["date"] = pd.to_datetime(transactions["date"])

# ------------------------------------
# 4. Basic train data cleaning
# ------------------------------------
demand = train.copy()

# Rename onpromotion for readability
demand = demand.rename(columns={"onpromotion": "items_on_promotion"})

# Make sure sales is numeric
demand["sales"] = pd.to_numeric(demand["sales"], errors="coerce")
demand["items_on_promotion"] = pd.to_numeric(demand["items_on_promotion"], errors="coerce")

# Remove impossible records if any
demand = demand[demand["sales"].notna()]
demand = demand[demand["sales"] >= 0]

print("\nDemand table after basic cleaning:", demand.shape)

# ------------------------------------
# 5. Join store attributes
# ------------------------------------
stores_clean = stores.rename(columns={"type": "store_type"})

demand = demand.merge(
    stores_clean,
    on="store_nbr",
    how="left"
)

print("After joining stores:", demand.shape)

# ------------------------------------
# 6. Join transactions
# ------------------------------------
transactions_clean = transactions.copy()

demand = demand.merge(
    transactions_clean,
    on=["date", "store_nbr"],
    how="left"
)

# If transaction is missing, set to 0
demand["transactions"] = demand["transactions"].fillna(0)

print("After joining transactions:", demand.shape)

# ------------------------------------
# 7. Clean and join oil price
# ------------------------------------
oil_clean = oil.copy()
oil_clean = oil_clean.rename(columns={"dcoilwtico": "oil_price"})

# Create complete daily date range for oil
full_dates = pd.DataFrame({
    "date": pd.date_range(demand["date"].min(), demand["date"].max(), freq="D")
})

oil_clean = full_dates.merge(oil_clean, on="date", how="left")

# Fill missing oil price using forward fill and back fill
oil_clean["oil_price"] = oil_clean["oil_price"].ffill().bfill()

demand = demand.merge(
    oil_clean,
    on="date",
    how="left"
)

print("After joining oil:", demand.shape)

# ------------------------------------
# 8. Clean and join holiday data
# ------------------------------------
holidays_clean = holidays.copy()

# Some holidays are transferred; for this first version, only actual holidays/events are flagged
holidays_clean["is_holiday_event"] = 1

# Aggregate to date level to avoid duplicate rows after merge
holiday_by_date = (
    holidays_clean
    .groupby("date", as_index=False)
    .agg(
        holiday_flag=("is_holiday_event", "max"),
        holiday_count=("is_holiday_event", "sum")
    )
)

demand = demand.merge(
    holiday_by_date,
    on="date",
    how="left"
)

demand["holiday_flag"] = demand["holiday_flag"].fillna(0).astype(int)
demand["holiday_count"] = demand["holiday_count"].fillna(0).astype(int)

print("After joining holidays:", demand.shape)

# ------------------------------------
# 9. Create calendar features
# ------------------------------------
demand["year"] = demand["date"].dt.year
demand["month"] = demand["date"].dt.month
demand["week"] = demand["date"].dt.isocalendar().week.astype(int)
demand["day"] = demand["date"].dt.day
demand["day_of_week"] = demand["date"].dt.dayofweek
demand["day_name"] = demand["date"].dt.day_name()
demand["is_weekend"] = demand["day_of_week"].isin([5, 6]).astype(int)

# Store-family key for forecasting level
demand["store_family_key"] = (
    demand["store_nbr"].astype(str) + "_" + demand["family"].astype(str)
)

# ------------------------------------
# 10. Reorder columns
# ------------------------------------
final_columns = [
    "id",
    "date",
    "store_nbr",
    "family",
    "store_family_key",
    "sales",
    "items_on_promotion",
    "transactions",
    "oil_price",
    "holiday_flag",
    "holiday_count",
    "city",
    "state",
    "store_type",
    "cluster",
    "year",
    "month",
    "week",
    "day",
    "day_of_week",
    "day_name",
    "is_weekend"
]

demand_clean = demand[final_columns].copy()

# ------------------------------------
# 11. Data quality checks
# ------------------------------------
print("\nData quality check:")
print("Final shape:", demand_clean.shape)
print("Date range:", demand_clean["date"].min(), "to", demand_clean["date"].max())
print("Number of stores:", demand_clean["store_nbr"].nunique())
print("Number of product families:", demand_clean["family"].nunique())
print("Number of store-family combinations:", demand_clean["store_family_key"].nunique())

print("\nMissing values:")
print(demand_clean.isna().sum())

print("\nSales summary:")
print(demand_clean["sales"].describe())

print("\nTop 10 product families by sales:")
print(
    demand_clean.groupby("family")["sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 stores by sales:")
print(
    demand_clean.groupby("store_nbr")["sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

# ------------------------------------
# 12. Export cleaned data
# ------------------------------------
output_file = PROCESSED_DIR / "daily_demand_clean.csv"
demand_clean.to_csv(output_file, index=False)

print("\nCleaned demand table exported successfully:")
print(output_file)