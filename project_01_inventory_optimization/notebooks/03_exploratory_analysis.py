import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Make numbers easier to read
pd.options.display.float_format = "{:,.2f}".format

# ------------------------------------
# 1. Set project paths
# ------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
VISUALS_DIR = BASE_DIR / "visuals"
OUTPUT_DIR = BASE_DIR / "output"

VISUALS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

input_file = PROCESSED_DIR / "daily_demand_clean.csv"

print("Reading cleaned demand table...")
demand = pd.read_csv(input_file)

# Convert date to datetime format
demand["date"] = pd.to_datetime(demand["date"])

print("Data loaded successfully.")
print("Shape:", demand.shape)
print("Date range:", demand["date"].min(), "to", demand["date"].max())

# ------------------------------------
# 2. Monthly sales trend
# ------------------------------------
monthly_sales = (
    demand
    .groupby(["year", "month"], as_index=False)
    .agg(total_sales=("sales", "sum"))
)

monthly_sales["year_month"] = pd.to_datetime(
    monthly_sales["year"].astype(str) + "-" +
    monthly_sales["month"].astype(str) + "-01"
)

monthly_sales.to_csv(OUTPUT_DIR / "monthly_sales_summary.csv", index=False)

plt.figure(figsize=(12, 6))
plt.plot(monthly_sales["year_month"], monthly_sales["total_sales"])
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.ticklabel_format(style="plain", axis="y")
plt.tight_layout()
plt.savefig(VISUALS_DIR / "monthly_sales_trend.png")
plt.close()

print("Created monthly sales trend chart.")

# ------------------------------------
# 3. Top 10 product families by sales
# ------------------------------------
top_families = (
    demand
    .groupby("family", as_index=False)
    .agg(total_sales=("sales", "sum"))
    .sort_values("total_sales", ascending=False)
    .head(10)
)

top_families.to_csv(OUTPUT_DIR / "top_10_product_families.csv", index=False)

plt.figure(figsize=(10, 6))
plt.barh(top_families["family"], top_families["total_sales"])
plt.title("Top 10 Product Families by Sales")
plt.xlabel("Total Sales")
plt.ylabel("Product Family")
plt.ticklabel_format(style="plain", axis="x")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(VISUALS_DIR / "top_10_product_families.png")
plt.close()

print("Created top 10 product families chart.")

# ------------------------------------
# 4. Top 10 stores by sales
# ------------------------------------
top_stores = (
    demand
    .groupby("store_nbr", as_index=False)
    .agg(total_sales=("sales", "sum"))
    .sort_values("total_sales", ascending=False)
    .head(10)
)

top_stores.to_csv(OUTPUT_DIR / "top_10_stores.csv", index=False)

plt.figure(figsize=(10, 6))
plt.bar(top_stores["store_nbr"].astype(str), top_stores["total_sales"])
plt.title("Top 10 Stores by Sales")
plt.xlabel("Store Number")
plt.ylabel("Total Sales")
plt.ticklabel_format(style="plain", axis="y")
plt.tight_layout()
plt.savefig(VISUALS_DIR / "top_10_stores.png")
plt.close()

print("Created top 10 stores chart.")

# ------------------------------------
# 5. Average sales by day of week
# ------------------------------------
weekday_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

sales_by_weekday = (
    demand
    .groupby("day_name", as_index=False)
    .agg(
        avg_sales=("sales", "mean"),
        total_sales=("sales", "sum")
    )
)

sales_by_weekday["day_name"] = pd.Categorical(
    sales_by_weekday["day_name"],
    categories=weekday_order,
    ordered=True
)

sales_by_weekday = sales_by_weekday.sort_values("day_name")
sales_by_weekday.to_csv(OUTPUT_DIR / "sales_by_weekday.csv", index=False)

plt.figure(figsize=(10, 6))
plt.bar(sales_by_weekday["day_name"], sales_by_weekday["avg_sales"])
plt.title("Average Sales by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Average Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(VISUALS_DIR / "avg_sales_by_weekday.png")
plt.close()

print("Created average sales by weekday chart.")

# ------------------------------------
# 6. Promotion impact
# ------------------------------------
demand["promotion_flag"] = (demand["items_on_promotion"] > 0).astype(int)

promotion_summary = (
    demand
    .groupby("promotion_flag", as_index=False)
    .agg(
        avg_sales=("sales", "mean"),
        total_sales=("sales", "sum"),
        row_count=("sales", "count")
    )
)

promotion_summary["promotion_status"] = promotion_summary["promotion_flag"].map({
    0: "No Promotion",
    1: "Promotion"
})

promotion_summary.to_csv(OUTPUT_DIR / "promotion_impact_summary.csv", index=False)

plt.figure(figsize=(8, 6))
plt.bar(promotion_summary["promotion_status"], promotion_summary["avg_sales"])
plt.title("Average Sales: Promotion vs No Promotion")
plt.xlabel("Promotion Status")
plt.ylabel("Average Sales")
plt.tight_layout()
plt.savefig(VISUALS_DIR / "promotion_impact.png")
plt.close()

print("Created promotion impact chart.")

# ------------------------------------
# 7. Holiday impact
# ------------------------------------
holiday_summary = (
    demand
    .groupby("holiday_flag", as_index=False)
    .agg(
        avg_sales=("sales", "mean"),
        total_sales=("sales", "sum"),
        row_count=("sales", "count")
    )
)

holiday_summary["holiday_status"] = holiday_summary["holiday_flag"].map({
    0: "Non-Holiday",
    1: "Holiday/Event"
})

holiday_summary.to_csv(OUTPUT_DIR / "holiday_impact_summary.csv", index=False)

plt.figure(figsize=(8, 6))
plt.bar(holiday_summary["holiday_status"], holiday_summary["avg_sales"])
plt.title("Average Sales: Holiday/Event vs Non-Holiday")
plt.xlabel("Holiday Status")
plt.ylabel("Average Sales")
plt.tight_layout()
plt.savefig(VISUALS_DIR / "holiday_impact.png")
plt.close()

print("Created holiday impact chart.")

# ------------------------------------
# 8. Print summary results
# ------------------------------------
print("\nEDA completed successfully.")
print("Summary files saved to:", OUTPUT_DIR)
print("Charts saved to:", VISUALS_DIR)

print("\nTop 10 Product Families:")
print(top_families)

print("\nTop 10 Stores:")
print(top_stores)

print("\nSales by Weekday:")
print(sales_by_weekday)

print("\nPromotion Summary:")
print(promotion_summary)

print("\nHoliday Summary:")
print(holiday_summary)