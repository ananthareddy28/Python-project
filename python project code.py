import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#
# Load dataset
df = pd.read_csv("C:\\Users\\BHAVESH\\Downloads\\7114_source_data.csv")
df.fillna(0, inplace=True)

# Clean and rename columns
df.columns = df.columns.str.strip().str.replace(r"[^\w]+", "_", regex=True)

df.rename(columns={
    "srcStateName": "State",
    "srcYear": "Financial_Year",
    "srcMonth": "Financial_Month",
    "Central_Goods_and_Services_Tax_CGST_Revenue": "CGST_Revenue",
    "State_Goods_and_Services_Tax_SGST_Revenue": "SGST_Revenue",
    "Integrated_Goods_and_Services_Tax_IGST_Revenue": "IGST_Revenue",
    "CESS_Tax_Revenue": "CESS_Revenue",
    "YearCode": "Year_Code",
    "Year": "Calendar_Year",
    "MonthCode": "Month_Code",
    "Month": "Month"
}, inplace=True)

# Create Total Revenue column
df["Total_Revenue"] = df[["CGST_Revenue", "SGST_Revenue", "IGST_Revenue", "CESS_Revenue"]].sum(axis=1)

# ------------------ OBJECTIVE 1: Exploratory Data Analysis (EDA) ------------------ #
print("\n--- EDA ---")
print(df.info())
print(df.describe())
print("\nMissing Values:\n", df.isnull().sum())
print("\nSample Data:\n", df.head())
plt.figure(figsize=(10,15))
sns.boxplot(df)
plt.show()

# 🔥 Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df[["CGST_Revenue", "SGST_Revenue", "IGST_Revenue", "CESS_Revenue", "Total_Revenue"]].corr(), annot=True, cmap="YlGnBu")
plt.title("Correlation Heatmap of GST Components")
plt.tight_layout()
plt.show()

# 🔄 Pairplot for tax components
sns.pairplot(df[["CGST_Revenue", "SGST_Revenue", "IGST_Revenue", "CESS_Revenue", "Total_Revenue"]])
plt.suptitle("Pairplot of GST Revenue Components", y=1.02)
plt.show()

# ------------------ OBJECTIVE 2: Descriptive Statistics ------------------ #
print("\n--- Summary Statistics ---")
summary_by_state = df.groupby("State")["Total_Revenue"].agg(['mean', 'median', 'max', 'min', 'std']).sort_values(by="mean", ascending=False)
print(summary_by_state.head())

# ------------------ OBJECTIVE 3: Equity Analysis ------------------ #
state_total = df.groupby("State")["Total_Revenue"].sum().sort_values(ascending=False)

plt.figure(figsize=(12,6))
sns.barplot(x=state_total.head(10).values, y=state_total.head(10).index)
plt.title("Top 10 States by Total GST Revenue")
plt.xlabel("Total Revenue (₹ Cr)")
plt.ylabel("State")
plt.tight_layout()
plt.show()

# ------------------ OBJECTIVE 4: Program and Course Distribution (Monthly View) ------------------ #
monthly_revenue = df.groupby("Month")["Total_Revenue"].sum().sort_values()

plt.figure(figsize=(10,10))
monthly_revenue.plot(kind="barh", color="skyblue")
plt.title("Monthly GST Revenue Distribution")
plt.xlabel("Total Revenue (₹ Cr)")
plt.ylabel("Month")
plt.tight_layout()
plt.show()

# ------------------ OBJECTIVE 5: Faculty Profile Analysis (SGST as proxy) ------------------ #
plt.figure(figsize=(14,6))
sns.boxplot(data=df, x="State", y="SGST_Revenue")
plt.xticks(rotation=90)
plt.title("SGST Revenue Distribution per State")
plt.tight_layout()
plt.show()

# ------------------ OTHER OBJECTIVE: State-wise Comparison Dashboard ------------------ #
statewise = df.groupby(["State", "Year_Code"])["Total_Revenue"].sum().unstack().fillna(0)

statewise.plot(kind="bar", figsize=(15,6), stacked=True, colormap="tab20")
plt.title("State-wise GST Collection by Year")
plt.ylabel("Total Revenue (₹ Cr)")
plt.xlabel("State")
plt.tight_layout()
plt.show()
