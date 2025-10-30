import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ignore warnings for better readability
import warnings
warnings.filterwarnings("ignore")

# Step 1: Load Sales Data
def load_data():
    csv_data = pd.read_csv("sales_data.csv")
    excel_data = pd.read_excel("sales_data.xlsx")
    json_data = pd.read_json("sales_data.json")
    return csv_data, excel_data, json_data

# Step 2: Explore the Data
def explore_data(df, name):
    print(f"\n--- {name} Dataset Info ---")
    print(df.info())
    print("\nDescriptive Statistics:")
    print(df.describe(include='all'))
    print("\nSample Data:")
    print(df.head())

# Step 3: Clean the Data
def clean_data(df):
    df = df.drop_duplicates()
    df = df.dropna(subset=["OrderID", "Product", "Sales"])
    df["Sales"] = pd.to_numeric(df["Sales"], errors='coerce')
    df["Quantity"] = pd.to_numeric(df.get("Quantity", 1), errors='coerce').fillna(1)
    df["UnitPrice"] = pd.to_numeric(df.get("UnitPrice", df["Sales"]), errors='coerce').fillna(df["Sales"])
    return df

# Step 4: Convert to Unified Format
def unify_data(csv_df, excel_df, json_df):
    csv_df["Source"] = "CSV"
    excel_df["Source"] = "Excel"
    json_df["Source"] = "JSON"
    combined_df = pd.concat([csv_df, excel_df, json_df], ignore_index=True)
    return combined_df

# Step 5: Data Transformation
def transform_data(df):
    df['OrderDate'] = pd.to_datetime(df['OrderDate'], errors='coerce')
    df['Year'] = df['OrderDate'].dt.year
    df['Month'] = df['OrderDate'].dt.month
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    return df

# Step 6: Data Analysis
def analyze_data(df):
    print("\n--- Data Analysis ---")
    print("Total Sales:", df['Sales'].sum())
    print("Average Order Value:", df.groupby('OrderID')['Sales'].sum().mean())

    print("\nSales by Category:")
    print(df.groupby('Category')['Sales'].sum())

    print("\nTop 5 Products by Sales:")
    print(df.groupby('Product')['Sales'].sum().sort_values(ascending=False).head())

# Step 7: Visualizations
def visualize_data(df):
    # Bar Plot: Sales by Category
    category_sales = df.groupby('Category')['Sales'].sum()
    category_sales.plot(kind='bar', title='Sales by Product Category', colormap='viridis')
    plt.ylabel("Total Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Box Plot: Sales Distribution
    sns.boxplot(x='Category', y='Sales', data=df)
    plt.title("Sales Distribution by Category")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Pie Chart: Sales Share
    df.groupby('Category')['Sales'].sum().plot.pie(autopct='%1.1f%%', figsize=(6,6))
    plt.title("Sales Share by Category")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()

# Main Function
def main():
    # Load
    csv_df, excel_df, json_df = load_data()

    # Explore
    explore_data(csv_df, "CSV")
    explore_data(excel_df, "Excel")
    explore_data(json_df, "JSON")

    # Clean
    csv_df = clean_data(csv_df)
    excel_df = clean_data(excel_df)
    json_df = clean_data(json_df)

    # Combine
    combined_df = unify_data(csv_df, excel_df, json_df)

    # Transform
    combined_df = transform_data(combined_df)

    # Analyze
    analyze_data(combined_df)

    # Visualize
    visualize_data(combined_df)

if __name__ == "__main__":
    main()