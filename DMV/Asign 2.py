import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ------------------- CONFIG ------------------- #
API_KEY = "5b2f6b73ce53a0679124051c5d952d1a"  # Replace with your API key
CITY = "Pune"
UNITS = "metric"  # Use "imperial" for Fahrenheit, "metric" for Celsius
# ---------------------------------------------- #

# 1. Get forecast data
def fetch_forecast(city, api_key):
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units={UNITS}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        forecast_list = data['list']
        records = []

        for entry in forecast_list:
            records.append({
                "datetime": datetime.strptime(entry['dt_txt'], "%Y-%m-%d %H:%M:%S"),
                "temperature": entry['main']['temp'],
                "humidity": entry['main']['humidity'],
                "wind_speed": entry['wind']['speed'],
                "weather": entry['weather'][0]['description']
            })

        return pd.DataFrame(records)
    else:
        print("❌ Failed to fetch forecast data:", response.status_code)
        return pd.DataFrame()

# 2. Clean the data
def clean_data(df):
    df = df.dropna()
    df = df.sort_values("datetime")
    df['date'] = df['datetime'].dt.date
    return df

# 3. Analyze
def analyze(df):
    print("\n📊 Summary Statistics:")
    print(df.describe())

    daily_avg = df.groupby("date").mean(numeric_only=True)
    print("\n📆 Daily Averages:")
    print(daily_avg)

    return daily_avg

# 4. Visualize
def visualize(df, daily_avg):
    # Line Plot: Temperature over time
    plt.figure(figsize=(12, 6))
    plt.plot(df['datetime'], df['temperature'], marker='o', label='Temperature (°C)')
    plt.title("Temperature Trend (3-Hour Interval)")
    plt.xlabel("Datetime")
    plt.ylabel("Temperature (°C)")
    plt.xticks(rotation=45)
    plt.grid()
    plt.tight_layout()
    plt.legend()
    plt.show()

    # Bar Plot: Daily Average Temperature
    daily_avg['temperature'].plot(kind='bar', title="Daily Average Temperature")
    plt.ylabel("Temperature (°C)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Correlation Heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(df[['temperature', 'humidity', 'wind_speed']].corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Between Weather Attributes")
    plt.tight_layout()
    plt.show()

# 5. Main Function
def main():
    df = fetch_forecast(CITY, API_KEY)
    if df.empty:
        return

    df = clean_data(df)
    df.to_csv("forecast_weather_data.csv", index=False)
    daily_avg = analyze(df)
    visualize(df, daily_avg)

if __name__ == "__main__":
    main()
