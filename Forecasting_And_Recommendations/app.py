from flask import Flask, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

app = Flask(__name__)

def generate_forecast_and_recommendations():
    # Generate sample historical data with weather and tariffs
    np.random.seed(0)
    days = pd.date_range(start="2024-09-01", periods=30, freq='D')
    weather_condition = np.random.choice(['Sunny', 'Cloudy', 'Rainy'], size=30)
    tariff_rate = np.random.uniform(0.1, 0.3, size=30)  # Random tariff rates between 0.1 and 0.3

    data = {
        'date': days,
        'historical_energy_usage_kWh': np.random.uniform(1, 5, size=30),
        'historical_solar_production_kWh': np.random.uniform(2, 6, size=30),
        'weather_condition': weather_condition,
        'tariff_rate': tariff_rate
    }
    df = pd.DataFrame(data)

    # Convert categorical weather data to numerical values
    df['weather_code'] = df['weather_condition'].map({'Sunny': 1, 'Cloudy': 2, 'Rainy': 3})

    # Features and target for model training
    X = df[['weather_code', 'tariff_rate']]
    y_energy = df['historical_energy_usage_kWh']
    y_solar = df['historical_solar_production_kWh']

    # Train-test split for energy usage
    X_train, X_test, y_train_energy, y_test_energy = train_test_split(X, y_energy, test_size=0.2, random_state=0)

    # Train-test split for solar production
    X_train_solar, X_test_solar, y_train_solar, y_test_solar = train_test_split(X, y_solar, test_size=0.2, random_state=0)

    # Initialize models
    model_energy = LinearRegression()
    model_solar = LinearRegression()

    # Train models
    model_energy.fit(X_train, y_train_energy)
    model_solar.fit(X_train_solar, y_train_solar)  # Ensure correct variable here

    # Forecast next 7 days using expected weather and tariff data
    forecast_weather = np.random.choice([1, 2, 3], size=7)  # Random forecasted weather for the next 7 days
    forecast_tariffs = np.random.uniform(0.1, 0.3, size=7)
    forecast_X = pd.DataFrame({'weather_code': forecast_weather, 'tariff_rate': forecast_tariffs})

    energy_forecast = model_energy.predict(forecast_X)
    solar_forecast = model_solar.predict(forecast_X)

    # Display forecasted data
    forecast_df = pd.DataFrame({
        'date': pd.date_range(start=days[-1] + pd.Timedelta(days=1), periods=7, freq='D'),
        'energy_forecast_kWh': energy_forecast,
        'solar_forecast_kWh': solar_forecast,
        'forecast_weather': forecast_weather,
        'forecast_tariff': forecast_tariffs
    })

    # Generate personalized recommendations based on forecast
    recommendations = []
    for i, row in forecast_df.iterrows():
        if row['energy_forecast_kWh'] > 4.0:  # Example threshold for high usage
            recommendations.append(f"Consider reducing usage on {row['date'].date()} due to high forecasted energy consumption.")
        if row['solar_forecast_kWh'] < 3.0:  # Example threshold for low solar production
            recommendations.append(f"Consider using stored energy on {row['date'].date()} due to low solar production.")
        if row['forecast_tariff'] > 0.25:  # High tariff rate example
            recommendations.append(f"Shift non-essential usage from {row['date'].date()} to avoid high tariffs.")

    return forecast_df, recommendations

@app.route('/')
def index():
    forecast_df, recommendations = generate_forecast_and_recommendations()

    # You can create visualizations here if needed and save them as images
    plt.figure(figsize=(12, 6))
    plt.plot(forecast_df['date'], forecast_df['energy_forecast_kWh'], label='Energy Forecast (kWh)', color='r')
    plt.plot(forecast_df['date'], forecast_df['solar_forecast_kWh'], label='Solar Forecast (kWh)', color='g')
    plt.title("7-Day Energy and Solar Production Forecast")
    plt.xlabel("Date")
    plt.ylabel("Energy (kWh)")
    plt.legend()
    plt.grid(True)
    plt.savefig('static/forecast_plot.png')  # Save plot as an image
    plt.close()

    return render_template('index.html', forecast=forecast_df.to_dict(orient='records'), recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
