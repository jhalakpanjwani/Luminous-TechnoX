from flask import Flask, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Generate and save plot
def generate_energy_data_and_plot():
    # Simulation settings
    np.random.seed(0)
    time_slots = pd.date_range("2024-10-01 00:00", periods=24, freq="H")
    peak_hours = [8, 9, 17, 18]  # Peak tariff hours

    # Generate sample data
    data = {
        'timestamp': time_slots,
        'solar_production_kWh': np.random.uniform(0, 5, size=24),  # Random solar production between 0-5 kWh
        'energy_usage_kWh': np.random.uniform(1, 4, size=24)  # Random energy usage between 1-4 kWh
    }

    df = pd.DataFrame(data)
    df['battery_storage_kWh'] = 0.0
    battery_capacity = 10.0  # Maximum battery capacity in kWh
    battery_charge = 0.0     # Current battery charge

    # Track solar production, usage, and battery storage
    for i in range(len(df)):
        production = df.loc[i, 'solar_production_kWh']
        usage = df.loc[i, 'energy_usage_kWh']
        surplus_energy = production - usage

        # Manage battery storage based on surplus and peak hours
        if surplus_energy > 0:
            battery_charge = min(battery_charge + surplus_energy, battery_capacity)
            df.loc[i, 'battery_storage_kWh'] = battery_charge
        elif i % 24 in peak_hours and battery_charge > 0:
            battery_usage = min(battery_charge, abs(surplus_energy))
            battery_charge -= battery_usage
            df.loc[i, 'battery_storage_kWh'] = battery_charge

    # Generate report
    report = {
        'total_solar_production': df['solar_production_kWh'].sum(),
        'total_energy_usage': df['energy_usage_kWh'].sum(),
        'end_battery_charge': battery_charge
    }

    # Create and save plot
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['solar_production_kWh'], label='Solar Production (kWh)')
    plt.plot(df['timestamp'], df['energy_usage_kWh'], label='Energy Usage (kWh)')
    plt.plot(df['timestamp'], df['battery_storage_kWh'], label='Battery Storage (kWh)')

    # Highlight peak hours only once
    for i, hour in enumerate(peak_hours):
        plt.axvspan(df['timestamp'][hour], df['timestamp'][hour+1], color='red', alpha=0.1, label='Peak Hours' if i == 0 else None)
    
    plt.title("Solar Production, Usage, and Battery Storage Over Time")
    plt.xlabel("Time")
    plt.ylabel("Energy (kWh)")
    plt.legend()
    plt.grid(True)
    
    # Save the plot to the static folder
    plot_path = "static/energy_plot.png"
    plt.savefig(plot_path)
    plt.close()

    return report, plot_path

@app.route("/")
def index():
    report, plot_path = generate_energy_data_and_plot()
    return render_template("index.html", report=report, plot_path=plot_path)

if __name__ == "__main__":
    app.run(debug=True)
