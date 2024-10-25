import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    
    # Calculate surplus or deficit energy
    surplus_energy = production - usage

    # Manage battery storage based on surplus and peak hours
    if surplus_energy > 0:  # Excess solar energy
        battery_charge = min(battery_charge + surplus_energy, battery_capacity)
        df.loc[i, 'battery_storage_kWh'] = battery_charge
    elif i % 24 in peak_hours and battery_charge > 0:  # Use battery during peak hours
        battery_usage = min(battery_charge, abs(surplus_energy))
        battery_charge -= battery_usage
        df.loc[i, 'battery_storage_kWh'] = battery_charge

# Generate Report
report = {
    'total_solar_production': df['solar_production_kWh'].sum(),
    'total_energy_usage': df['energy_usage_kWh'].sum(),
    'end_battery_charge': battery_charge
}

print("Solar Energy Management Report:")
print(f"Total Solar Production (kWh): {report['total_solar_production']:.2f}")
print(f"Total Energy Usage (kWh): {report['total_energy_usage']:.2f}")
print(f"End Battery Charge (kWh): {report['end_battery_charge']:.2f}")

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], df['solar_production_kWh'], label='Solar Production (kWh)')
plt.plot(df['timestamp'], df['energy_usage_kWh'], label='Energy Usage (kWh)')
plt.plot(df['timestamp'], df['battery_storage_kWh'], label='Battery Storage (kWh)')
plt.axvspan(df['timestamp'][8], df['timestamp'][9], color='red', alpha=0.1, label='Peak Hours')
plt.axvspan(df['timestamp'][17], df['timestamp'][18], color='red', alpha=0.1)
plt.title("Solar Production, Usage, and Battery Storage Over Time")
plt.xlabel("Time")
plt.ylabel("Energy (kWh)")
plt.legend()
plt.grid(True)
plt.show()
