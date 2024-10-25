import pandas as pd
import matplotlib.pyplot as plt

# Sample energy data
data = {
    'timestamp': ['2024-10-01 00:00', '2024-10-01 01:00', '2024-10-01 02:00', '2024-10-01 03:00', '2024-10-01 04:00'],
    'energy_consumption_kWh': [2.5, 3.0, 1.8, 2.2, 1.6],
    'cost_per_kWh': [0.15, 0.15, 0.10, 0.10, 0.20]
}
df = pd.DataFrame(data)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Calculate total cost and potential savings
df['cost'] = df['energy_consumption_kWh'] * df['cost_per_kWh']
potential_savings = df[df['cost_per_kWh'] > 0.15]['cost'].sum() * 0.10  # Adjust savings calculation as needed

# Generate a report
report = {
    'total_energy_usage_kWh': df['energy_consumption_kWh'].sum(),
    'total_cost': df['cost'].sum(),
    'potential_savings': potential_savings
}

print("Energy Consumption Report:")
print(f"Total Energy Usage (kWh): {report['total_energy_usage_kWh']}")
print(f"Total Cost ($): {report['total_cost']:.2f}")
print(f"Potential Savings ($): {report['potential_savings']:.2f}")

# Visualize energy consumption over time
plt.figure(figsize=(10, 5))
plt.plot(df['timestamp'], df['energy_consumption_kWh'], marker='o', label='Energy Consumption (kWh)')
plt.title('Energy Consumption Over Time')
plt.xlabel('Time')
plt.ylabel('Energy Consumption (kWh)')
plt.grid(True)
plt.legend()
plt.show()
