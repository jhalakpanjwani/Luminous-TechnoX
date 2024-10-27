# app.py
from flask import Flask, render_template, jsonify
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/energy_data')
def energy_data():
    # Sample energy data
    data = {
        'timestamp': ['2024-10-01 00:00', '2024-10-01 01:00', '2024-10-01 02:00', '2024-10-01 03:00', '2024-10-01 04:00'],
        'energy_consumption_kWh': [2.5, 3.0, 1.8, 2.2, 1.6],
        'cost_per_kWh': [0.15, 0.15, 0.10, 0.10, 0.20]
    }
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['cost'] = df['energy_consumption_kWh'] * df['cost_per_kWh']
    
    # Calculate total energy usage, total cost, and potential savings
    potential_savings = df[df['cost_per_kWh'] > 0.15]['cost'].sum() * 0.10
    report = {
        'total_energy_usage_kWh': df['energy_consumption_kWh'].sum(),
        'total_cost': df['cost'].sum(),
        'potential_savings': potential_savings
    }

    # Format data for JSON response
    response = {
        'timestamps': df['timestamp'].astype(str).tolist(),
        'energy_consumption': df['energy_consumption_kWh'].tolist(),
        'report': report
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
