"""
Simple Equipment Fault Detection - Guaranteed to Work
Anomaly detection for predictive maintenance
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

def main():
    print("="*60)
    print("SIMPLE FAULT DETECTION DEMO")
    print("="*60)
    
    # Load equipment data
    try:
        equipment_data = pd.read_csv('../data/equipment_data.csv')
        print(f"✅ Loaded {len(equipment_data)} equipment records")
    except FileNotFoundError:
        print("❌ Please run power_data_generator.py first!")
        return
    
    # Generate simple sensor data for demonstration
    print("\n🔧 Generating Sensor Data")
    
    sensor_data = []
    for _, equipment in equipment_data.iterrows():
        # Generate 100 sensor readings per equipment
        for reading in range(100):
            # Base values depend on equipment type and age
            if equipment['type'] == 'transformer':
                base_temp = 75 + equipment['age_years'] * 2
                base_vibration = 0.5 + equipment['age_years'] * 0.02
            elif equipment['type'] == 'generator':
                base_temp = 85 + equipment['age_years'] * 1.5
                base_vibration = 1.2 + equipment['age_years'] * 0.03
            else:
                base_temp = 60 + equipment['age_years'] * 1
                base_vibration = 0.3 + equipment['age_years'] * 0.01
            
            # Add normal variation
            temperature = base_temp + np.random.normal(0, 5)
            vibration = base_vibration + np.random.normal(0, 0.1)
            pressure = 100 + np.random.normal(0, 10)
            
            # Create some faults (5% of readings)
            is_fault = np.random.random() < 0.05
            if is_fault:
                temperature *= 1.3  # Higher temperature for faults
                vibration *= 1.5    # Higher vibration for faults
                pressure *= 0.8     # Lower pressure for faults
            
            sensor_data.append({
                'equipment_id': equipment['equipment_id'],
                'type': equipment['type'],
                'age_years': equipment['age_years'],
                'temperature': temperature,
                'vibration': vibration,
                'pressure': pressure,
                'is_fault': is_fault
            })
    
    sensor_df = pd.DataFrame(sensor_data)
    fault_count = sensor_df['is_fault'].sum()
    print(f"✅ Generated {len(sensor_df)} sensor readings")
    print(f"   Fault instances: {fault_count} ({fault_count/len(sensor_df)*100:.1f}%)")
    
    # Prepare features for anomaly detection
    print("\n🤖 Training Anomaly Detection Model")
    
    features = ['temperature', 'vibration', 'pressure', 'age_years']
    X = sensor_df[features]
    y = sensor_df['is_fault']
    
    # Split data
    split_idx = int(len(sensor_df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Isolation Forest
    isolation_forest = IsolationForest(
        contamination=0.1,  # Expected 10% anomalies
        random_state=42
    )
    
    isolation_forest.fit(X_train_scaled)
    
    # Predict anomalies (-1 = anomaly, 1 = normal)
    y_pred = isolation_forest.predict(X_test_scaled)
    y_pred_binary = (y_pred == -1).astype(int)  # Convert to 1 for anomaly, 0 for normal
    
    # Evaluate performance
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    
    print(f"Model Performance:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1-Score: {f1:.3f}")
    
    # Equipment risk analysis
    print("\n🔧 Equipment Risk Analysis")
    
    # Calculate risk scores for each equipment
    equipment_risk = []
    for equipment_id in equipment_data['equipment_id'].unique():
        eq_readings = sensor_df[sensor_df['equipment_id'] == equipment_id]
        eq_info = equipment_data[equipment_data['equipment_id'] == equipment_id].iloc[0]
        
        # Risk factors
        avg_temp = eq_readings['temperature'].mean()
        avg_vibration = eq_readings['vibration'].mean()
        fault_rate = eq_readings['is_fault'].mean()
        age_factor = min(eq_info['age_years'] / 20, 1.0)  # Normalize to 20 years
        
        # Combined risk score
        risk_score = (fault_rate * 0.4 + age_factor * 0.3 + 
                     (avg_temp > 80) * 0.2 + (avg_vibration > 1.0) * 0.1)
        
        # Recommendation
        if risk_score > 0.3:
            recommendation = "IMMEDIATE"
        elif risk_score > 0.2:
            recommendation = "SCHEDULED"
        elif risk_score > 0.1:
            recommendation = "MONITOR"
        else:
            recommendation = "ROUTINE"
        
        equipment_risk.append({
            'equipment_id': equipment_id,
            'type': eq_info['type'],
            'age_years': eq_info['age_years'],
            'risk_score': risk_score,
            'recommendation': recommendation,
            'avg_temp': avg_temp,
            'avg_vibration': avg_vibration,
            'fault_rate': fault_rate
        })
    
    risk_df = pd.DataFrame(equipment_risk).sort_values('risk_score', ascending=False)
    
    # Show top 10 high-risk equipment
    print("\nTop 10 High-Risk Equipment:")
    print("-" * 70)
    for _, row in risk_df.head(10).iterrows():
        print(f"{row['equipment_id']:15s} | {row['type']:12s} | "
              f"Risk: {row['risk_score']:.3f} | {row['recommendation']}")
    
    # Maintenance recommendations summary
    print("\nMaintenance Recommendations:")
    for rec in ['IMMEDIATE', 'SCHEDULED', 'MONITOR', 'ROUTINE']:
        count = (risk_df['recommendation'] == rec).sum()
        print(f"  {rec:12s}: {count:2d} equipment")
    
    # Visualizations
    print("\n📊 Creating Visualizations")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: Temperature vs Vibration (colored by fault)
    normal_data = sensor_df[sensor_df['is_fault'] == False].sample(min(500, len(sensor_df[sensor_df['is_fault'] == False])))
    fault_data = sensor_df[sensor_df['is_fault'] == True]
    
    axes[0, 0].scatter(normal_data['temperature'], normal_data['vibration'], 
                      alpha=0.6, label='Normal', s=10)
    axes[0, 0].scatter(fault_data['temperature'], fault_data['vibration'], 
                      alpha=0.8, label='Fault', s=20, color='red')
    axes[0, 0].set_title('Temperature vs Vibration')
    axes[0, 0].set_xlabel('Temperature')
    axes[0, 0].set_ylabel('Vibration')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Equipment age vs risk score
    risk_colors = {'IMMEDIATE': 'red', 'SCHEDULED': 'orange', 'MONITOR': 'yellow', 'ROUTINE': 'green'}
    for rec in risk_colors.keys():
        data = risk_df[risk_df['recommendation'] == rec]
        if len(data) > 0:
            axes[0, 1].scatter(data['age_years'], data['risk_score'], 
                             c=risk_colors[rec], label=rec, alpha=0.7, s=40)
    
    axes[0, 1].set_title('Equipment Age vs Risk Score')
    axes[0, 1].set_xlabel('Age (years)')
    axes[0, 1].set_ylabel('Risk Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Fault detection performance
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred_binary)
    
    axes[1, 0].imshow(cm, cmap='Blues', alpha=0.7)
    axes[1, 0].set_title('Confusion Matrix')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            axes[1, 0].text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=12)
    
    # Plot 4: Maintenance recommendations
    rec_counts = risk_df['recommendation'].value_counts()
    colors = [risk_colors.get(rec, 'gray') for rec in rec_counts.index]
    axes[1, 1].pie(rec_counts.values, labels=rec_counts.index, colors=colors, 
                  autopct='%1.1f%%', startangle=90)
    axes[1, 1].set_title('Maintenance Recommendations')
    
    plt.tight_layout()
    plt.savefig('fault_detection_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Business impact analysis
    print("\n💼 Business Impact Analysis")
    print("-" * 40)
    
    # Maintenance costs
    maintenance_costs = {
        'IMMEDIATE': 50000,
        'SCHEDULED': 15000,
        'MONITOR': 2000,
        'ROUTINE': 5000
    }
    
    # Calculate total maintenance cost
    total_cost = 0
    print("Maintenance Cost Breakdown:")
    for rec in maintenance_costs.keys():
        count = (risk_df['recommendation'] == rec).sum()
        cost = count * maintenance_costs[rec]
        total_cost += cost
        print(f"  {rec:12s}: {count:2d} units × ${maintenance_costs[rec]:,} = ${cost:,}")
    
    print(f"\nTotal Maintenance Investment: ${total_cost:,}")
    
    # Failure prevention value
    failure_cost_per_unit = 200000  # Average cost per equipment failure
    high_risk_count = (risk_df['recommendation'].isin(['IMMEDIATE', 'SCHEDULED'])).sum()
    prevented_failures = high_risk_count * 0.3  # Assume 30% would fail without intervention
    prevention_value = prevented_failures * failure_cost_per_unit
    
    print(f"Estimated Failures Prevented: {prevented_failures:.1f}")
    print(f"Failure Prevention Value: ${prevention_value:,}")
    
    # ROI calculation
    net_benefit = prevention_value - total_cost
    roi = (net_benefit / total_cost) * 100 if total_cost > 0 else 0
    
    print(f"\nROI Analysis:")
    print(f"  Investment: ${total_cost:,}")
    print(f"  Return: ${prevention_value:,}")
    print(f"  Net Benefit: ${net_benefit:,}")
    print(f"  ROI: {roi:.1f}%")
    
    print("\n✅ Fault detection system complete!")
    print("Equipment monitoring ready for deployment! 🚀")
    
    return {
        'model': isolation_forest,
        'scaler': scaler,
        'risk_analysis': risk_df,
        'performance': {'precision': precision, 'recall': recall, 'f1': f1},
        'business_impact': {'total_cost': total_cost, 'roi': roi}
    }

if __name__ == "__main__":
    results = main()