"""
Equipment Fault Detection using Modern AI
Predictive maintenance for power system equipment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class EquipmentFaultDetector:
    """
    AI-powered equipment fault detection system
    """
    
    def __init__(self, equipment_data_path='../data/equipment_data.csv'):
        self.equipment_data = pd.read_csv(equipment_data_path)
        self.scaler = StandardScaler()
        
        print(f"Loaded {len(self.equipment_data)} equipment records")
        print(f"Equipment types: {self.equipment_data['type'].value_counts().to_dict()}")
        
        # Create synthetic time series data for each equipment
        self.generate_time_series_data()
    
    def generate_time_series_data(self):
        """
        Generate realistic time series sensor data for equipment
        """
        print("\n🔧 Generating Equipment Sensor Data")
        
        # Parameters for different equipment types
        equipment_params = {
            'transformer': {
                'temperature_base': 75,
                'temperature_std': 10,
                'vibration_base': 0.5,
                'vibration_std': 0.2,
                'oil_level_base': 0.85,
                'oil_level_std': 0.05
            },
            'generator': {
                'temperature_base': 85,
                'temperature_std': 15,
                'vibration_base': 1.2,
                'vibration_std': 0.4,
                'oil_level_base': 0.90,
                'oil_level_std': 0.03
            },
            'breaker': {
                'temperature_base': 45,
                'temperature_std': 8,
                'vibration_base': 0.3,
                'vibration_std': 0.1,
                'oil_level_base': 0.80,
                'oil_level_std': 0.08
            },
            'motor': {
                'temperature_base': 65,
                'temperature_std': 12,
                'vibration_base': 0.8,
                'vibration_std': 0.3,
                'oil_level_base': 0.75,
                'oil_level_std': 0.10
            },
            'switchgear': {
                'temperature_base': 50,
                'temperature_std': 6,
                'vibration_base': 0.2,
                'vibration_std': 0.08,
                'oil_level_base': 0.70,
                'oil_level_std': 0.12
            }
        }
        
        time_series_data = []
        
        # Generate 30 days of hourly data for each equipment
        for _, equipment in self.equipment_data.iterrows():
            eq_type = equipment['type']
            params = equipment_params[eq_type]
            
            # Base degradation over time (age effect)
            age_factor = 1 + 0.01 * equipment['age_years']
            
            # Generate 30 days of data
            for hour in range(24 * 30):
                # Temperature increases with age and load
                temp_base = params['temperature_base'] * age_factor
                temp_variation = params['temperature_std'] * np.random.normal(0, 1)
                load_effect = equipment['load_factor'] * 5  # Load increases temperature
                temperature = temp_base + temp_variation + load_effect
                
                # Vibration increases with age and decreases with maintenance
                days_since_maintenance = equipment['last_maintenance_days']
                maintenance_factor = 1 + 0.001 * days_since_maintenance
                vibration = params['vibration_base'] * age_factor * maintenance_factor + \
                           params['vibration_std'] * np.random.normal(0, 1)
                
                # Oil level decreases over time
                oil_level = params['oil_level_base'] - 0.0001 * days_since_maintenance + \
                           params['oil_level_std'] * np.random.normal(0, 1)
                oil_level = np.clip(oil_level, 0.1, 1.0)
                
                # Current increases with load
                current = equipment['load_factor'] * 100 + np.random.normal(0, 5)
                
                # Power factor varies
                power_factor = 0.95 + np.random.normal(0, 0.05)
                power_factor = np.clip(power_factor, 0.8, 1.0)
                
                # Failure probability increases with bad conditions
                failure_prob = equipment['failure_probability']
                if temperature > temp_base * 1.2:
                    failure_prob *= 1.5
                if vibration > params['vibration_base'] * 2:
                    failure_prob *= 1.3
                if oil_level < 0.5:
                    failure_prob *= 2.0
                
                # Create faults randomly based on failure probability
                is_fault = np.random.random() < failure_prob / 1000  # Scale down
                
                time_series_data.append({
                    'equipment_id': equipment['equipment_id'],
                    'type': eq_type,
                    'hour': hour,
                    'temperature': temperature,
                    'vibration': vibration,
                    'oil_level': oil_level,
                    'current': current,
                    'power_factor': power_factor,
                    'is_fault': is_fault,
                    'age_years': equipment['age_years'],
                    'load_factor': equipment['load_factor']
                })
        
        self.time_series_data = pd.DataFrame(time_series_data)
        
        fault_count = self.time_series_data['is_fault'].sum()
        print(f"  - Generated {len(self.time_series_data)} sensor readings")
        print(f"  - Fault instances: {fault_count} ({fault_count/len(self.time_series_data)*100:.3f}%)")
        
    def train_isolation_forest(self):
        """
        Traditional anomaly detection using Isolation Forest
        """
        print("\n🌲 Training Isolation Forest for Anomaly Detection")
        
        # Features for anomaly detection
        feature_columns = [
            'temperature', 'vibration', 'oil_level', 'current', 'power_factor',
            'age_years', 'load_factor'
        ]
        
        # Prepare data
        X = self.time_series_data[feature_columns]
        y = self.time_series_data['is_fault']
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Isolation Forest
        self.isolation_forest = IsolationForest(
            contamination=0.1,  # Expected proportion of outliers
            random_state=42,
            n_estimators=100
        )
        
        self.isolation_forest.fit(X_train_scaled)
        
        # Predictions (-1 for anomaly, 1 for normal)
        y_pred_train = self.isolation_forest.predict(X_train_scaled)
        y_pred_test = self.isolation_forest.predict(X_test_scaled)
        
        # Convert to binary (1 for anomaly, 0 for normal)
        y_pred_train_binary = (y_pred_train == -1).astype(int)
        y_pred_test_binary = (y_pred_test == -1).astype(int)
        
        # Evaluate
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precision = precision_score(y_test, y_pred_test_binary)
        recall = recall_score(y_test, y_pred_test_binary)
        f1 = f1_score(y_test, y_pred_test_binary)
        
        print(f"  - Precision: {precision:.3f}")
        print(f"  - Recall: {recall:.3f}")
        print(f"  - F1-Score: {f1:.3f}")
        
        return y_pred_test_binary, y_test
    
    def train_autoencoder(self):
        """
        Deep learning anomaly detection using Autoencoder
        """
        print("\n🧠 Training Autoencoder for Anomaly Detection")
        
        # Features for autoencoder
        feature_columns = [
            'temperature', 'vibration', 'oil_level', 'current', 'power_factor'
        ]
        
        # Prepare data (use only normal data for training)
        normal_data = self.time_series_data[self.time_series_data['is_fault'] == False]
        X_normal = normal_data[feature_columns]
        
        # Scale features
        X_normal_scaled = self.scaler.fit_transform(X_normal)
        
        # Split normal data
        split_idx = int(len(X_normal_scaled) * 0.8)
        X_train = X_normal_scaled[:split_idx]
        X_val = X_normal_scaled[split_idx:]
        
        # Build autoencoder
        input_dim = len(feature_columns)
        
        # Encoder
        encoder_input = keras.Input(shape=(input_dim,))
        encoded = layers.Dense(32, activation='relu')(encoder_input)
        encoded = layers.Dense(16, activation='relu')(encoded)
        encoded = layers.Dense(8, activation='relu')(encoded)
        
        # Decoder
        decoded = layers.Dense(16, activation='relu')(encoded)
        decoded = layers.Dense(32, activation='relu')(decoded)
        decoded = layers.Dense(input_dim, activation='linear')(decoded)
        
        # Create model
        autoencoder = keras.Model(encoder_input, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        print("\n  Autoencoder Architecture:")
        autoencoder.summary()
        
        # Train
        history = autoencoder.fit(
            X_train, X_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_val, X_val),
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ],
            verbose=1
        )
        
        # Test on all data
        X_test = self.time_series_data[feature_columns]
        X_test_scaled = self.scaler.transform(X_test)
        y_test = self.time_series_data['is_fault']
        
        # Calculate reconstruction error
        X_pred = autoencoder.predict(X_test_scaled)
        reconstruction_error = np.mean((X_test_scaled - X_pred) ** 2, axis=1)
        
        # Determine threshold (95th percentile of normal data errors)
        normal_errors = reconstruction_error[y_test == False]
        threshold = np.percentile(normal_errors, 95)
        
        # Predictions
        y_pred_ae = (reconstruction_error > threshold).astype(int)
        
        # Evaluate
        precision = precision_score(y_test, y_pred_ae)
        recall = recall_score(y_test, y_pred_ae)
        f1 = f1_score(y_test, y_pred_ae)
        
        print(f"\n  - Reconstruction threshold: {threshold:.4f}")
        print(f"  - Precision: {precision:.3f}")
        print(f"  - Recall: {recall:.3f}")
        print(f"  - F1-Score: {f1:.3f}")
        
        self.autoencoder = autoencoder
        self.threshold = threshold
        
        return y_pred_ae, y_test, reconstruction_error
    
    def predictive_maintenance_analysis(self):
        """
        Predictive maintenance recommendations
        """
        print("\n🔧 Predictive Maintenance Analysis")
        
        # Calculate risk scores for each equipment
        equipment_risk = []
        
        for equipment_id in self.equipment_data['equipment_id'].unique():
            eq_data = self.time_series_data[
                self.time_series_data['equipment_id'] == equipment_id
            ]
            
            # Recent sensor readings (last 24 hours)
            recent_data = eq_data.tail(24)
            
            # Calculate risk indicators
            temp_risk = (recent_data['temperature'] > recent_data['temperature'].quantile(0.9)).mean()
            vibration_risk = (recent_data['vibration'] > recent_data['vibration'].quantile(0.9)).mean()
            oil_risk = (recent_data['oil_level'] < recent_data['oil_level'].quantile(0.1)).mean()
            
            # Age and maintenance factors
            equipment_info = self.equipment_data[
                self.equipment_data['equipment_id'] == equipment_id
            ].iloc[0]
            
            age_risk = min(equipment_info['age_years'] / 25, 1.0)  # Normalize to 25 years
            maintenance_risk = min(equipment_info['last_maintenance_days'] / 365, 1.0)  # Normalize to 1 year
            
            # Overall risk score
            risk_score = (temp_risk * 0.3 + vibration_risk * 0.25 + oil_risk * 0.2 + 
                         age_risk * 0.15 + maintenance_risk * 0.1)
            
            # Maintenance recommendation
            if risk_score > 0.7:
                recommendation = "IMMEDIATE"
            elif risk_score > 0.5:
                recommendation = "SCHEDULED"
            elif risk_score > 0.3:
                recommendation = "MONITOR"
            else:
                recommendation = "ROUTINE"
            
            equipment_risk.append({
                'equipment_id': equipment_id,
                'type': equipment_info['type'],
                'risk_score': risk_score,
                'recommendation': recommendation,
                'age_years': equipment_info['age_years'],
                'last_maintenance_days': equipment_info['last_maintenance_days']
            })
        
        risk_df = pd.DataFrame(equipment_risk)
        risk_df = risk_df.sort_values('risk_score', ascending=False)
        
        print("\n  Top 10 High-Risk Equipment:")
        print("-" * 70)
        for _, row in risk_df.head(10).iterrows():
            print(f"{row['equipment_id']:15s} | {row['type']:12s} | "
                  f"Risk: {row['risk_score']:.3f} | {row['recommendation']}")
        
        # Summary by recommendation
        print("\n  Maintenance Recommendations Summary:")
        print("-" * 40)
        for rec in ['IMMEDIATE', 'SCHEDULED', 'MONITOR', 'ROUTINE']:
            count = (risk_df['recommendation'] == rec).sum()
            print(f"{rec:12s}: {count:2d} equipment")
        
        return risk_df
    
    def visualize_results(self, if_pred, ae_pred, actual, reconstruction_error, risk_df):
        """
        Comprehensive visualization of fault detection results
        """
        print("\n📊 Creating Visualizations")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. Model comparison
        from sklearn.metrics import roc_curve, auc
        
        # ROC curves
        fpr_if, tpr_if, _ = roc_curve(actual, if_pred)
        fpr_ae, tpr_ae, _ = roc_curve(actual, ae_pred)
        
        roc_auc_if = auc(fpr_if, tpr_if)
        roc_auc_ae = auc(fpr_ae, tpr_ae)
        
        axes[0, 0].plot(fpr_if, tpr_if, label=f'Isolation Forest (AUC = {roc_auc_if:.3f})')
        axes[0, 0].plot(fpr_ae, tpr_ae, label=f'Autoencoder (AUC = {roc_auc_ae:.3f})')
        axes[0, 0].plot([0, 1], [0, 1], 'k--')
        axes[0, 0].set_title('ROC Curves')
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Reconstruction error distribution
        normal_errors = reconstruction_error[actual == False]
        fault_errors = reconstruction_error[actual == True]
        
        axes[0, 1].hist(normal_errors, bins=50, alpha=0.7, label='Normal', density=True)
        axes[0, 1].hist(fault_errors, bins=50, alpha=0.7, label='Fault', density=True)
        axes[0, 1].axvline(self.threshold, color='red', linestyle='--', label='Threshold')
        axes[0, 1].set_title('Reconstruction Error Distribution')
        axes[0, 1].set_xlabel('Reconstruction Error')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Equipment risk scores
        risk_colors = {'IMMEDIATE': 'red', 'SCHEDULED': 'orange', 'MONITOR': 'yellow', 'ROUTINE': 'green'}
        for rec in risk_colors.keys():
            data = risk_df[risk_df['recommendation'] == rec]
            axes[0, 2].scatter(data['age_years'], data['risk_score'], 
                             c=risk_colors[rec], label=rec, alpha=0.7, s=60)
        
        axes[0, 2].set_title('Equipment Risk vs Age')
        axes[0, 2].set_xlabel('Age (years)')
        axes[0, 2].set_ylabel('Risk Score')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Fault detection over time (sample equipment)
        sample_equipment = self.time_series_data[
            self.time_series_data['equipment_id'] == self.time_series_data['equipment_id'].iloc[0]
        ].head(168)  # First week
        
        axes[1, 0].plot(sample_equipment['hour'], sample_equipment['temperature'], 
                       label='Temperature', alpha=0.8)
        axes[1, 0].plot(sample_equipment['hour'], sample_equipment['vibration'] * 50, 
                       label='Vibration x50', alpha=0.8)
        
        # Mark fault instances
        faults = sample_equipment[sample_equipment['is_fault'] == True]
        axes[1, 0].scatter(faults['hour'], faults['temperature'], 
                          color='red', s=100, marker='x', label='Faults')
        
        axes[1, 0].set_title('Equipment Sensor Data (Sample)')
        axes[1, 0].set_xlabel('Hour')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Confusion matrix for autoencoder
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(actual, ae_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_title('Autoencoder Confusion Matrix')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        
        # 6. Maintenance recommendations
        rec_counts = risk_df['recommendation'].value_counts()
        colors = [risk_colors[rec] for rec in rec_counts.index]
        axes[1, 2].pie(rec_counts.values, labels=rec_counts.index, 
                      colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 2].set_title('Maintenance Recommendations')
        
        plt.tight_layout()
        plt.savefig('fault_detection_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def business_impact_analysis(self, risk_df):
        """
        Calculate business impact of predictive maintenance
        """
        print("\n💼 Business Impact Analysis")
        print("-" * 50)
        
        # Maintenance costs (realistic estimates)
        maintenance_costs = {
            'IMMEDIATE': 50000,    # Emergency maintenance
            'SCHEDULED': 15000,    # Planned maintenance
            'MONITOR': 2000,       # Increased monitoring
            'ROUTINE': 5000        # Regular maintenance
        }
        
        # Failure costs if not maintained
        failure_costs = {
            'transformer': 500000,
            'generator': 1000000,
            'breaker': 200000,
            'motor': 100000,
            'switchgear': 300000
        }
        
        # Calculate costs
        total_maintenance_cost = 0
        total_failure_prevention = 0
        
        print("\nMaintenance Cost Analysis:")
        for rec in maintenance_costs.keys():
            equipment_count = (risk_df['recommendation'] == rec).sum()
            cost = equipment_count * maintenance_costs[rec]
            total_maintenance_cost += cost
            
            print(f"{rec:12s}: {equipment_count:2d} units × ${maintenance_costs[rec]:,} = ${cost:,}")
        
        print(f"\nTotal Maintenance Cost: ${total_maintenance_cost:,}")
        
        # Calculate failure prevention value
        print("\nFailure Prevention Value:")
        for eq_type in risk_df['type'].unique():
            high_risk = risk_df[
                (risk_df['type'] == eq_type) & 
                (risk_df['recommendation'].isin(['IMMEDIATE', 'SCHEDULED']))
            ]
            
            if len(high_risk) > 0:
                # Assume 30% of high-risk equipment would fail without intervention
                prevented_failures = len(high_risk) * 0.3
                prevention_value = prevented_failures * failure_costs[eq_type]
                total_failure_prevention += prevention_value
                
                print(f"{eq_type:12s}: {prevented_failures:.1f} failures × ${failure_costs[eq_type]:,} = ${prevention_value:,}")
        
        print(f"\nTotal Failure Prevention Value: ${total_failure_prevention:,}")
        
        # ROI calculation
        net_benefit = total_failure_prevention - total_maintenance_cost
        roi = (net_benefit / total_maintenance_cost) * 100 if total_maintenance_cost > 0 else 0
        
        print(f"\nROI Analysis:")
        print(f"Investment (Maintenance): ${total_maintenance_cost:,}")
        print(f"Return (Failure Prevention): ${total_failure_prevention:,}")
        print(f"Net Benefit: ${net_benefit:,}")
        print(f"ROI: {roi:.1f}%")
        
        return {
            'maintenance_cost': total_maintenance_cost,
            'failure_prevention': total_failure_prevention,
            'net_benefit': net_benefit,
            'roi': roi
        }

def main():
    """
    Complete fault detection pipeline
    """
    print("="*60)
    print("EQUIPMENT FAULT DETECTION SYSTEM")
    print("="*60)
    
    # Initialize detector
    detector = EquipmentFaultDetector()
    
    # Train models
    if_pred, if_actual = detector.train_isolation_forest()
    ae_pred, ae_actual, reconstruction_error = detector.train_autoencoder()
    
    # Predictive maintenance analysis
    risk_df = detector.predictive_maintenance_analysis()
    
    # Visualize results
    detector.visualize_results(if_pred, ae_pred, ae_actual, reconstruction_error, risk_df)
    
    # Business impact
    business_impact = detector.business_impact_analysis(risk_df)
    
    # Engineering insights
    print("\n" + "="*60)
    print("ENGINEERING INSIGHTS")
    print("="*60)
    
    insights = """
    1. ANOMALY DETECTION APPROACHES
       - Isolation Forest: Good for mixed anomaly types
       - Autoencoder: Better for complex patterns
       - Hybrid approaches often work best in practice
    
    2. SENSOR FUSION IMPORTANCE
       - Multiple sensors provide redundancy
       - Correlated failures easier to detect
       - Similar to protective relaying philosophy
    
    3. MAINTENANCE OPTIMIZATION
       - Predictive maintenance reduces unplanned outages
       - Risk-based approach prioritizes resources
       - 3-5x ROI typical for power utilities
    
    4. REAL-WORLD IMPLEMENTATION
       - Integration with SCADA/historian systems
       - Alarm management and prioritization
       - Crew scheduling and parts inventory optimization
    """
    
    print(insights)
    
    print("\n✅ Fault detection system complete!")
    print("Equipment health monitoring ready for deployment! 🚀")
    
    return detector

if __name__ == "__main__":
    detector = main()