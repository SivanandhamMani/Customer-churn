import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import shap
import warnings
warnings.filterwarnings('ignore')


np.random.seed(42)

class CustomerChurnAnalyzer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.shap_explainer = None
        
    def generate_synthetic_data(self, n_samples=10000):
        
        print("Generating synthetic customer churn data")
        
        np.random.seed(42)
        
        data = {
            'customer_id': range(n_samples),
            'tenure': np.random.randint(1, 72, n_samples),  # months
            'monthly_charges': np.random.uniform(20, 120, n_samples),
            'total_charges': np.random.uniform(50, 5000, n_samples),
            'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.5, 0.3, 0.2]),
            'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
            'paperless_billing': np.random.choice(['Yes', 'No'], n_samples, p=[0.7, 0.3]),
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'senior_citizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'partner': np.random.choice(['Yes', 'No'], n_samples, p=[0.5, 0.5]),
            'dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
            'phone_service': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
            'multiple_lines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples, p=[0.4, 0.5, 0.1]),
            'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.4, 0.4, 0.2]),
            'online_security': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.6, 0.1]),
            'online_backup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.6, 0.1]),
            'device_protection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.6, 0.1]),
            'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.6, 0.1]),
            'streaming_tv': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.4, 0.5, 0.1]),
            'streaming_movies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.4, 0.5, 0.1]),
            'customer_service_calls': np.random.randint(0, 10, n_samples),
            'days_since_last_complaint': np.random.randint(1, 365, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        
        churn_prob = (
            (df['tenure'] < 12) * 0.3 +
            (df['contract_type'] == 'Month-to-month') * 0.2 +
            (df['customer_service_calls'] > 5) * 0.25 +
            (df['monthly_charges'] > 80) * 0.15 +
            (df['online_security'] == 'No') * 0.1 +
            np.random.normal(0, 0.1, n_samples)
        )
        
        churn_prob = np.clip(churn_prob, 0, 1)
        df['churn'] = (churn_prob > 0.5).astype(int)
        
        
        print(f"Generated dataset with {df['churn'].sum()} churned customers ({df['churn'].mean():.2%})")
        
        return df
    
    def preprocess_data(self, df):
       
        print("Preprocessing data...")
        
       
        df_processed = df.copy()
        
       
        df_processed = df_processed.drop('customer_id', axis=1)
        
        
        X = df_processed.drop('churn', axis=1)
        y = df_processed['churn']
        
        
        categorical_cols = X.select_dtypes(include=['object']).columns
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
       
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le
        
       
        X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        
       
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.feature_names = X.columns.tolist()
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        
        return X, y
    
    def train_model(self, model_type='xgboost'):
        
        print(f"Training {model_type} model...")
        
        if model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss'
            )
        elif model_type == 'lightgbm':
            self.model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        else:  
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
        
        self.model.fit(self.X_train, self.y_train)
        
    
        train_score = self.model.score(self.X_train, self.y_train)
        test_score = self.model.score(self.X_test, self.y_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        auc_score = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"Training Accuracy: {train_score:.4f}")
        print(f"Test Accuracy: {test_score:.4f}")
        print(f"Test AUC: {auc_score:.4f}")
        
       
        y_pred = self.model.predict(self.X_test)
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        return test_score, auc_score
    
    def compute_shap_values(self):
      
        print("Computing SHAP values")
        
       
        self.shap_explainer = shap.TreeExplainer(self.model)
        
        
        shap_values = self.shap_explainer.shap_values(self.X_test)
        
       
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        return shap_values
    
    def global_interpretation(self, shap_values):
       
        print("\n GLOBAL INTERPRETATION ")
        
       
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, self.X_test, feature_names=self.feature_names, show=False)
        plt.title("SHAP Summary Plot - Global Feature Importance", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
        
       
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, self.X_test, feature_names=self.feature_names, plot_type="bar", show=False)
        plt.title("Mean |SHAP Value| - Feature Importance", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('shap_bar_plot.png', dpi=300, bbox_inches='tight')
        plt.show()
        
      
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance_df.head(10).to_string(index=False))
        
        return feature_importance_df
    
    def local_interpretation(self, shap_values, num_cases=5):
        
        print(f"\n LOCAL INTERPRETATION ({num_cases} CASES) ")
        
       
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
       
        high_risk_indices = np.argsort(y_pred_proba)[-num_cases:]  
        low_risk_indices = np.argsort(y_pred_proba)[:num_cases]    
        
        print("\nHIGH-RISK CUSTOMERS (Likely to Churn)")
        for i, idx in enumerate(high_risk_indices):
            actual_churn = self.y_test.iloc[idx]
            pred_prob = y_pred_proba[idx]
            
            print(f"\nHigh-Risk Case {i+1}:")
            print(f"  Actual Churn: {actual_churn}")
            print(f"  Predicted Churn Probability: {pred_prob:.4f}")
            
            
            plt.figure(figsize=(10, 3))
            shap.force_plot(
                self.shap_explainer.expected_value[1] if hasattr(self.shap_explainer.expected_value, '__len__') else self.shap_explainer.expected_value,
                shap_values[idx, :],
                self.X_test.iloc[idx, :],
                feature_names=self.feature_names,
                matplotlib=True,
                show=False
            )
            plt.title(f"High-Risk Customer {i+1} - SHAP Force Plot", fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'high_risk_customer_{i+1}_force_plot.png', dpi=300, bbox_inches='tight')
            plt.show()
            
           
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[idx, :],
                    base_values=self.shap_explainer.expected_value[1] if hasattr(self.shap_explainer.expected_value, '__len__') else self.shap_explainer.expected_value,
                    data=self.X_test.iloc[idx, :],
                    feature_names=self.feature_names
                ),
                show=False
            )
            plt.title(f"High-Risk Customer {i+1} - SHAP Waterfall Plot", fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'high_risk_customer_{i+1}_waterfall.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        print("\nLOW-RISK CUSTOMERS (Unlikely to Churn) ")
        for i, idx in enumerate(low_risk_indices):
            actual_churn = self.y_test.iloc[idx]
            pred_prob = y_pred_proba[idx]
            
            print(f"\nLow-Risk Case {i+1}:")
            print(f"  Actual Churn: {actual_churn}")
            print(f"  Predicted Churn Probability: {pred_prob:.4f}")
            
           
            plt.figure(figsize=(10, 3))
            shap.force_plot(
                self.shap_explainer.expected_value[1] if hasattr(self.shap_explainer.expected_value, '__len__') else self.shap_explainer.expected_value,
                shap_values[idx, :],
                self.X_test.iloc[idx, :],
                feature_names=self.feature_names,
                matplotlib=True,
                show=False
            )
            plt.title(f"Low-Risk Customer {i+1} - SHAP Force Plot", fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'low_risk_customer_{i+1}_force_plot.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def dependency_analysis(self, shap_values, top_features=3):
       
        print(f"\nFEATURE DEPENDENCY ANALYSIS")
        
       
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-top_features:][::-1]
        
        for i, feature_idx in enumerate(top_indices):
            feature_name = self.feature_names[feature_idx]
            print(f"\nDependency Analysis for '{feature_name}':")
            
            plt.figure(figsize=(10, 6))
            shap.dependence_plot(
                feature_idx,
                shap_values,
                self.X_test,
                feature_names=self.feature_names,
                show=False
            )
            plt.title(f"SHAP Dependence Plot: {feature_name}", fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'dependence_plot_{feature_name}.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def generate_business_insights(self, feature_importance_df):
       
        print("\n=== BUSINESS INSIGHTS & RECOMMENDATIONS ===")
        
        top_features = feature_importance_df.head(5)['feature'].tolist()
        
        print("\nKey Drivers of Customer Churn:")
        for i, feature in enumerate(top_features, 1):
            print(f"  {i}. {feature}")
        
        print("\n" + "="*60)
        print("ACTIONABLE RECOMMENDATIONS:")
       
        
        insights = {
            'tenure': "Focus retention efforts on newer customers (low tenure) who are at higher risk",
            'monthly_charges': "Review pricing strategy for high-value customers and consider loyalty discounts",
            'contract_type': "Promote longer-term contracts to improve customer retention",
            'customer_service_calls': "Improve first-call resolution and customer service quality",
            'online_security': "Bundle security features to increase customer stickiness",
            'total_charges': "Monitor high-spending customers for potential churn risks",
            'payment_method': "Encourage automated payment methods to reduce friction"
        }
        
        for feature in top_features:
            if feature in insights:
                print(f"• {feature.upper()}: {insights[feature]}")
        
        print("\n" + "="*60)
        print("RISK MITIGATION STRATEGIES:")
        print("="*60)
        print("1. Implement early warning system for high-risk customer profiles")
        print("2. Develop targeted retention campaigns based on key churn drivers")
        print("3. Create personalized offers for customers showing churn signals")
        print("4. Enhance customer service for segments with high complaint rates")
        print("5. Regularly monitor model performance and feature importance shifts")

def main():
   
    print("Customer Churn Prediction with SHAP Analysis")
    
    
  
    analyzer = CustomerChurnAnalyzer()
    
   
    df = analyzer.generate_synthetic_data(n_samples=10000)
    X, y = analyzer.preprocess_data(df)
    
    
    try:
        test_score, auc_score = analyzer.train_model('xgboost')
    except:
        print("XGBoost failed, trying LightGBM")
        test_score, auc_score = analyzer.train_model('lightgbm')
    
   
    if auc_score < 0.75:
        print("Warning: Model performance may be insufficient for reliable interpretation")
    
    
    shap_values = analyzer.compute_shap_values()
    
  
    feature_importance_df = analyzer.global_interpretation(shap_values)
    
   
    analyzer.local_interpretation(shap_values, num_cases=3)
    
   
    analyzer.dependency_analysis(shap_values, top_features=3)
    
   
    analyzer.generate_business_insights(feature_importance_df)
    
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    
    print("Generated files:")
    print("- shap_summary_plot.png: Global feature importance")
    print("- shap_bar_plot.png: Mean absolute SHAP values")
    print("- high_risk_customer_*_force_plot.png: Individual explanations for high-risk customers")
    print("- low_risk_customer_*_force_plot.png: Individual explanations for low-risk customers")
    print("- dependence_plot_*.png: Feature dependency plots")
    
   
    feature_importance_df.to_csv('feature_importance.csv', index=False)
    print("- feature_importance.csv: Feature importance rankings")
    
    print(f"\nFinal Model Performance:")
    print(f"Test Accuracy: {test_score:.4f}")
    print(f"Test AUC: {auc_score:.4f}")

if __name__ == "__main__":
    main()


