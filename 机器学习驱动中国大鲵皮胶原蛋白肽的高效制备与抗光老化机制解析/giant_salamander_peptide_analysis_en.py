import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
)
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib as mpl

# Configure font for Chinese compatibility (can be ignored if not needed)
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('default')


class GiantSalamanderPeptideAnalysis:
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.ann_model = None
        self.rf_model = None
        self.ann_best_params = None
        self.rf_best_params = None

    def load_data(self, file_path):
        """Load dataset from Excel file"""
        self.data = pd.read_excel(file_path)
        print(f"Data loaded successfully, shape: {self.data.shape}")
        print("\nDataset information:")
        print(self.data.info())
        print("\nStatistical summary:")
        print(self.data.describe())

    def prepare_data(self, test_size=0.2, random_state=42):
        """Prepare training and testing datasets"""
        # Select features and target variable
        feature_columns = ['HydrolysisTime_h', 'EnzymeDosage_U_per_g', 'Temperature_°C', 'pH', 'SolidLiquidRatio_w_v']
        self.X = self.data[feature_columns]
        self.y = self.data['ElastaseInhibitionRate_EIR']

        # Split training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

        # Standardize features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print("\nData preparation completed:")
        print(f"Training set: {self.X_train.shape}")
        print(f"Testing set: {self.X_test.shape}")

    def train_ann_model(self):
        """Train Artificial Neural Network (ANN) model"""
        print("\nTraining Artificial Neural Network (ANN) model...")

        # Grid search parameter optimization
        param_grid = {
            'hidden_layer_sizes': [(9,), (12,), (9, 6), (12, 6)],
            'activation': ['relu', 'tanh'],
            'solver': ['adam', 'lbfgs'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive'],
            'max_iter': [1000, 2000]
        }

        grid_search = GridSearchCV(
            MLPRegressor(random_state=42),
            param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(self.X_train_scaled, self.y_train)
        self.ann_best_params = grid_search.best_params_
        self.ann_model = grid_search.best_estimator_

        print(f"\nBest parameters for ANN model: {self.ann_best_params}")

    def train_rf_model(self):
        """Train Random Forest (RF) model"""
        print("\nTraining Random Forest (RF) model...")

        # Grid search parameter optimization
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']
        }

        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(self.X_train, self.y_train)
        self.rf_best_params = grid_search.best_params_
        self.rf_model = grid_search.best_estimator_

        print(f"\nBest parameters for RF model: {self.rf_best_params}")

    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)

        metrics = {
            'R²': r2_score(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'MAPE': mean_absolute_percentage_error(y_test, y_pred) * 100
        }

        print(f"\nPerformance evaluation for {model_name} model:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        return metrics, y_pred

    def feature_importance_analysis(self):
        """Analyze feature importance using RF model"""
        if self.rf_model:
            feature_importance = self.rf_model.feature_importances_
            feature_names = self.X.columns

            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)

            print("\nFeature importance analysis:")
            print(importance_df)

            return importance_df
        return None

    def create_comparison_plots(self, ann_metrics, rf_metrics, ann_pred, rf_pred):
        """Generate model comparison plots"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Optimization Model Comparison for Giant Salamander Collagen Peptide Hydrolysis', 
                     fontsize=16, fontweight='bold')

        # 1. Radar chart of model performance
        metrics = ['R²', 'RMSE', 'MAE', 'MAPE']
        ann_values = [ann_metrics['R²'], ann_metrics['RMSE'], ann_metrics['MAE'], ann_metrics['MAPE']]
        rf_values = [rf_metrics['R²'], rf_metrics['RMSE'], rf_metrics['MAE'], rf_metrics['MAPE']]

        scaler = MinMaxScaler()
        all_values = np.array([ann_values, rf_values])
        scaled_values = scaler.fit_transform(all_values.T).T

        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        scaled_values = np.concatenate((scaled_values, scaled_values[:, :1]), axis=1)
        angles += angles[:1]
        metrics += metrics[:1]

        ax1 = axes[0, 0]
        ax1.plot(angles, scaled_values[0], 'o-', linewidth=2, label='ANN Model', color='#2E86AB')
        ax1.fill(angles, scaled_values[0], alpha=0.25, color='#2E86AB')
        ax1.plot(angles, scaled_values[1], 'o-', linewidth=2, label='RF Model', color='#A23B72')
        ax1.fill(angles, scaled_values[1], alpha=0.25, color='#A23B72')
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(metrics[:-1])
        ax1.set_ylim(0, 1)
        ax1.set_title('Model Performance Radar Chart', fontweight='bold')
        ax1.legend()
        ax1.grid(True)

        # 2. Predicted vs Actual
        ax2 = axes[0, 1]
        ax2.scatter(self.y_test, ann_pred, alpha=0.6, label='ANN Model', color='#2E86AB', s=50)
        ax2.scatter(self.y_test, rf_pred, alpha=0.6, label='RF Model', color='#A23B72', s=50)
        ax2.plot([self.y_test.min(), self.y_test.max()],
                 [self.y_test.min(), self.y_test.max()],
                 'k--', lw=2, label='Ideal Prediction')
        ax2.set_xlabel('Actual EIR (%)')
        ax2.set_ylabel('Predicted EIR (%)')
        ax2.set_title('Predicted vs Actual Values', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Residual analysis
        ax3 = axes[1, 0]
        ann_residuals = self.y_test - ann_pred
        rf_residuals = self.y_test - rf_pred

        ax3.scatter(ann_pred, ann_residuals, alpha=0.6, label='ANN Model', color='#2E86AB', s=50)
        ax3.scatter(rf_pred, rf_residuals, alpha=0.6, label='RF Model', color='#A23B72', s=50)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.8)
        ax3.set_xlabel('Predicted EIR (%)')
        ax3.set_ylabel('Residuals')
        ax3.set_title('Residual Analysis', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Feature importance (RF)
        ax4 = axes[1, 1]
        if self.rf_model:
            importance_df = self.feature_importance_analysis()
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            bars = ax4.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
            ax4.set_xlabel('Importance Score')
            ax4.set_title('Feature Importance (RF Model)', fontweight='bold')
            ax4.grid(True, alpha=0.3, axis='x')

            # Annotate bars
            for bar in bars:
                width = bar.get_width()
                ax4.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                         f'{width:.3f}', ha='left', va='center')

        plt.tight_layout()
        plt.savefig('model_comparison_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("\nModel comparison plots saved as model_comparison_analysis.png")

    def optimization_analysis(self):
        """Perform process parameter optimization analysis"""
        if self.ann_model:
            print("\nConducting process parameter optimization analysis...")

            # Define parameter ranges
            param_ranges = {
                'HydrolysisTime_h': np.linspace(1, 8, 50),
                'EnzymeDosage_U_per_g': np.linspace(2000, 10000, 50),
                'Temperature_°C': np.linspace(40, 80, 50),
                'pH': np.linspace(6, 10, 50),
                'SolidLiquidRatio_w_v': np.linspace(0.5, 3, 50)
            }

            fig, axes = plt.subplots(3, 2, figsize=(15, 18))
            fig.suptitle('Single-Factor Optimization Analysis', fontsize=16, fontweight='bold')

            for i, (param_name, param_values) in enumerate(param_ranges.items()):
                if i < 5:
                    row, col = divmod(i, 2)
                    ax = axes[row, col]

                    base_params = self.X.mean().values
                    eir_predictions = []

                    for value in param_values:
                        test_params = base_params.copy()
                        test_params[i] = value
                        test_params_scaled = self.scaler.transform([test_params])
                        prediction = self.ann_model.predict(test_params_scaled)[0]
                        eir_predictions.append(prediction)

                    ax.plot(param_values, eir_predictions, 'b-', linewidth=2, color='#2E86AB')
                    ax.set_xlabel(param_name)
                    ax.set_ylabel('Predicted EIR (%)')
                    ax.set_title(f'Effect of {param_name} on EIR', fontweight='bold')
                    ax.grid(True, alpha=0.3)

                    max_idx = np.argmax(eir_predictions)
                    ax.scatter(param_values[max_idx], eir_predictions[max_idx], color='red', s=100, zorder=5)
                    ax.annotate(f'Optimal: {param_values[max_idx]:.2f}',
                                xy=(param_values[max_idx], eir_predictions[max_idx]),
                                xytext=(10, 10), textcoords='offset points',
                                bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))

            axes[2, 1].remove()

            plt.tight_layout()
            plt.savefig('parameter_optimization_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()

            print("Parameter optimization analysis saved as parameter_optimization_analysis.png")

            # Multi-parameter optimization
            def objective(params):
                params_scaled = self.scaler.transform([params])
                return -self.ann_model.predict(params_scaled)[0]

            initial_guess = self.X.mean().values
            bounds = [(1, 8), (2000, 10000), (40, 80), (6, 10), (0.5, 3)]

            result = minimize(objective, initial_guess, bounds=bounds, method='L-BFGS-B')

            optimal_params = result.x
            optimal_eir = -result.fun

            print("\nMulti-parameter optimization results:")
            print("Optimal process parameter combination:")
            for i, param_name in enumerate(self.X.columns):
                print(f"  {param_name}: {optimal_params[i]:.2f}")
            print(f"Predicted optimal EIR: {optimal_eir:.2f}%")

            return optimal_params, optimal_eir
        return None, None

    def generate_report(self, ann_metrics, rf_metrics, optimal_params=None, optimal_eir=None):
        """Generate analytical report"""
        report = f"""
# Machine Learning Optimization Report on Giant Salamander Collagen Peptide Hydrolysis Process

## 1. Overview
This study applies machine learning to optimize the enzymatic hydrolysis parameters of giant salamander skin collagen peptides to enhance their elastase inhibition rate (EIR), improving anti-photoaging activity.

## 2. Data Summary
- Dataset size: {self.data.shape[0]} samples, {self.data.shape[1]} variables
- Features: Hydrolysis time, enzyme dosage, temperature, pH, and solid-liquid ratio
- Target variable: Elastase inhibition rate (EIR)

## 3. Model Performance Comparison

### 3.1 Artificial Neural Network (ANN)
- Best parameters: {self.ann_best_params}
- R²: {ann_metrics['R²']:.4f}
- RMSE: {ann_metrics['RMSE']:.4f}
- MAE: {ann_metrics['MAE']:.4f}
- MAPE: {ann_metrics['MAPE']:.2f}%

### 3.2 Random Forest (RF)
- Best parameters: {self.rf_best_params}
- R²: {rf_metrics['R²']:.4f}
- RMSE: {rf_metrics['RMSE']:.4f}
- MAE: {rf_metrics['MAE']:.4f}
- MAPE: {rf_metrics['MAPE']:.2f}%

### 3.3 Model Selection Recommendation
Based on R² comparison, {'the ANN model' if ann_metrics['R²'] > rf_metrics['R²'] else 'the RF model'} shows superior performance and is recommended for process optimization.

## 4. Feature Importance Analysis
"""

        importance_df = self.feature_importance_analysis()
        if importance_df is not None:
            report += "\n| Rank | Feature | Importance |\n"
            report += "|------|----------|------------|\n"
            for i, row in importance_df.iterrows():
                report += f"| {len(importance_df)-i} | {row['Feature']} | {row['Importance']:.4f} |\n"

        if optimal_params is not None:
            report += f"""
## 5. Optimization Results

### 5.1 Optimal Parameter Combination
"""
            for i, param_name in enumerate(self.X.columns):
                report += f"- **{param_name}**: {optimal_params[i]:.2f}\n"

            report += f"""
### 5.2 Expected Effect
- Predicted optimal EIR: {optimal_eir:.2f}%
- Improvement over dataset mean: {((optimal_eir - self.y.mean()) / self.y.mean() * 100):.1f}%

## 6. Conclusions and Recommendations
1. Machine learning models effectively capture the relationship between process parameters and EIR.
2. The {'ANN' if ann_metrics['R²'] > rf_metrics['R²'] else 'RF'} model exhibits superior predictive accuracy.
3. Parameter optimization can significantly enhance anti-photoaging potential.

## 7. Future Work
1. Increase sample size to improve generalization.
2. Include additional factors such as enzyme type combinations and pretreatment methods.
3. Conduct in vitro and in vivo experiments to validate predicted anti-photoaging efficacy.
"""

        with open('analysis_report.md', 'w', encoding='utf-8') as f:
            f.write(report)

        print("\nAnalytical report generated and saved as analysis_report.md")
        return report


def main():
    analyzer = GiantSalamanderPeptideAnalysis()

    # 1. Load data
    analyzer.load_data('simulated_giant_salamander_data.xlsx')

    # 2. Prepare data
    analyzer.prepare_data()

    # 3. Train models
    analyzer.train_ann_model()
    analyzer.train_rf_model()

    # 4. Evaluate models
    ann_metrics, ann_pred = analyzer.evaluate_model(
        analyzer.ann_model, analyzer.X_test_scaled, analyzer.y_test, 'ANN'
    )
    rf_metrics, rf_pred = analyzer.evaluate_model(
        analyzer.rf_model, analyzer.X_test, analyzer.y_test, 'RF'
    )

    # 5. Create visualization plots
    analyzer.create_comparison_plots(ann_metrics, rf_metrics, ann_pred, rf_pred)

    # 6. Optimization analysis
    optimal_params, optimal_eir = analyzer.optimization_analysis()

    # 7. Generate report
    analyzer.generate_report(ann_metrics, rf_metrics, optimal_params, optimal_eir)

    print("\n" + "=" * 60)
    print("Optimization analysis of giant salamander collagen peptide hydrolysis completed!")
    print("=" * 60)
    print("Generated files:")
    print("1. simulated_giant_salamander_data.xlsx - Simulated dataset")
    print("2. model_comparison_analysis.png - Model comparison visualization")
    print("3. parameter_optimization_analysis.png - Parameter optimization plots")
    print("4. analysis_report.md - Detailed analytical report")

    return analyzer


if __name__ == "__main__":
    analyzer = main()
