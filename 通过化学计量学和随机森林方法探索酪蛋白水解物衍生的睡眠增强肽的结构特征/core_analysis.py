import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')

class OPLS:
    """Simplified Orthogonal Projections to Latent Structures implementation"""
    
    def __init__(self, n_components=1):
        self.n_components = n_components
        self.W = None  # Weight matrix
        self.T = None  # Score matrix
        self.P = None  # Loading matrix
        self.Q = None  # Y loading matrix
        self.R2Y = None  # Model fit
        self.Q2 = None  # Predictive ability
        
    def fit(self, X, y, cv=5):
        # Standardize data
        X = StandardScaler().fit_transform(X)
        y = y.reshape(-1, 1) if len(y.shape) == 1 else y
        
        n, p = X.shape
        self.W = np.zeros((p, self.n_components))
        self.T = np.zeros((n, self.n_components))
        self.P = np.zeros((p, self.n_components))
        self.Q = np.zeros((y.shape[1], self.n_components))
        
        X_residual = X.copy()
        y_residual = y.copy()
        
        # Calculate initial R2Y
        ss_total = np.sum(y**2)
        
        for i in range(self.n_components):
            # Compute weights
            w = X_residual.T @ y_residual / (y_residual.T @ y_residual)
            w = w / np.linalg.norm(w)
            self.W[:, i] = w.flatten()
            
            # Compute scores
            t = X_residual @ w
            self.T[:, i] = t.flatten()
            
            # Compute Y loadings
            q = y_residual.T @ t / (t.T @ t)
            self.Q[:, i] = q.flatten()
            
            # Compute X loadings
            p = X_residual.T @ t / (t.T @ t)
            self.P[:, i] = p.flatten()
            
            # Deflate X and Y
            X_residual = X_residual - t @ p.T
            y_residual = y_residual - t @ q.T
        
        # Calculate R2Y
        y_pred = self.T @ self.Q.T
        ss_pred = np.sum((y - y_pred)**2)
        self.R2Y = 1 - (ss_pred / ss_total)
        
        # Calculate Q2 using cross-validation
        loo = LeaveOneOut()
        q2_scores = []
        
        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Fit on training data
            opls_cv = OPLS(n_components=self.n_components)
            opls_cv.fit(X_train, y_train, cv=None)
            
            # Predict on test data
            y_pred_cv = opls_cv.predict(X_test)
            ss_test = np.sum((y_test - y_pred_cv)** 2)
            q2_scores.append(1 - (ss_test / np.sum(y_test**2)))
        
        self.Q2 = np.mean(q2_scores)
        
        return self
    
    def predict(self, X):
        X = StandardScaler().fit_transform(X)
        t = X @ self.W
        return t @ self.Q.T
    
    def plot_scores(self, sample_ids=None, title="OPLS Score Plot"):
        if self.T is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        plt.figure(figsize=(10, 7))
        if sample_ids is not None:
            unique_ids = np.unique(sample_ids)
            for sid in unique_ids:
                mask = sample_ids == sid
                plt.scatter(self.T[mask, 0], self.T[mask, 1] if self.n_components > 1 else np.zeros(sum(mask)), 
                           label=sid, alpha=0.7)
            plt.legend()
        else:
            plt.scatter(self.T[:, 0], self.T[:, 1] if self.n_components > 1 else np.zeros(len(self.T)), 
                       alpha=0.7)
        
        plt.xlabel(f"t1 (R²Y: {self.R2Y:.3f}, Q²: {self.Q2:.3f})")
        if self.n_components > 1:
            plt.ylabel("t2")
        plt.title(title)
        plt.savefig("opls_score_plot.png", dpi=300, bbox_inches='tight')
        plt.close()


class PeptideAnalyzer:
    """Class for peptide analysis implementing key methods from the paper"""
    
    def __init__(self):
        self.rf_model = None
        self.feature_importance = None
        
    def prepare_features(self, peptide_data):
        """Prepare features for modeling"""
        features = peptide_data.drop(['peptide_sequence', 'sleep_activity'], axis=1, errors='ignore')
        return features
    
    def train_random_forest(self, X, y, n_estimators=100, random_state=42):
        """Train Random Forest regression model"""
        self.rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        self.rf_model.fit(X, y)
        
        # Calculate feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Evaluate model
        cv_scores = cross_val_score(self.rf_model, X, y, cv=5, scoring='r2')
        train_r2 = r2_score(y, self.rf_model.predict(X))
        
        return {
            'train_r2': train_r2,
            'cv_r2_mean': np.mean(cv_scores),
            'cv_r2_std': np.std(cv_scores)
        }
    
    def analyze_tyr_peptides(self, peptide_data):
        """Analyze tyrosine-containing peptides"""
        tyr_peptides = peptide_data[peptide_data['contains_tyrosine'] == 1].copy()
        non_tyr_peptides = peptide_data[peptide_data['contains_tyrosine'] == 0].copy()
        
        # Calculate average activity
        avg_activity_tyr = tyr_peptides['sleep_activity'].mean()
        avg_activity_non_tyr = non_tyr_peptides['sleep_activity'].mean()
        
        # Count specific peptide types
        type_counts = {
            'yp_type': tyr_peptides['yp_type'].sum(),
            'yl_type': tyr_peptides['yl_type'].sum(),
            'yi_type': tyr_peptides['yi_type'].sum(),
            'yq_type': tyr_peptides['yq_type'].sum()
        }
        
        # Average activity by type
        type_activity = {
            'yp_type': tyr_peptides[tyr_peptides['yp_type'] == 1]['sleep_activity'].mean(),
            'yl_type': tyr_peptides[tyr_peptides['yl_type'] == 1]['sleep_activity'].mean(),
            'yi_type': tyr_peptides[tyr_peptides['yi_type'] == 1]['sleep_activity'].mean(),
            'yq_type': tyr_peptides[tyr_peptides['yq_type'] == 1]['sleep_activity'].mean()
        }
        
        # Plot distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(tyr_peptides['sleep_activity'], label='Tyrosine-containing', kde=True, alpha=0.5)
        sns.histplot(non_tyr_peptides['sleep_activity'], label='Non-tyrosine', kde=True, alpha=0.5)
        plt.xlabel('Sleep Activity')
        plt.ylabel('Count')
        plt.title('Distribution of Sleep Activity: Tyrosine vs Non-tyrosine Peptides')
        plt.legend()
        plt.savefig('tyr_peptide_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'count_tyr': len(tyr_peptides),
            'count_non_tyr': len(non_tyr_peptides),
            'avg_activity_tyr': avg_activity_tyr,
            'avg_activity_non_tyr': avg_activity_non_tyr,
            'type_counts': type_counts,
            'type_activity': type_activity
        }
    
    def analyze_length_effect(self, peptide_data):
        """Analyze effect of peptide length on activity"""
        length_activity = peptide_data.groupby('length')['sleep_activity'].agg(['mean', 'std', 'count']).reset_index()
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.errorbar(length_activity['length'], length_activity['mean'], 
                     yerr=length_activity['std'], fmt='o-', capsize=5)
        plt.xlabel('Peptide Length')
        plt.ylabel('Average Sleep Activity')
        plt.title('Correlation Between Peptide Length and Sleep Activity')
        plt.axvspan(4, 10, color='green', alpha=0.1, label='Optimal Length (4-10)')
        plt.legend()
        plt.savefig('peptide_length_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return length_activity
    
    def plot_feature_importance(self, top_n=15):
        """Plot feature importance from Random Forest"""
        if self.feature_importance is None:
            raise ValueError("No feature importance available. Train a model first.")
            
        top_features = self.feature_importance.head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Features Contributing to Sleep Activity')
        plt.savefig('feature_importance_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def identify_potential_peptides(self, peptide_data, top_n=100):
        """Identify top potential sleep-enhancing peptides"""
        if 'sleep_activity' in peptide_data.columns:
            return peptide_data.sort_values('sleep_activity', ascending=False).head(top_n)
        elif self.rf_model is not None:
            X = self.prepare_features(peptide_data)
            peptide_data['predicted_activity'] = self.rf_model.predict(X)
            return peptide_data.sort_values('predicted_activity', ascending=False).head(top_n)
        else:
            raise ValueError("Either provide data with 'sleep_activity' or train a model first.")