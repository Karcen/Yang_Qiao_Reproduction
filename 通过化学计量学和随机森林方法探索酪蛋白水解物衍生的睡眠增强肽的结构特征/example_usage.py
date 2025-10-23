import pandas as pd
from core_analysis import PeptideAnalyzer, OPLS

def main():
    # Load simulated data
    print("Loading simulated data...")
    peptide_data = pd.read_csv('enhanced_peptides_data.csv')
    metadata = pd.read_csv('samples_metadata.csv')
    
    # Initialize analyzer
    analyzer = PeptideAnalyzer()
    
    # Prepare features and target
    X = analyzer.prepare_features(peptide_data)
    y = peptide_data['sleep_activity']
    
    # 1. Train Random Forest model
    print("\nTraining Random Forest regression model...")
    rf_results = analyzer.train_random_forest(X, y)
    print(f"Random Forest Results:")
    print(f"  Training R²: {rf_results['train_r2']:.4f}")
    print(f"  Cross-validation R² (mean ± std): {rf_results['cv_r2_mean']:.4f} ± {rf_results['cv_r2_std']:.4f}")
    
    # Plot feature importance
    analyzer.plot_feature_importance()
    print("Feature importance plot saved as 'feature_importance_plot.png'")
    
    # 2. Analyze tyrosine-containing peptides
    print("\nAnalyzing tyrosine-containing peptides...")
    tyr_results = analyzer.analyze_tyr_peptides(peptide_data)
    print(f"  Tyrosine-containing peptides: {tyr_results['count_tyr']}")
    print(f"  Non-tyrosine peptides: {tyr_results['count_non_tyr']}")
    print(f"  Avg activity (tyr): {tyr_results['avg_activity_tyr']:.4f}")
    print(f"  Avg activity (non-tyr): {tyr_results['avg_activity_non_tyr']:.4f}")
    print(f"  Most active peptide types:")
    for typ, act in sorted(tyr_results['type_activity'].items(), key=lambda x: x[1], reverse=True):
        print(f"    {typ}: {act:.4f} (count: {tyr_results['type_counts'][typ]})")
    
    # 3. Analyze peptide length effect
    print("\nAnalyzing peptide length effect...")
    length_results = analyzer.analyze_length_effect(peptide_data)
    optimal_length = length_results.loc[length_results['mean'].idxmax()]['length']
    print(f"  Optimal peptide length (by activity): {optimal_length}")
    print("  Length correlation plot saved as 'peptide_length_correlation.png'")
    
    # 4. Perform OPLS analysis
    print("\nPerforming OPLS analysis...")
    opls = OPLS(n_components=2)
    opls.fit(X.values, y.values)
    print(f"  OPLS Model Results:")
    print(f"  R²Y (model fit): {opls.R2Y:.4f}")
    print(f"  Q² (predictive power): {opls.Q2:.4f}")
    opls.plot_scores(title="OPLS Score Plot: Peptide Features vs Sleep Activity")
    print("  OPLS score plot saved as 'opls_score_plot.png'")
    
    # 5. Identify top potential peptides
    print("\nIdentifying top potential sleep-enhancing peptides...")
    top_peptides = analyzer.identify_potential_peptides(peptide_data)
    top_peptides.to_csv('identified_sleep_peptides.csv', index=False)
    print(f"  Top {len(top_peptides)} peptides saved to 'identified_sleep_peptides.csv'")
    
    # 6. Save top feature importance
    analyzer.feature_importance.to_csv('top_importance_peptides.csv', index=False)
    print("  Feature importance saved to 'top_importance_peptides.csv'")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()