import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data(csv_file='Parkinsson disease.csv'):
    """
    Load the CSV file and prepare data for RFE
    """
    try:
        # Load the data
        df = pd.read_csv(csv_file)
        print(f"Dataset loaded successfully!")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check for missing values
        if df.isnull().sum().any():
            print("\nWarning: Missing values found!")
            print(df.isnull().sum())
            # Fill missing values with median
            df.fillna(df.median(), inplace=True)
            print("Missing values filled with median.")
        
        # Separate features and labels
        # Remove non-feature columns (filename, name, status)
        feature_columns = [col for col in df.columns if col not in ['filename', 'name', 'status']]
        X = df[feature_columns]
        y = df['status']
        
        print(f"\nFeatures shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        print(f"Feature columns: {feature_columns}")
        
        # Check label distribution
        print(f"\nLabel distribution:")
        print(f"Healthy (0): {(y == 0).sum()}")
        print(f"Parkinson's (1): {(y == 1).sum()}")
        
        return X, y, feature_columns, df
        
    except FileNotFoundError:
        print(f"Error: {csv_file} not found!")
        print("Make sure you have run the feature extraction code first.")
        return None, None, None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None

def apply_rfe_analysis(X, y, feature_names, n_features_to_select=5):
    """
    Apply RFE with different algorithms and compare results
    """
    # Standardize the features (important for SVM and Logistic Regression)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Define different estimators for RFE
    estimators = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(kernel='linear', random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    results = {}
    
    print(f"\n{'='*60}")
    print(f"APPLYING RFE WITH {n_features_to_select} FEATURES")
    print(f"{'='*60}")
    
    for name, estimator in estimators.items():
        print(f"\n--- {name} ---")
        
        # Apply RFE
        rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select)
        rfe.fit(X_train, y_train)
        
        # Get selected features
        selected_features = [feature_names[i] for i in range(len(feature_names)) if rfe.support_[i]]
        feature_rankings = [(feature_names[i], rfe.ranking_[i]) for i in range(len(feature_names))]
        feature_rankings.sort(key=lambda x: x[1])
        
        print(f"Selected features: {selected_features}")
        print(f"Feature rankings: {feature_rankings}")
        
        # Transform data with selected features
        X_train_selected = rfe.transform(X_train)
        X_test_selected = rfe.transform(X_test)
        
        # Train and evaluate
        estimator.fit(X_train_selected, y_train)
        y_pred = estimator.predict(X_test_selected)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Accuracy with selected features: {accuracy:.4f}")
        
        # Store results
        results[name] = {
            'rfe': rfe,
            'selected_features': selected_features,
            'feature_rankings': feature_rankings,
            'accuracy': accuracy,
            'estimator': estimator
        }
    
    return results, X_train, X_test, y_train, y_test, scaler

def analyze_feature_importance(results, feature_names):
    """
    Analyze and visualize feature importance across different methods
    """
    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE ANALYSIS")
    print(f"{'='*60}")
    
    # Create a summary of feature selections
    feature_selection_summary = {}
    for method, result in results.items():
        for feature in result['selected_features']:
            if feature not in feature_selection_summary:
                feature_selection_summary[feature] = []
            feature_selection_summary[feature].append(method)
    
    print("\nFeature Selection Summary:")
    print("-" * 40)
    for feature, methods in feature_selection_summary.items():
        print(f"{feature}: Selected by {', '.join(methods)}")
    
    # Find consensus features (selected by multiple methods)
    consensus_features = [feature for feature, methods in feature_selection_summary.items() if len(methods) > 1]
    
    print(f"\nConsensus Features (selected by multiple methods):")
    print(f"{consensus_features}")
    
    return consensus_features, feature_selection_summary

def plot_feature_rankings(results, feature_names):
    """
    Plot feature rankings for different methods
    """
    fig, axes = plt.subplots(1, len(results), figsize=(15, 5))
    if len(results) == 1:
        axes = [axes]
    
    for i, (method, result) in enumerate(results.items()):
        rankings = result['feature_rankings']
        features = [r[0] for r in rankings]
        ranks = [r[1] for r in rankings]
        
        # Create color map (rank 1 = best, higher ranks = worse)
        colors = ['green' if r == 1 else 'orange' if r <= 3 else 'red' for r in ranks]
        
        axes[i].barh(features, ranks, color=colors)
        axes[i].set_title(f'{method}\nFeature Rankings')
        axes[i].set_xlabel('Rank (1 = Best)')
        axes[i].invert_yaxis()
        
        # Add text annotations
        for j, (feature, rank) in enumerate(rankings):
            axes[i].text(rank + 0.1, j, str(rank), va='center')
    
    plt.tight_layout()
    plt.show()

def compare_performance_with_without_rfe(X, y, feature_names, best_rfe_result):
    """
    Compare performance with all features vs RFE-selected features
    """
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Test with all features
    lr_all = LogisticRegression(random_state=42, max_iter=1000)
    lr_all.fit(X_train, y_train)
    y_pred_all = lr_all.predict(X_test)
    accuracy_all = accuracy_score(y_test, y_pred_all)
    
    # Test with RFE-selected features
    rfe = best_rfe_result['rfe']
    X_train_rfe = rfe.transform(X_train)
    X_test_rfe = rfe.transform(X_test)
    
    lr_rfe = LogisticRegression(random_state=42, max_iter=1000)
    lr_rfe.fit(X_train_rfe, y_train)
    y_pred_rfe = lr_rfe.predict(X_test_rfe)
    accuracy_rfe = accuracy_score(y_test, y_pred_rfe)
    
    print(f"Performance with ALL features ({len(feature_names)} features):")
    print(f"  Accuracy: {accuracy_all:.4f}")
    
    print(f"\nPerformance with RFE-selected features ({len(best_rfe_result['selected_features'])} features):")
    print(f"  Accuracy: {accuracy_rfe:.4f}")
    print(f"  Selected features: {best_rfe_result['selected_features']}")
    
    print(f"\nImprovement: {accuracy_rfe - accuracy_all:.4f}")
    
    return accuracy_all, accuracy_rfe

def save_selected_features(consensus_features, feature_selection_summary, output_file='selected_features.txt'):
    """
    Save the selected features to a text file
    """
    with open(output_file, 'w') as f:
        f.write("RFE FEATURE SELECTION RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("CONSENSUS FEATURES (Selected by multiple methods):\n")
        f.write("-" * 50 + "\n")
        for feature in consensus_features:
            f.write(f"• {feature}\n")
        
        f.write(f"\nFEATURE SELECTION SUMMARY:\n")
        f.write("-" * 50 + "\n")
        for feature, methods in feature_selection_summary.items():
            f.write(f"{feature}: {', '.join(methods)}\n")
    
    print(f"\nResults saved to {output_file}")

def main():
    """
    Main function to run the complete RFE analysis
    """
    print("PARKINSON'S VOICE FEATURE SELECTION WITH RFE")
    print("=" * 60)
    
    # Load data
    X, y, feature_names, df = load_and_prepare_data('Parkinsson disease.csv')
    
    if X is None:
        return
    
    # Check if we have enough samples for meaningful analysis
    if len(X) < 10:
        print("Warning: Very few samples. Results may not be reliable.")
        print("Consider collecting more voice samples for better analysis.")
    
    # Apply RFE analysis
    n_features = min(5, len(feature_names) - 1)  # Select at most 5 features
    results, X_train, X_test, y_train, y_test, scaler = apply_rfe_analysis(
        X, y, feature_names, n_features_to_select=n_features
    )
    
    if not results:
        return
    
    # Analyze feature importance
    consensus_features, feature_selection_summary = analyze_feature_importance(results, feature_names)
    
    # Plot feature rankings
    try:
        plot_feature_rankings(results, feature_names)
    except Exception as e:
        print(f"Could not create plots: {e}")
    
    # Find best performing method
    best_method = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_result = results[best_method]
    
    print(f"\nBEST PERFORMING METHOD: {best_method}")
    print(f"Best accuracy: {best_result['accuracy']:.4f}")
    print(f"Best selected features: {best_result['selected_features']}")
    
    # Compare performance
    if len(X) > 5:  # Only if we have enough samples
        compare_performance_with_without_rfe(X, y, feature_names, best_result)
    
    # Save results
    save_selected_features(consensus_features, feature_selection_summary)
    
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*60}")
    
    return results, consensus_features

if __name__ == "__main__":
    results, consensus_features = main()