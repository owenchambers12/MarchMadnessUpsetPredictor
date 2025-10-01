import pandas as pd
import os
from scipy.stats import pointbiserialr
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Paths to your data folders
bart_torvik_path = 'bart_torvik_data'
kenpom_path = 'kenpom_data'
tournament_path = 'tournament_data'

# Load all Bart Torvik data
def load_bart_torvik():
    bart_files = [f for f in os.listdir(bart_torvik_path) if f.endswith('.csv')]
    all_bart_data = []
    for file in bart_files:
        year = int(file.split('_')[-1].split('.')[0])
        data = pd.read_csv(os.path.join(bart_torvik_path, file))
        data['Year'] = year
        data['Team'] = data['Team'].str.extract(r'^([a-zA-Z\s\.\'\&\-]+)')[0].str.strip()
        all_bart_data.append(data)
    return pd.concat(all_bart_data, ignore_index=True)

# Load all KenPom data
def load_kenpom():
    kenpom_files = [f for f in os.listdir(kenpom_path) if f.endswith('.csv')]
    all_kenpom_data = []
    for file in kenpom_files:
        year = int(file.split('_')[-1].split('.')[0])
        data = pd.read_csv(os.path.join(kenpom_path, file))
        data['Year'] = year
        data['Team'] = data['Team'].str.rsplit(' ', 1).str[0]
        all_kenpom_data.append(data)
    return pd.concat(all_kenpom_data, ignore_index=True)

# Normalize team names
def normalize_team_name(team):
    name_replacements = {
        'UNC': 'North Carolina',
        'FDU': 'Fairleigh Dickinson',
        'UConn': 'Connecticut',
        # Add other replacements as needed
    }
    team = name_replacements.get(team, team).replace('-', ' ')
    if 'State' in team and team not in ['N.C. State', 'NC State']:
        team = team.replace('State', 'St.')
    return team

# Load tournament data
def load_tournament_data():
    tournament_files = [f for f in os.listdir(tournament_path) if f.endswith('.csv')]
    all_tournament_data = []
    for file in tournament_files:
        data = pd.read_csv(os.path.join(tournament_path, file))
        year = int(file.split('_')[0])
        data['Year'] = year
        data['Team1'] = data['Team1'].apply(normalize_team_name)
        data['Team2'] = data['Team2'].apply(normalize_team_name)
        all_tournament_data.append(data)
    return pd.concat(all_tournament_data, ignore_index=True)

# Lookup team stats
def lookup_team_stats(team_name, year, dataset):
    team_data = dataset[(dataset['Team'] == team_name) & (dataset['Year'] == year)]
    return team_data.iloc[0] if not team_data.empty else None

def process_tournament_data(tournament_data, bart_data, kenpom_data):
    """
    Processes tournament data and adds matchup-level features dynamically.
    """
    features = []
    labels = []
    feature_names = []
    
    for _, row in tournament_data.iterrows():
        year = row['Year']
        team1 = row['Team1']
        team2 = row['Team2']
        seed1 = row['Seed1']
        seed2 = row['Seed2']
        score1 = row['Score1']
        score2 = row['Score2']
        
        # Look up team stats
        stats1_bart = lookup_team_stats(team1, year, bart_data)
        stats2_bart = lookup_team_stats(team2, year, bart_data)
        
        stats1_kenpom = lookup_team_stats(team1, year, kenpom_data)
        stats2_kenpom = lookup_team_stats(team2, year, kenpom_data)
        
        # Skip games where stats are missing
        if stats1_bart is None or stats2_bart is None or stats1_kenpom is None or stats2_kenpom is None:
            continue
        
        feature_row = []

        # Filter numeric data
        stats1_bart_numeric = stats1_bart[stats1_bart.apply(lambda x: isinstance(x, (int, float)))]
        stats2_bart_numeric = stats2_bart[stats2_bart.apply(lambda x: isinstance(x, (int, float)))]
        stats1_kenpom_numeric = stats1_kenpom[stats1_kenpom.apply(lambda x: isinstance(x, (int, float)))]
        stats2_kenpom_numeric = stats2_kenpom[stats2_kenpom.apply(lambda x: isinstance(x, (int, float)))]

        # Add dynamic feature names
        if not feature_names:
            feature_names += [f"Bart_{name}" for name in stats1_bart_numeric.index]
            feature_names += [f"KenPom_{name}" for name in stats1_kenpom_numeric.index]

        # Add differences to feature row
        feature_row += list(stats1_bart_numeric.values - stats2_bart_numeric.values)
        feature_row += list(stats1_kenpom_numeric.values - stats2_kenpom_numeric.values)
        
        # Create label
        is_upset = ((seed1 > seed2) and (score1 > score2)) or ((seed2 > seed1) and (score2 > score1))
        
        # Append features and label
        features.append(feature_row)
        labels.append(is_upset)
    
    return features, labels, feature_names

def compute_correlations(features, labels, feature_names):
    """
    Compute correlations between features and the label (upset outcome).
    """
    feature_df = pd.DataFrame(features, columns=feature_names)
    label_series = pd.Series(labels, name="Upset")
    
    # Combine features and labels
    combined_data = pd.concat([feature_df, label_series], axis=1)
    
    # Compute correlations
    correlations = combined_data.corr()['Upset'].drop('Upset')
    
    print("\nFeature Correlations with Upsets:")
    for feature, corr in correlations.sort_values(ascending=False).items():
        print(f"{feature}: {corr:.4f}")
    
    # Plot correlations
    plt.figure(figsize=(10, 6))
    sns.barplot(x=correlations.index, y=correlations.values)
    plt.xticks(rotation=45, ha="right")
    plt.title("Feature Correlations with Upsets")
    plt.ylabel("Correlation")
    plt.tight_layout()
    plt.show()
    
    return correlations

def compute_vif(features, feature_names):
    """
    Compute the Variance Inflation Factor (VIF) for each feature.
    """
    feature_df = pd.DataFrame(features, columns=feature_names)
    
    # Add a constant for intercept
    X = sm.add_constant(feature_df)
    
    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    print("\nVariance Inflation Factors (VIF):")
    print(vif_data)
    
    return vif_data

def compute_correlation_matrix(features, feature_names):
    """
    Compute and visualize the correlation matrix of features.
    """
    feature_df = pd.DataFrame(features, columns=feature_names)
    
    # Compute the correlation matrix
    corr_matrix = feature_df.corr()
    
    print("\nCorrelation Matrix of Features:")
    print(corr_matrix)
    
    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", fmt=".2f", cbar=True)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.show()
    
    return corr_matrix

# Main function
def main():
    print("Loading data...")
    bart_data = load_bart_torvik()
    kenpom_data = load_kenpom()
    tournament_data = load_tournament_data()
    
    print("Processing tournament data...")
    features, labels, feature_names = process_tournament_data(tournament_data, bart_data, kenpom_data)
    
    print("Computing correlations with upsets...")
    compute_correlations(features, labels, feature_names)
    
    print("Computing correlation matrix...")
    compute_correlation_matrix(features, feature_names)
    
    print("Computing VIF...")
    compute_vif(features, feature_names)

if __name__ == "__main__":
    main()
