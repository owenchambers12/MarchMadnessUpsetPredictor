import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*does not have valid feature names.*")
from pathlib import Path


# Base dirs relative to this script’s location
SCRIPT_DIR = Path(__file__).resolve().parent                 # .../TournamentModelsNoInput
PROJECT_ROOT = SCRIPT_DIR.parent                             # .../MarchMadnessUpsetPredictor

bart_torvik_path = PROJECT_ROOT / 'bart_torvik_data'
kenpom_path      = PROJECT_ROOT / 'kenpom_data'
tournament_path  = PROJECT_ROOT / 'tournament_data'

# Load all Bart Torvik data
def load_bart_torvik():
    bart_files = [f for f in os.listdir(bart_torvik_path) if f.endswith('.csv')]
    all_bart_data = []
    for file in bart_files:
        year = int(file.split('_')[-1].split('.')[0])
        data = pd.read_csv(os.path.join(bart_torvik_path, file))
        data['Year'] = year
        data['Team'] = data['Team'].str.extract(r'^([a-zA-Z\s\.\'\&\-]+)')[0].str.strip()  # Standardize team names
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
        data['Team'] = data['Team'].str.rsplit(' ', 1).str[0]  # Standardize team names
        all_kenpom_data.append(data)
    return pd.concat(all_kenpom_data, ignore_index=True)

# Helper function to normalize team names
def normalize_team_name(team):
    # Map of specific team name replacements
    name_replacements = {
        'UNC': 'North Carolina',
        'FDU': 'Fairleigh Dickinson',
        'UConn': 'Connecticut',
        'Loyola (IL)': 'Loyola Chicago',
        'UCSB': 'UC Santa Barbara',
        "St. Peter's": "Saint Peter's",
        'Miami (FL)': 'Miami FL',
        'Miami (OH)': 'Miami OH',
        'College of Charleston': 'Charleston',
        'NC State': 'N.C. State',
        "St. Joseph's": "Saint Joseph's",
        'Pitt': 'Pittsburgh',
        'ETSU': 'East Tennessee St.',
        'Loyola (MD)': 'Loyola MD',
        "St. John's (NY)": "St. John's",
        'Ole Miss': 'Mississippi'
    }

    # Replace team names based on the dictionary
    team = name_replacements.get(team, team)

    # Replace hyphens with spaces
    team = team.replace('-', ' ')

    # Replace 'State' with 'St.', excluding 'N.C. State' and 'NC State'
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

        # Normalize team names in 'Team1' and 'Team2' columns
        data['Team1'] = data['Team1'].apply(normalize_team_name)
        data['Team2'] = data['Team2'].apply(normalize_team_name)

        all_tournament_data.append(data)

    return pd.concat(all_tournament_data, ignore_index=True)

# Feature extraction for a single row in the tournament data
def lookup_team_stats(team_name, year, dataset):
    """
    Look up a team's stats for a given year in the specified dataset.
    """
    team_data = dataset[(dataset['Team'] == team_name) & (dataset['Year'] == year)]
    if not team_data.empty:
        return team_data.iloc[0]  # Return the first matching row as a series
    else:
        return None  # Return None if no match is found

# Feature engineering for tournament data row-by-row
def process_tournament_data(tournament_data, bart_data, kenpom_data):
    """
    Processes tournament data and adds matchup-level features dynamically.
    """
    features = []
    labels = []
    
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
        
        # Compute features (only using selected features)
        bart_adjde_diff = stats1_bart['AdjDE'] - stats2_bart['AdjDE']
        bart_defense_fg_pct_diff = stats1_bart['DEFG_Pct'] - stats2_bart['DEFG_Pct']
        sos_def_rating_diff = stats1_kenpom['Strength of Schedule Defensive Rating'] - stats2_kenpom['Strength of Schedule Defensive Rating']
        turnover_pct = stats1_bart['TO_Pct'] - stats2_bart['TO_Pct']
        bart_defense_ft_rate_diff = stats1_bart['DFTR'] - stats2_bart['DFTR']
        
        # Create label (upset detection)
        is_upset = ((seed1 > seed2) and (score1 > score2)) or ((seed2 > seed1) and (score2 > score1))
        
        # Append features and label
        features.append([bart_adjde_diff, bart_defense_fg_pct_diff, sos_def_rating_diff, turnover_pct, bart_defense_ft_rate_diff])
        labels.append(is_upset)
        feature_names = ['bart_adjde_diff', 'bart_defense_fg_pct_diff', 'sos_def_rating_diff', 'turnover_pct', 'bart_defense_ft_rate_diff']
    
    return features, labels, feature_names

# Train model using dynamically created features with Logistic Regression
def train_model_dynamically(features, labels, tournament_data):
    features = pd.DataFrame(features, columns=['bart_adjde_diff', 'bart_defense_fg_pct_diff', 'sos_def_rating_diff', 'turnover_pct', 'bart_defense_ft_rate_diff'])
    labels = pd.Series(labels)
    
    # Add 'Year' to features so we can split the data by year
    features['Year'] = tournament_data['Year']

    # Split into train (pre-2020) and test (2020-2024)
    train_data = features[features['Year'] < 2020]
    test_data = features[features['Year'] >= 2020]
    
    y_train = labels[features['Year'] < 2020]
    y_test = labels[features['Year'] >= 2020]
    
    # Standardize features (important for Logistic Regression)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_data.drop('Year', axis=1))
    X_test = scaler.transform(test_data.drop('Year', axis=1))
    
    # Train model using Logistic Regression
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, predictions))
    print(f"ROC AUC Score: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])}")
    print(f"Training data size: {len(train_data)}")
    print(f"Test data size: {len(test_data)}")

    return model


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

# Main function
def main():
    print("Loading data...")
    bart_data = load_bart_torvik()
    kenpom_data = load_kenpom()
    tournament_data = load_tournament_data()
    
    print("Processing tournament data...")
    features, labels, feature_names = process_tournament_data(tournament_data, bart_data, kenpom_data)
    
    print("Training model...")
    train_model_dynamically(features, labels, tournament_data)

    compute_vif(features, feature_names)
    

if __name__ == "__main__":
    main()
