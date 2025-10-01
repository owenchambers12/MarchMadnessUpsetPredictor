import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*does not have valid feature names.*")
from pathlib import Path

# Base dirs relative to this script’s location
SCRIPT_DIR = Path(__file__).resolve().parent                 # .../TournamentModelsNoInput
PROJECT_ROOT = SCRIPT_DIR.parent                             # .../MarchMadnessUpsetPredictor

bart_torvik_path = PROJECT_ROOT / 'bart_torvik_data'
kenpom_path      = PROJECT_ROOT / 'kenpom_data'
tournament_path  = PROJECT_ROOT / 'tournament_data'
in_files_path         = PROJECT_ROOT / 'in_files'

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

    team = name_replacements.get(team, team)
    team = team.replace('-', ' ')
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

# Feature extraction for a single row in the tournament data
def lookup_team_stats(team_name, year, dataset):
    team_data = dataset[(dataset['Team'] == team_name) & (dataset['Year'] == year)]
    if not team_data.empty:
        return team_data.iloc[0]
    else:
        return None

# Feature engineering for tournament data row-by-row
def process_tournament_data(tournament_data, bart_data, kenpom_data):
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

        if seed2 < seed1:
            team1, team2 = team2, team1
            seed1, seed2 = seed2, seed1
            score1, score2 = score2, score1
        
        stats1_bart = lookup_team_stats(team1, year, bart_data)
        stats2_bart = lookup_team_stats(team2, year, bart_data)
        stats1_kenpom = lookup_team_stats(team1, year, kenpom_data)
        stats2_kenpom = lookup_team_stats(team2, year, kenpom_data)
        
        if stats1_bart is None or stats2_bart is None or stats1_kenpom is None or stats2_kenpom is None:
            continue
        
        bart_adjde_diff = stats1_bart['AdjDE'] - stats2_bart['AdjDE']
        bart_defense_fg_pct_diff = stats1_bart['DEFG_Pct'] - stats2_bart['DEFG_Pct']
        sos_def_rating_diff = stats1_kenpom['Strength of Schedule Defensive Rating'] - stats2_kenpom['Strength of Schedule Defensive Rating']
        turnover_pct = stats1_bart['TO_Pct'] - stats2_bart['TO_Pct']
        bart_defense_ft_rate_diff = stats1_bart['DFTR'] - stats2_bart['DFTR']
        
        is_upset = ((seed1 > seed2) and (score1 > score2)) or ((seed2 > seed1) and (score2 > score1))
        
        features.append([bart_adjde_diff, bart_defense_fg_pct_diff, sos_def_rating_diff, turnover_pct, bart_defense_ft_rate_diff])
        labels.append(is_upset)
    
    return features, labels

# Train model without testing phase
def train_model_dynamically(features, labels, tournament_data):
    features = pd.DataFrame(features, columns=['bart_adjde_diff', 'bart_defense_fg_pct_diff', 'sos_def_rating_diff', 'turnover_pct', 'bart_defense_ft_rate_diff'])
    labels = pd.Series(labels)
    
    features['Year'] = tournament_data['Year']
    train_data = features[features['Year'] < 2022]
    
    y_train = labels[features['Year'] < 2022]
    
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_data.drop('Year', axis=1))
    
    # Train model using Logistic Regression
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Return the trained model and scaler (no testing phase here)
    return model, scaler

def predict_upset_from_file(model, scaler, bart_data, kenpom_data, file_path):
    year = int(file_path.split('_')[0])
    file_path = str(in_files_path) + "/" + file_path
    if not os.path.exists(file_path):
        print(f"File {file_path} not found.")
        return
    
    matchups = []  # List to store each matchup and its upset probability

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            try:
                # Parse the input line
                team1, seed1, team2, seed2 = line.split(',')
                seed1 = int(seed1.strip())
                seed2 = int(seed2.strip())
            except ValueError:
                print(f"Invalid input format in line: {line}. Skipping.")
                continue

            # Check if the difference in seeds is at least 3
            if abs(seed1 - seed2) < 3:
                continue  # Skip matchups where seed difference is less than 3

            # Normalize team names
            team1 = normalize_team_name(team1.strip())
            team2 = normalize_team_name(team2.strip())

            # Lookup team stats
            stats1_bart = lookup_team_stats(team1, year, bart_data)
            if team2 == 'SIUE':
                stats2_bart = lookup_team_stats("SIU Edwardsville", year, bart_data)
            else:
                stats2_bart = lookup_team_stats(team2, year, bart_data)
            stats1_kenpom = lookup_team_stats(team1, year, kenpom_data)
            if team2 == 'McNeese St.':
                stats2_kenpom = lookup_team_stats('McNeese', year, kenpom_data)
            else:
                stats2_kenpom = lookup_team_stats(team2, year, kenpom_data)

            # Skip if stats are missing
            if stats1_bart is None or stats2_bart is None or stats1_kenpom is None or stats2_kenpom is None:
                print(f"Missing stats for one or both teams in match-up: {team1} vs {team2}. Skipping.")
                continue

            # Compute features
            bart_adjde_diff = stats1_bart['AdjDE'] - stats2_bart['AdjDE']
            bart_defense_fg_pct_diff = stats1_bart['DEFG_Pct'] - stats2_bart['DEFG_Pct']
            sos_def_rating_diff = stats1_kenpom['Strength of Schedule Defensive Rating'] - stats2_kenpom['Strength of Schedule Defensive Rating']
            turnover_pct = stats1_bart['TO_Pct'] - stats2_bart['TO_Pct']
            bart_defense_ft_rate_diff = stats1_bart['DFTR'] - stats2_bart['DFTR']

            # Prepare feature for prediction
            feature = [[bart_adjde_diff, bart_defense_fg_pct_diff, sos_def_rating_diff, turnover_pct, bart_defense_ft_rate_diff]]
            feature_scaled = scaler.transform(feature)

            # Predict the probability of an upset
            upset_prob = model.predict_proba(feature_scaled)[0][1]

            # Store the matchup and upset probability in the list
            matchups.append({
                "matchup": f"{team1} ({seed1} seed) vs {team2} ({seed2} seed)",
                "upset_prob": upset_prob
            })

    # Sort matchups by upset probability in descending order
    matchups_sorted = sorted(matchups, key=lambda x: x['upset_prob'], reverse=True)

    # Output the sorted matchups
    print(f"Match-ups for {year}:")
    print('-' * 50)
    for matchup in matchups_sorted:
        print(f"Match-up: {matchup['matchup']}")
        print(f"The probability of an upset is: {matchup['upset_prob']:.2f}")
        print('-' * 50)





# Main function
def main():
    print("Loading data...")
    bart_data = load_bart_torvik()
    kenpom_data = load_kenpom()
    tournament_data = load_tournament_data()
    
    print("Processing tournament data...")
    features, labels = process_tournament_data(tournament_data, bart_data, kenpom_data)
    
    print("Training model...")
    model, scaler = train_model_dynamically(features, labels, tournament_data)
    
    print("Predicting upset probability for user input...")
    file_path = input("Enter the name of the input file: ")
    predict_upset_from_file(model, scaler, bart_data, kenpom_data, file_path)

if __name__ == "__main__":
    main()
