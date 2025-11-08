from turtle import pen
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load CSV files
race_results = pd.read_csv('data/Race_Results.csv')
qualifying = pd.read_csv('data/Qualifying_Results.csv')
race_schedule = pd.read_csv('data/Race_Schedule.csv')
driver_details = pd.read_csv('data/Driver_Details.csv')
team_details = pd.read_csv('data/Team_Details.csv')
race_status = pd.read_csv('data/Race_Status.csv')

print("CSV files loaded successfully")

# Filter races from 2020 onwards
valid_race_ids = race_schedule[race_schedule['year'] >= 2020]['raceId'].unique()
race_results = race_results[race_results['raceId'].isin(valid_race_ids)]
qualifying = qualifying[qualifying['raceId'].isin(valid_race_ids)]

print(f"Filtered to {len(race_results)} race results from 2020 onwards")

# Merge Race Results with Qualifying based on raceId and driverId
df = race_results.merge(
    qualifying[['raceId', 'driverId', 'position']],
    on=['raceId', 'driverId'],
    how='left'
)

df.rename(columns={'position_y': 'qualifying_position'}, inplace=True)
# Add driver and team info (optional for feature enrichment)
driver_details['driver_name'] = driver_details['forename'] + ' ' + driver_details['surname']
df = df.merge(driver_details[['driverId', 'driver_name']], on='driverId', how='left')

df = df.merge(team_details[['constructorId', 'name']], on='constructorId', how='left')

# Map race status codes to descriptions
status_map = dict(zip(race_status['statusId'], race_status['status']))
df['race_status'] = df['statusId'].map(status_map)

# Basic feature engineering
# Create DNF flag (Did Not Finish)
df['dnf'] = df['race_status'].apply(lambda x: 0 if x == 'Finished' else 1)

# Define features for ML model
features = [
    'grid',                # Starting grid position
    'qualifying_position', # Qualifying rank
    'points',              # Points from this race
    'dnf'                  # DNF flag
]

# Fill missing values for qualifying_position with a high number (poor rank)
df['qualifying_position'] = df['qualifying_position'].fillna(30)

X = df[features].copy()

# Binary target: 1 if podium (finish_position <= 3), else 0
y = (df['positionOrder'] <= 3).astype(int)

# Handle missing or invalid data if needed
X = X.fillna(0)

print(f"Feature matrix shape: {X.shape}")
print(f"Target distribution - Podium: {y.sum()}, Non-podium: {(y == 0).sum()}")

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Gradient Boosting Classifier
print("Training Gradient Boosting model...")
model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
model.fit(X_train_scaled, y_train)

print("Model training complete!")

# Evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f'\nModel Accuracy: {accuracy:.3f}')
print('\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=['Non-Podium', 'Podium']))

# Save model and scaler for deployment
joblib.dump(model, 'models/model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("\nModel and scaler saved to models/ directory")
print("Features used:", features)