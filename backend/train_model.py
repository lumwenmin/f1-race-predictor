from turtle import pen
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
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
df = df.merge(race_schedule[['raceId', 'circuitId']], on='raceId', how='left')

df_sorted = df.sort_values(['driverId', 'raceId']).reset_index(drop=True)

df_sorted['driver_recent_form'] = df_sorted.groupby('driverId')['positionOrder'].rolling(
    window=5, 
    min_periods=1
).mean().reset_index(0, drop=True)

df_sorted['driver_circuit_avg_finish'] = df_sorted.groupby(
    ['driverId', 'circuitId']
)['positionOrder'].transform('mean')

df = df_sorted

# Define features for ML model
features = ['driverId', 'circuitId', 'grid', 'driver_recent_form', 'driver_circuit_avg_finish']

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

# Setup column transformer for categorical encoding and numeric scaling
categorical_features = ['driverId', 'circuitId']
numeric_features = ['grid', 'driver_recent_form', 'driver_circuit_avg_finish']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ]
)

# Create pipeline with preprocessing and Gradient Boosting model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    ))
])

print("Training Gradient Boosting model with circuitId and driverId encoding...")
model_pipeline.fit(X_train, y_train)

print("Model training complete!")

# Evaluate the model
y_pred = model_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'\nModel Accuracy: {accuracy:.3f}')
print('\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=['Non-Podium', 'Podium']))

# Save pipeline (includes scaler and encoder) for deployment
joblib.dump(model_pipeline, 'models/model_pipeline.pkl')

print("\nModel pipeline saved to models/model_pipeline.pkl")
print("Features used:", features)

print("\nGenerating lookup tables for frontend predictions...")

driver_stats = df.groupby('driverId').agg({
    'positionOrder': 'mean'
}).rename(columns={'positionOrder': 'avg_finish_position'})

circuit_driver_stats = df.groupby(['driverId', 'circuitId']).agg({
    'positionOrder': 'mean'
}).rename(columns={'positionOrder': 'avg_finish_at_circuit'})

driver_stats.to_csv('models/driver_stats.csv')
circuit_driver_stats.to_csv('models/circuit_driver_stats.csv')

print(f"Driver stats saved: {len(driver_stats)} drivers")
print(f"Circuit-driver stats saved: {len(circuit_driver_stats)} combinations")