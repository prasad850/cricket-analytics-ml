# ml_models.py
# Machine Learning Models for Cricket Score and Match Prediction
# Uses: Linear Regression, Logistic Regression, Random Forest
# Features: Player performance prediction, match outcome prediction, advanced feature engineering

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
import joblib
import os
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class CricketMLPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = {}
        self.is_trained = False
        
    def load_and_prepare_data(self, batsmen_file: str, team_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and prepare cricket data for ML training
        """
        # Load batsmen data
        df_bat = pd.read_excel(batsmen_file, sheet_name="Batsmen", skiprows=2, header=0)
        df_bat.columns = [
            "Player", "Role", "Season_Performance", "Team", "Match_5_vs_MI",
            "Match_10_vs_RCB", "Match_14_vs_PBKS", "Match_18_vs_LSG",
            "Match_23_vs_MI", "Match_28_vs_SRH", "Match_31_vs_CSK",
            "Match_32_vs_RCB", "Unused", "Wickets"
        ]
        df_bat = df_bat.dropna(subset=["Player"]).drop(columns=["Unused"], errors="ignore")
        
        # Melt the data for better structure
        match_cols = [c for c in df_bat.columns if c.startswith("Match")]
        df_melted = pd.melt(
            df_bat, id_vars=["Player", "Role", "Team", "Wickets"],
            value_vars=match_cols, var_name="Match", value_name="Performance"
        )
        
        # Extract opponent from match name with better error handling
        df_melted["Opponent"] = df_melted["Match"].str.extract(r'vs_([A-Z]+)')[0]
        df_melted["Opponent"] = df_melted["Opponent"].fillna("UNKNOWN")
        
        match_numbers = df_melted["Match"].str.extract(r'Match (\d+)')[0]
        df_melted["Match_Number"] = pd.to_numeric(match_numbers, errors='coerce').fillna(0).astype(int)
        
        # Clean performance data
        df_melted["Performance"] = (
            df_melted["Performance"].astype(str).str.replace("*", "", regex=False)
            .apply(pd.to_numeric, errors="coerce").fillna(0)
        )
        df_melted["Wickets"] = pd.to_numeric(df_melted["Wickets"], errors="coerce").fillna(0)
        
        # Separate runs and wickets based on role
        df_melted["Runs"] = df_melted.apply(
            lambda x: float(x["Performance"]) if "Bowler" not in str(x["Role"]) else 0, axis=1
        )
        df_melted["WicketsTaken"] = df_melted.apply(
            lambda x: float(x["Wickets"]) if "Bowler" in str(x["Role"]) else 0, axis=1
        )
        
        # Load team data
        try:
            df_team = pd.read_excel(team_file, sheet_name="Sheet1")
            # Clean column names
            df_team.columns = df_team.columns.str.strip()
            
            # Check if we have the required columns
            required_cols = ["Team", "Matches Played", "Wins"]
            available_cols = df_team.columns.tolist()
            
            if all(col in available_cols for col in required_cols):
                df_team = df_team[required_cols + (["Losses", "Net Run Rate"] if "Net Run Rate" in available_cols else [])].dropna()
                df_team["Win_Rate"] = df_team["Wins"] / df_team["Matches Played"]
            else:
                logger.warning(f"Required columns not found. Available: {available_cols}")
                df_team = pd.DataFrame()
        except Exception as e:
            logger.warning(f"Could not load team data: {e}")
            df_team = pd.DataFrame()
            
        return df_melted, df_team
    
    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced features for better prediction
        """
        features_df = df.copy()
        
        # Encode categorical variables
        if 'Role' not in self.encoders:
            self.encoders['Role'] = LabelEncoder()
            features_df['Role_Encoded'] = self.encoders['Role'].fit_transform(features_df['Role'].astype(str))
        else:
            features_df['Role_Encoded'] = self.encoders['Role'].transform(features_df['Role'].astype(str))
            
        if 'Team' not in self.encoders:
            self.encoders['Team'] = LabelEncoder()
            features_df['Team_Encoded'] = self.encoders['Team'].fit_transform(features_df['Team'].astype(str))
        else:
            features_df['Team_Encoded'] = self.encoders['Team'].transform(features_df['Team'].astype(str))
            
        if 'Opponent' not in self.encoders:
            self.encoders['Opponent'] = LabelEncoder()
            features_df['Opponent_Encoded'] = self.encoders['Opponent'].fit_transform(features_df['Opponent'].astype(str))
        else:
            features_df['Opponent_Encoded'] = self.encoders['Opponent'].transform(features_df['Opponent'].astype(str))
        
        # Player-specific features
        player_stats = features_df.groupby('Player').agg({
            'Runs': ['mean', 'std', 'max', 'min'],
            'WicketsTaken': ['mean', 'std', 'max'],
            'Match_Number': 'count'
        }).reset_index()
        
        player_stats.columns = ['Player', 'Avg_Runs', 'Std_Runs', 'Max_Runs', 'Min_Runs',
                               'Avg_Wickets', 'Std_Wickets', 'Max_Wickets', 'Total_Matches']
        
        features_df = features_df.merge(player_stats, on='Player', how='left')
        
        # Recent form (last 3 matches)
        features_df = features_df.sort_values(['Player', 'Match_Number'])
        features_df['Recent_Form_Runs'] = features_df.groupby('Player')['Runs'].rolling(3, min_periods=1).mean().values
        features_df['Recent_Form_Wickets'] = features_df.groupby('Player')['WicketsTaken'].rolling(3, min_periods=1).mean().values
        
        # Opposition strength (if team data available)
        # This would be enhanced with actual team strength data
        
        return features_df
    
    def train_player_performance_models(self, df: pd.DataFrame):
        """
        Train models for predicting individual player performance
        """
        logger.info("Training player performance models...")
        
        # Prepare features
        df_features = self.feature_engineering(df)
        
        # Define feature columns
        feature_cols = [
            'Role_Encoded', 'Team_Encoded', 'Opponent_Encoded', 'Match_Number',
            'Avg_Runs', 'Std_Runs', 'Max_Runs', 'Min_Runs',
            'Avg_Wickets', 'Std_Wickets', 'Max_Wickets', 'Total_Matches',
            'Recent_Form_Runs', 'Recent_Form_Wickets'
        ]
        
        # Fill NaN values
        df_features[feature_cols] = df_features[feature_cols].fillna(0)
        
        # Store feature columns
        self.feature_columns['player_performance'] = feature_cols
        
        X = df_features[feature_cols]
        
        # Train runs prediction model (for batsmen)
        batsmen_data = df_features[df_features['Runs'] > 0]
        if len(batsmen_data) > 10:
            X_bat = batsmen_data[feature_cols]
            y_runs = batsmen_data['Runs']
            
            # Scale features
            self.scalers['runs'] = StandardScaler()
            X_bat_scaled = self.scalers['runs'].fit_transform(X_bat)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_bat_scaled, y_runs, test_size=0.2, random_state=42
            )
            
            # Train Linear Regression
            self.models['runs_linear'] = LinearRegression()
            self.models['runs_linear'].fit(X_train, y_train)
            
            # Train Random Forest
            self.models['runs_rf'] = RandomForestRegressor(n_estimators=100, random_state=42)
            self.models['runs_rf'].fit(X_train, y_train)
            
            # Evaluate
            y_pred_lr = self.models['runs_linear'].predict(X_test)
            y_pred_rf = self.models['runs_rf'].predict(X_test)
            
            logger.info(f"Runs prediction - Linear Regression RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.2f}")
            logger.info(f"Runs prediction - Random Forest RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.2f}")
        
        # Train wickets prediction model (for bowlers)
        bowler_data = df_features[df_features['WicketsTaken'] > 0]
        if len(bowler_data) > 10:
            X_bowl = bowler_data[feature_cols]
            y_wickets = bowler_data['WicketsTaken']
            
            # Scale features
            self.scalers['wickets'] = StandardScaler()
            X_bowl_scaled = self.scalers['wickets'].fit_transform(X_bowl)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_bowl_scaled, y_wickets, test_size=0.2, random_state=42
            )
            
            # Train Linear Regression
            self.models['wickets_linear'] = LinearRegression()
            self.models['wickets_linear'].fit(X_train, y_train)
            
            # Train Random Forest
            self.models['wickets_rf'] = RandomForestRegressor(n_estimators=100, random_state=42)
            self.models['wickets_rf'].fit(X_train, y_train)
            
            # Evaluate
            y_pred_lr = self.models['wickets_linear'].predict(X_test)
            y_pred_rf = self.models['wickets_rf'].predict(X_test)
            
            logger.info(f"Wickets prediction - Linear Regression RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_lr)):.2f}")
            logger.info(f"Wickets prediction - Random Forest RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.2f}")
    
    def train_match_outcome_model(self, df_team: pd.DataFrame, df_players: pd.DataFrame):
        """
        Train model for predicting match outcomes
        """
        logger.info("Training match outcome prediction model...")
        
        if df_team.empty:
            logger.warning("No team data available for match outcome prediction")
            return
        
        # Create synthetic match data based on team strength and player performance
        matches = []
        teams = df_team['Team'].unique()
        
        for team1 in teams:
            for team2 in teams:
                if team1 != team2:
                    # Get team stats
                    team1_stats = df_team[df_team['Team'] == team1].iloc[0]
                    team2_stats = df_team[df_team['Team'] == team2].iloc[0]
                    
                    # Get player performance for teams
                    team1_players = df_players[df_players['Team'] == team1]
                    team2_players = df_players[df_players['Team'] == team2]
                    
                    match_features = {
                        'team1_win_rate': team1_stats.get('Win_Rate', 0),
                        'team2_win_rate': team2_stats.get('Win_Rate', 0),
                        'team1_nrr': team1_stats.get('Net Run Rate', 0),
                        'team2_nrr': team2_stats.get('Net Run Rate', 0),
                        'team1_avg_runs': team1_players['Runs'].mean() if not team1_players.empty else 0,
                        'team2_avg_runs': team2_players['Runs'].mean() if not team2_players.empty else 0,
                        'team1_avg_wickets': team1_players['WicketsTaken'].mean() if not team1_players.empty else 0,
                        'team2_avg_wickets': team2_players['WicketsTaken'].mean() if not team2_players.empty else 0,
                    }
                    
                    # Determine winner (simple heuristic for training)
                    team1_strength = (match_features['team1_win_rate'] * 0.4 + 
                                    (match_features['team1_nrr'] / 2) * 0.3 + 
                                    (match_features['team1_avg_runs'] / 100) * 0.3)
                    team2_strength = (match_features['team2_win_rate'] * 0.4 + 
                                    (match_features['team2_nrr'] / 2) * 0.3 + 
                                    (match_features['team2_avg_runs'] / 100) * 0.3)
                    
                    match_features['winner'] = 1 if team1_strength > team2_strength else 0
                    matches.append(match_features)
        
        if matches:
            match_df = pd.DataFrame(matches)
            
            feature_cols = [
                'team1_win_rate', 'team2_win_rate', 'team1_nrr', 'team2_nrr',
                'team1_avg_runs', 'team2_avg_runs', 'team1_avg_wickets', 'team2_avg_wickets'
            ]
            
            X = match_df[feature_cols].fillna(0)
            y = match_df['winner']
            
            # Scale features
            self.scalers['match_outcome'] = StandardScaler()
            X_scaled = self.scalers['match_outcome'].fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )
            
            # Train Logistic Regression
            self.models['match_outcome_logistic'] = LogisticRegression(random_state=42)
            self.models['match_outcome_logistic'].fit(X_train, y_train)
            
            # Train Random Forest Classifier
            self.models['match_outcome_rf'] = RandomForestClassifier(n_estimators=100, random_state=42)
            self.models['match_outcome_rf'].fit(X_train, y_train)
            
            # Evaluate
            y_pred_lr = self.models['match_outcome_logistic'].predict(X_test)
            y_pred_rf = self.models['match_outcome_rf'].predict(X_test)
            
            logger.info(f"Match outcome - Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr):.3f}")
            logger.info(f"Match outcome - Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.3f}")
            
            self.feature_columns['match_outcome'] = feature_cols
    
    def train_all_models(self, batsmen_file: str, team_file: str):
        """
        Train all ML models
        """
        logger.info("Starting ML model training...")
        
        # Load data
        df_players, df_team = self.load_and_prepare_data(batsmen_file, team_file)
        
        # Train models
        self.train_player_performance_models(df_players)
        self.train_match_outcome_model(df_team, df_players)
        
        self.is_trained = True
        logger.info("All models trained successfully!")
    
    def predict_player_performance(self, player: str, opponent: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict player performance using ML models
        """
        if not self.is_trained:
            return {"error": "Models not trained yet"}
        
        try:
            # Get player data
            player_data = df[df['Player'] == player]
            if player_data.empty:
                return {"error": f"No data found for player {player}"}
            
            # Prepare features
            df_features = self.feature_engineering(df)
            player_features = df_features[df_features['Player'] == player].iloc[-1]  # Latest data
            
            # Update opponent
            if opponent in self.encoders['Opponent'].classes_:
                player_features['Opponent_Encoded'] = self.encoders['Opponent'].transform([opponent])[0]
            
            feature_cols = self.feature_columns['player_performance']
            X = player_features[feature_cols].values.reshape(1, -1)
            X = np.nan_to_num(X, 0)
            
            role = str(player_data['Role'].iloc[0])
            predictions = {}
            
            if "Bowler" in role:
                # Predict wickets
                if 'wickets_linear' in self.models and 'wickets' in self.scalers:
                    X_scaled = self.scalers['wickets'].transform(X)
                    pred_lr = self.models['wickets_linear'].predict(X_scaled)[0]
                    pred_rf = self.models['wickets_rf'].predict(X_scaled)[0]
                    
                    predictions['wickets_linear'] = max(0, round(pred_lr, 1))
                    predictions['wickets_rf'] = max(0, round(pred_rf, 1))
                    predictions['wickets_avg'] = round((pred_lr + pred_rf) / 2, 1)
            else:
                # Predict runs
                if 'runs_linear' in self.models and 'runs' in self.scalers:
                    X_scaled = self.scalers['runs'].transform(X)
                    pred_lr = self.models['runs_linear'].predict(X_scaled)[0]
                    pred_rf = self.models['runs_rf'].predict(X_scaled)[0]
                    
                    predictions['runs_linear'] = max(0, round(pred_lr, 1))
                    predictions['runs_rf'] = max(0, round(pred_rf, 1))
                    predictions['runs_avg'] = round((pred_lr + pred_rf) / 2, 1)
            
            predictions['role'] = role
            predictions['player'] = player
            predictions['opponent'] = opponent
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in player prediction: {e}")
            return {"error": str(e)}
    
    def predict_match_outcome(self, team1: str, team2: str, df_team: pd.DataFrame, df_players: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict match outcome using ML models
        """
        if not self.is_trained or 'match_outcome_logistic' not in self.models:
            return {"error": "Match outcome model not trained"}
        
        try:
            # Get team stats
            team1_stats = df_team[df_team['Team'] == team1]
            team2_stats = df_team[df_team['Team'] == team2]
            
            if team1_stats.empty or team2_stats.empty:
                return {"error": "Team data not found"}
            
            team1_stats = team1_stats.iloc[0]
            team2_stats = team2_stats.iloc[0]
            
            # Get player stats
            team1_players = df_players[df_players['Team'] == team1]
            team2_players = df_players[df_players['Team'] == team2]
            
            # Prepare features
            features = np.array([[
                team1_stats.get('Win_Rate', 0),
                team2_stats.get('Win_Rate', 0),
                team1_stats.get('Net Run Rate', 0),
                team2_stats.get('Net Run Rate', 0),
                team1_players['Runs'].mean() if not team1_players.empty else 0,
                team2_players['Runs'].mean() if not team2_players.empty else 0,
                team1_players['WicketsTaken'].mean() if not team1_players.empty else 0,
                team2_players['WicketsTaken'].mean() if not team2_players.empty else 0,
            ]])
            
            features = np.nan_to_num(features, 0)
            
            # Scale features
            X_scaled = self.scalers['match_outcome'].transform(features)
            
            # Predict
            pred_lr = self.models['match_outcome_logistic'].predict_proba(X_scaled)[0]
            pred_rf = self.models['match_outcome_rf'].predict_proba(X_scaled)[0]
            
            # Average predictions
            avg_pred = (pred_lr + pred_rf) / 2
            
            return {
                'team1': team1,
                'team2': team2,
                'team1_win_prob': round(avg_pred[1] * 100, 2),
                'team2_win_prob': round(avg_pred[0] * 100, 2),
                'predicted_winner': team1 if avg_pred[1] > avg_pred[0] else team2,
                'confidence': round(abs(avg_pred[1] - avg_pred[0]) * 100, 2),
                'logistic_pred': {'team1': round(pred_lr[1] * 100, 2), 'team2': round(pred_lr[0] * 100, 2)},
                'rf_pred': {'team1': round(pred_rf[1] * 100, 2), 'team2': round(pred_rf[0] * 100, 2)}
            }
            
        except Exception as e:
            logger.error(f"Error in match prediction: {e}")
            return {"error": str(e)}
    
    def save_models(self, filepath: str = "cricket_ml_models.joblib"):
        """Save trained models to file"""
        if self.is_trained:
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'encoders': self.encoders,
                'feature_columns': self.feature_columns,
                'is_trained': self.is_trained
            }
            joblib.dump(model_data, filepath)
            logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str = "cricket_ml_models.joblib"):
        """Load trained models from file"""
        if os.path.exists(filepath):
            model_data = joblib.load(filepath)
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.encoders = model_data['encoders']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = model_data['is_trained']
            logger.info(f"Models loaded from {filepath}")
        else:
            logger.warning(f"Model file {filepath} not found")

# Global ML predictor instance
ml_predictor = CricketMLPredictor()