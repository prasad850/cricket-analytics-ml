# train_models.py
# Script to train and test the ML models

import sys
import os
sys.path.append(os.path.dirname(__file__))

from ml_models import ml_predictor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Train the ML models and test predictions
    """
    try:
        logger.info("🚀 Starting ML model training...")
        
        # File paths
        batsmen_file = "IPLBAT4.xlsx"
        team_file = "ipl.xlsx"
        
        # Check if files exist
        if not os.path.exists(batsmen_file):
            logger.error(f"❌ Batsmen file not found: {batsmen_file}")
            return
        if not os.path.exists(team_file):
            logger.error(f"❌ Team file not found: {team_file}")
            return
        
        # Train models
        ml_predictor.train_all_models(batsmen_file, team_file)
        
        # Save models
        ml_predictor.save_models()
        logger.info("💾 Models saved successfully!")
        
        # Test predictions
        logger.info("🧪 Testing predictions...")
        
        # Load data for testing
        df_players, df_team = ml_predictor.load_and_prepare_data(batsmen_file, team_file)
        
        # Test player prediction
        if not df_players.empty:
            test_player = df_players['Player'].iloc[0]
            test_opponent = df_players['Opponent'].iloc[0] if not df_players.empty else "MI"
            
            prediction = ml_predictor.predict_player_performance(test_player, test_opponent, df_players)
            logger.info(f"📊 Player prediction test for {test_player} vs {test_opponent}:")
            logger.info(f"   Result: {prediction}")
        
        # Test match outcome prediction
        if not df_team.empty and len(df_team) >= 2:
            team1 = df_team['Team'].iloc[0]
            team2 = df_team['Team'].iloc[1]
            
            match_prediction = ml_predictor.predict_match_outcome(team1, team2, df_team, df_players)
            logger.info(f"🏏 Match prediction test for {team1} vs {team2}:")
            logger.info(f"   Result: {match_prediction}")
        
        logger.info("✅ All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()