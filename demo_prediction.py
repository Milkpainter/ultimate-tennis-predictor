#!/usr/bin/env python3
"""
Ultimate Tennis Predictor - Runnable Demo

This script demonstrates the tennis prediction algorithm with mock data.
It shows how the system would work with real data integration.

Usage:
    python demo_prediction.py
    
Requirements:
    pip install numpy pandas scikit-learn xgboost
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb

# Mock the complex classes for demo purposes
class PlayerProfile:
    def __init__(self, name, ranking=50, age=25, elo_rating=1600):
        self.name = name
        self.current_ranking = ranking
        self.age = age
        self.height = 180 + np.random.randint(-15, 15)
        self.weight = 75 + np.random.randint(-10, 10)
        self.handed = np.random.choice(['L', 'R'], p=[0.1, 0.9])
        self.elo_rating = elo_rating
        self.career_wins = max(50, ranking * 2 + np.random.randint(0, 100))
        self.career_losses = int(self.career_wins * 0.3) + np.random.randint(0, 50)
        self.fitness_score = 0.7 + np.random.random() * 0.3
        self.mental_toughness_score = 0.3 + np.random.random() * 0.7
        self.pressure_performance_rating = 0.3 + np.random.random() * 0.7
        
        # Surface stats (mock based on ranking)
        base_matches = 200 + np.random.randint(0, 100)
        self.clay_wins = int(base_matches * (0.4 + np.random.random() * 0.4))
        self.clay_losses = int(base_matches * 0.3)
        self.grass_wins = int(base_matches * (0.3 + np.random.random() * 0.4))
        self.grass_losses = int(base_matches * 0.25)
        self.hard_wins = int(base_matches * (0.4 + np.random.random() * 0.4))
        self.hard_losses = int(base_matches * 0.3)

class DemoTennisPredictor:
    """
    Simplified but functional version of the tennis predictor
    Demonstrates core concepts with actual ML models
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        self._initialize_models()
        
        # Mock ELO system
        self.player_ratings = {}
        
        print("🎾 Demo Tennis Predictor Initialized!")
        print("📊 Models loaded: XGBoost, Random Forest, Logistic Regression")
    
    def _initialize_models(self):
        """Initialize ML models"""
        self.models = {
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            'logistic': LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        }
        
        # Meta-model for ensemble
        self.meta_model = RandomForestClassifier(
            n_estimators=50,
            random_state=42
        )
    
    def create_features(self, player1: PlayerProfile, player2: PlayerProfile, 
                       surface='hard', tournament_level='ATP 500'):
        """Create feature vector for prediction"""
        features = {}
        
        # Basic comparative features
        features['rank_diff'] = player1.current_ranking - player2.current_ranking
        features['rank_ratio'] = player1.current_ranking / max(player2.current_ranking, 1)
        features['age_diff'] = player1.age - player2.age
        features['height_diff'] = player1.height - player2.height
        features['weight_diff'] = player1.weight - player2.weight
        
        # Career statistics
        p1_total = player1.career_wins + player1.career_losses
        p2_total = player2.career_wins + player2.career_losses
        features['p1_win_pct'] = player1.career_wins / max(p1_total, 1)
        features['p2_win_pct'] = player2.career_wins / max(p2_total, 1)
        features['win_pct_diff'] = features['p1_win_pct'] - features['p2_win_pct']
        
        # Surface-specific features
        surface_stats = self._get_surface_stats(player1, player2, surface)
        features.update(surface_stats)
        
        # ELO ratings
        features['elo_diff'] = player1.elo_rating - player2.elo_rating
        features['elo_ratio'] = player1.elo_rating / max(player2.elo_rating, 1)
        
        # Physical and mental features
        features['fitness_diff'] = player1.fitness_score - player2.fitness_score
        features['mental_toughness_diff'] = player1.mental_toughness_score - player2.mental_toughness_score
        features['pressure_rating_diff'] = player1.pressure_performance_rating - player2.pressure_performance_rating
        
        # Hand matchup
        features['same_handed'] = int(player1.handed == player2.handed)
        features['p1_lefty'] = int(player1.handed == 'L')
        features['p2_lefty'] = int(player2.handed == 'L')
        
        # Tournament context
        features['is_grand_slam'] = int(tournament_level == 'Grand Slam')
        features['is_masters'] = int(tournament_level == 'Masters')
        features['is_hard_court'] = int(surface == 'hard')
        features['is_clay_court'] = int(surface == 'clay')
        features['is_grass_court'] = int(surface == 'grass')
        
        return features
    
    def _get_surface_stats(self, p1: PlayerProfile, p2: PlayerProfile, surface: str):
        """Get surface-specific statistics"""
        if surface == 'clay':
            p1_wins, p1_losses = p1.clay_wins, p1.clay_losses
            p2_wins, p2_losses = p2.clay_wins, p2.clay_losses
        elif surface == 'grass':
            p1_wins, p1_losses = p1.grass_wins, p1.grass_losses
            p2_wins, p2_losses = p2.grass_wins, p2.grass_losses
        else:  # hard court
            p1_wins, p1_losses = p1.hard_wins, p1.hard_losses
            p2_wins, p2_losses = p2.hard_wins, p2.hard_losses
        
        p1_total = p1_wins + p1_losses
        p2_total = p2_wins + p2_losses
        
        return {
            'p1_surface_win_pct': p1_wins / max(p1_total, 1),
            'p2_surface_win_pct': p2_wins / max(p2_total, 1),
            'surface_win_pct_diff': (p1_wins / max(p1_total, 1)) - (p2_wins / max(p2_total, 1)),
            'p1_surface_matches': p1_total,
            'p2_surface_matches': p2_total
        }
    
    def generate_training_data(self, n_matches=1000):
        """Generate synthetic training data for demonstration"""
        print(f"🔄 Generating {n_matches} synthetic matches for training...")
        
        X_list = []
        y_list = []
        
        # Create pool of players
        player_names = [
            "Djokovic", "Alcaraz", "Medvedev", "Zverev", "Rublev", "Tsitsipas",
            "Nadal", "Federer", "Murray", "Wawrinka", "Dimitrov", "Shapovalov",
            "Berrettini", "Hurkacz", "Sinner", "Ruud", "Fritz", "Tiafoe",
            "Norrie", "Khachanov", "Kyrgios", "Raonic", "Isner", "Opelka"
        ]
        
        surfaces = ['hard', 'clay', 'grass']
        tournaments = ['Grand Slam', 'Masters', 'ATP 500', 'ATP 250']
        
        for i in range(n_matches):
            # Create two random players
            p1_name, p2_name = np.random.choice(player_names, 2, replace=False)
            
            # Create player profiles with some correlation to outcome
            p1_ranking = np.random.randint(1, 100)
            p2_ranking = np.random.randint(1, 100)
            
            player1 = PlayerProfile(p1_name, p1_ranking, 
                                  age=20 + np.random.randint(0, 20),
                                  elo_rating=1400 + (100 - p1_ranking) * 3)
            player2 = PlayerProfile(p2_name, p2_ranking,
                                  age=20 + np.random.randint(0, 20), 
                                  elo_rating=1400 + (100 - p2_ranking) * 3)
            
            surface = np.random.choice(surfaces)
            tournament = np.random.choice(tournaments)
            
            # Create features
            features = self.create_features(player1, player2, surface, tournament)
            
            # Determine winner based on features (simulate realistic outcomes)
            # Higher ranked player (lower number) more likely to win
            ranking_advantage = (p2_ranking - p1_ranking) / 100
            elo_advantage = (player1.elo_rating - player2.elo_rating) / 400
            surface_advantage = features['surface_win_pct_diff'] * 0.5
            fitness_advantage = features['fitness_diff'] * 0.3
            
            total_advantage = ranking_advantage + elo_advantage + surface_advantage + fitness_advantage
            win_probability = 1 / (1 + np.exp(-total_advantage * 2))  # Sigmoid function
            
            winner = 1 if np.random.random() < win_probability else 0
            
            X_list.append(features)
            y_list.append(winner)
        
        # Convert to DataFrame and numpy array
        X_df = pd.DataFrame(X_list)
        X_df = X_df.fillna(0)  # Handle any NaN values
        y = np.array(y_list)
        
        print(f"✅ Generated {len(X_df)} training samples with {len(X_df.columns)} features")
        print(f"📊 Win distribution: P1 wins: {y.sum()}, P2 wins: {len(y) - y.sum()}")
        
        return X_df, y
    
    def train(self, X, y):
        """Train the ensemble model"""
        print("🚀 Training ensemble models...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train individual models
        model_predictions = np.zeros((len(X), len(self.models)))
        
        for i, (name, model) in enumerate(self.models.items()):
            print(f"   Training {name}...")
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_scaled, y, cv=3, scoring='accuracy')
            print(f"   {name} CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            
            # Fit on full data
            model.fit(X_scaled, y)
            
            # Get predictions for meta-model training
            if hasattr(model, 'predict_proba'):
                model_predictions[:, i] = model.predict_proba(X_scaled)[:, 1]
            else:
                model_predictions[:, i] = model.predict(X_scaled)
        
        # Train meta-model
        print("   Training meta-model...")
        self.meta_model.fit(model_predictions, y)
        
        # Evaluate ensemble
        ensemble_pred = self.meta_model.predict_proba(model_predictions)[:, 1]
        ensemble_accuracy = accuracy_score(y, ensemble_pred > 0.5)
        ensemble_logloss = log_loss(y, ensemble_pred)
        
        print(f"\n🎯 Ensemble Results:")
        print(f"   Accuracy: {ensemble_accuracy:.1%}")
        print(f"   Log Loss: {ensemble_logloss:.3f}")
        
        self.is_trained = True
        print("✅ Training completed!")
    
    def predict(self, player1: PlayerProfile, player2: PlayerProfile,
               surface='hard', tournament_level='ATP 500'):
        """Make prediction for a match"""
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        
        # Create features
        features = self.create_features(player1, player2, surface, tournament_level)
        feature_df = pd.DataFrame([features])
        
        # Scale features
        X_scaled = self.scaler.transform(feature_df)
        
        # Get predictions from all models
        model_predictions = []
        individual_preds = {}
        
        for name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X_scaled)[0, 1]
            else:
                pred = model.predict(X_scaled)[0]
            model_predictions.append(pred)
            individual_preds[name] = pred
        
        # Ensemble prediction
        meta_input = np.array(model_predictions).reshape(1, -1)
        ensemble_prob = self.meta_model.predict_proba(meta_input)[0, 1]
        
        # Calculate confidence
        pred_std = np.std(model_predictions)
        confidence = max(0, 1 - pred_std * 2)  # Higher std = lower confidence
        
        return {
            'player1': player1.name,
            'player2': player2.name,
            'surface': surface,
            'tournament_level': tournament_level,
            'probability_p1_wins': ensemble_prob,
            'probability_p2_wins': 1 - ensemble_prob,
            'confidence': confidence,
            'predicted_winner': player1.name if ensemble_prob > 0.5 else player2.name,
            'individual_predictions': individual_preds,
            'model_agreement': 1 - pred_std
        }

def run_demo():
    """Run the tennis prediction demo"""
    print("🎾 ULTIMATE TENNIS PREDICTOR - DEMO VERSION")
    print("=" * 50)
    
    # Initialize predictor
    predictor = DemoTennisPredictor()
    
    # Generate training data
    X, y = predictor.generate_training_data(n_matches=2000)
    
    # Train models
    predictor.train(X, y)
    
    print("\n" + "=" * 50)
    print("🏆 MAKING PREDICTIONS")
    print("=" * 50)
    
    # Demo predictions
    test_matches = [
        {
            'p1': PlayerProfile("Novak Djokovic", ranking=1, age=36, elo_rating=2000),
            'p2': PlayerProfile("Carlos Alcaraz", ranking=2, age=20, elo_rating=1950),
            'surface': 'hard',
            'tournament': 'Grand Slam'
        },
        {
            'p1': PlayerProfile("Rafael Nadal", ranking=10, age=37, elo_rating=1800),
            'p2': PlayerProfile("Daniil Medvedev", ranking=3, age=27, elo_rating=1900),
            'surface': 'clay',
            'tournament': 'Masters'
        },
        {
            'p1': PlayerProfile("Roger Federer", ranking=25, age=42, elo_rating=1700),
            'p2': PlayerProfile("Stefanos Tsitsipas", ranking=6, age=25, elo_rating=1750),
            'surface': 'grass',
            'tournament': 'ATP 500'
        }
    ]
    
    for i, match in enumerate(test_matches, 1):
        print(f"\n🎯 PREDICTION {i}:")
        print(f"Match: {match['p1'].name} vs {match['p2'].name}")
        print(f"Surface: {match['surface'].title()}, Tournament: {match['tournament']}")
        
        result = predictor.predict(
            match['p1'], match['p2'], 
            match['surface'], match['tournament']
        )
        
        print(f"\n📊 RESULT:")
        print(f"   Predicted Winner: {result['predicted_winner']}")
        print(f"   {result['player1']}: {result['probability_p1_wins']:.1%} chance")
        print(f"   {result['player2']}: {result['probability_p2_wins']:.1%} chance")
        print(f"   Confidence: {result['confidence']:.1%}")
        print(f"   Model Agreement: {result['model_agreement']:.1%}")
        
        print(f"\n🔍 Individual Model Predictions:")
        for model_name, pred in result['individual_predictions'].items():
            print(f"   {model_name.title()}: {pred:.1%}")
        
        print("-" * 50)
    
    print("\n✅ Demo completed successfully!")
    print("\n💡 This demonstrates the core concepts of the full system.")
    print("   In production, we would use:")
    print("   • Real historical match data (100,000+ matches)")
    print("   • 1000+ advanced features")
    print("   • Deep learning models (Transformers, Graph NNs)")
    print("   • Real-time data feeds")
    print("   • Advanced ELO rating system")
    print("   • Much more sophisticated infrastructure")

if __name__ == "__main__":
    try:
        run_demo()
    except Exception as e:
        print(f"❌ Error running demo: {str(e)}")
        print("\n💡 Make sure you have the required packages installed:")
        print("   pip install numpy pandas scikit-learn xgboost")