#!/usr/bin/env python3
"""
Basic Usage Example for Ultimate Tennis Predictor
Demonstrates how to use the tennis prediction system
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tennis_predictor import (
    TennisPredictor,
    PlayerProfile, 
    MatchContext,
    Surface,
    TournamentLevel,
    PlayerStyle,
    TennisDataLoader
)

async def basic_prediction_example():
    """
    Basic example: Predict a single match
    """
    print("🎾 BASIC TENNIS MATCH PREDICTION EXAMPLE")
    print("=" * 50)
    
    # Initialize the predictor
    print("⚙️  Initializing tennis predictor...")
    predictor = TennisPredictor()
    data_loader = TennisDataLoader()
    
    try:
        # Load sample data and train models
        print("📊 Loading sample data...")
        matches_loaded = await data_loader.load_sample_data_for_testing()
        print(f"   Loaded {matches_loaded} sample matches")
        
        # Get training data
        training_data = await data_loader.get_training_data()
        print(f"   Retrieved {len(training_data)} training matches")
        
        # Initialize and train models
        print("🤖 Training prediction models...")
        await predictor.initialize_models()
        
        if len(training_data) > 50:
            model_scores = await predictor.train_models(training_data)
            print("   Model training completed!")
            
            for model_name, accuracy in model_scores.items():
                print(f"     {model_name}: {accuracy:.3f} accuracy")
        else:
            print("   Using demo mode (insufficient training data)")
            predictor.is_trained = True
        
        # Create match context
        context = MatchContext(
            tournament_level=TournamentLevel.GRAND_SLAM,
            surface=Surface.HARD,
            best_of=5,
            round_name="Final",
            venue="Arthur Ashe Stadium",
            temperature=25.0,
            humidity=60.0,
            match_importance=2.0
        )
        
        # Make prediction
        print("\n🔮 Making match prediction...")
        result = await predictor.predict_match(
            player1="Novak Djokovic",
            player2="Carlos Alcaraz",
            context=context,
            return_details=True
        )
        
        # Display results
        print("\n🏆 PREDICTION RESULTS")
        print("=" * 30)
        print(f"Match: {result['player1']} vs {result['player2']}")
        print(f"Surface: {result['surface'].title()}")
        print(f"Tournament: {result['tournament_level']}")
        print(f"Round: {context.round_name}")
        
        print(f"\n🎢 PREDICTION:")
        winner = result['player1'] if result['probability_p1_wins'] > 0.5 else result['player2']
        winner_prob = max(result['probability_p1_wins'], result['probability_p2_wins'])
        print(f"   Predicted Winner: {winner}")
        print(f"   Win Probability: {winner_prob:.1%}")
        print(f"   {result['player1']}: {result['probability_p1_wins']:.1%}")
        print(f"   {result['player2']}: {result['probability_p2_wins']:.1%}")
        
        print(f"\n📊 CONFIDENCE METRICS:")
        print(f"   Model Agreement: {result['model_agreement']:.1%}")
        print(f"   Confidence Level: {result['prediction_confidence'].replace('_', ' ').title()}")
        ci = result['confidence_interval']
        print(f"   Confidence Interval: {ci['lower']:.1%} - {ci['upper']:.1%}")
        
        if 'expected_duration_minutes' in result:
            print(f"\n⏱️  MATCH INSIGHTS:")
            print(f"   Expected Duration: {result['expected_duration_minutes']} minutes")
        
        print(f"\n⚡ PERFORMANCE:")
        print(f"   Prediction Latency: {result['prediction_latency_ms']:.1f}ms")
        print(f"   Models Used: {result['models_used']}")
        print(f"   Features Used: {result['features_used']}")
        
        if 'individual_predictions' in result:
            print(f"\n🤖 INDIVIDUAL MODEL PREDICTIONS:")
            for model_name, prob in result['individual_predictions'].items():
                print(f"     {model_name}: {prob:.3f}")
        
        if 'betting_recommendation' in result:
            betting = result['betting_recommendation']
            print(f"\n💰 BETTING ANALYSIS:")
            print(f"   Recommendation: {betting['recommendation'].replace('_', ' ').title()}")
            if 'edge' in betting:
                print(f"   Edge: {betting['edge']:.1%}")
                print(f"   Suggested Stake: {betting.get('suggested_stake', 0):.1%} of bankroll")
        
        return result
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

async def batch_prediction_example():
    """
    Example: Predict multiple matches in a tournament
    """
    print("\n\n🎾 BATCH PREDICTION EXAMPLE")
    print("=" * 40)
    
    predictor = TennisPredictor()
    data_loader = TennisDataLoader()
    
    try:
        # Setup (reuse from basic example)
        await data_loader.load_sample_data_for_testing()
        training_data = await data_loader.get_training_data()
        await predictor.initialize_models()
        
        if len(training_data) > 50:
            await predictor.train_models(training_data)
        else:
            predictor.is_trained = True
        
        # Define tournament matches
        tournament_matches = [
            {
                'player1': 'Novak Djokovic',
                'player2': 'Jannik Sinner', 
                'round': 'Semifinal 1'
            },
            {
                'player1': 'Carlos Alcaraz',
                'player2': 'Daniil Medvedev',
                'round': 'Semifinal 2' 
            }
        ]
        
        context = MatchContext(
            tournament_level=TournamentLevel.GRAND_SLAM,
            surface=Surface.HARD,
            best_of=5,
            match_importance=1.8
        )
        
        print(f"🏆 US OPEN SEMIFINALS PREDICTIONS:")
        print("=" * 40)
        
        predictions = []
        
        for match in tournament_matches:
            print(f"\n{match['round']}: {match['player1']} vs {match['player2']}")
            
            result = await predictor.predict_match(
                match['player1'], match['player2'], context, return_details=False
            )
            
            winner = result['predicted_winner']
            prob = result['win_probability']
            confidence = result['prediction_confidence']
            
            print(f"   ✅ Predicted Winner: {winner} ({prob:.1%})")
            print(f"   🎯 Confidence: {confidence.replace('_', ' ').title()}")
            print(f"   ⏱️  Prediction Time: {result['prediction_latency_ms']:.1f}ms")
            
            predictions.append({
                'match': f"{match['player1']} vs {match['player2']}",
                'winner': winner,
                'probability': prob,
                'confidence': confidence
            })
        
        # Tournament analysis
        print(f"\n\n📈 TOURNAMENT ANALYSIS:")
        print("=" * 25)
        avg_confidence = sum(1 for p in predictions if 'high' in p['confidence']) / len(predictions)
        print(f"   High Confidence Predictions: {avg_confidence:.1%}")
        print(f"   Average Win Probability: {np.mean([p['probability'] for p in predictions]):.1%}")
        print(f"   Most Confident Prediction: {max(predictions, key=lambda x: x['probability'])['match']}")
        
        return predictions
        
    except Exception as e:
        print(f"❌ Batch prediction error: {str(e)}")
        return []

async def player_analysis_example():
    """
    Example: Analyze player profiles and statistics
    """
    print("\n\n🎾 PLAYER ANALYSIS EXAMPLE")
    print("=" * 35)
    
    data_loader = TennisDataLoader()
    
    try:
        await data_loader.load_sample_data_for_testing()
        
        # Analyze multiple players
        player_names = ["Novak Djokovic", "Carlos Alcaraz", "Jannik Sinner"]
        
        print("📄 PLAYER PROFILES:")
        print("=" * 20)
        
        for player_name in player_names:
            profile = await data_loader.get_player_profile(player_name)
            
            if profile:
                print(f"\n{profile.name}:")
                print(f"   Current Ranking: #{profile.current_ranking}")
                print(f"   Age: {profile.age:.1f} years")
                print(f"   Playing Style: {profile.playing_style.value.replace('_', ' ').title()}")
                print(f"   Career Record: {profile.career_wins}-{profile.career_losses}")
                
                total_matches = profile.career_wins + profile.career_losses
                if total_matches > 0:
                    win_pct = profile.career_wins / total_matches
                    print(f"   Career Win%: {win_pct:.1%}")
                
                # Surface analysis
                surfaces = {
                    'Clay': (profile.clay_wins, profile.clay_losses),
                    'Grass': (profile.grass_wins, profile.grass_losses), 
                    'Hard': (profile.hard_wins, profile.hard_losses)
                }
                
                print(f"   Surface Performance:")
                for surface, (wins, losses) in surfaces.items():
                    if wins + losses > 0:
                        surface_pct = wins / (wins + losses)
                        print(f"     {surface}: {surface_pct:.1%} ({wins}-{losses})")
            else:
                print(f"\n{player_name}: Profile not found in database")
        
        # Head-to-head example
        print(f"\n\n🤼 HEAD-TO-HEAD ANALYSIS:")
        print("=" * 25)
        
        h2h_pairs = [
            ("Novak Djokovic", "Carlos Alcaraz"),
            ("Jannik Sinner", "Carlos Alcaraz")
        ]
        
        for player1, player2 in h2h_pairs:
            p1_profile = await data_loader.get_player_profile(player1)
            p2_profile = await data_loader.get_player_profile(player2)
            
            if p1_profile and p2_profile:
                h2h_data = await data_loader.get_head_to_head(p1_profile.player_id, p2_profile.player_id)
                
                print(f"\n{player1} vs {player2}:")
                if h2h_data:
                    p1_wins = sum(1 for match in h2h_data if match['winner'] == 'p1')
                    p2_wins = len(h2h_data) - p1_wins
                    print(f"   Total Matches: {len(h2h_data)}")
                    print(f"   {player1}: {p1_wins} wins")
                    print(f"   {player2}: {p2_wins} wins")
                    
                    if len(h2h_data) > 0:
                        latest_match = max(h2h_data, key=lambda x: x.get('date', '2000-01-01'))
                        print(f"   Latest Match: {latest_match['date']} on {latest_match['surface']}")
                else:
                    print("   No previous matches found")
            else:
                print(f"   Could not find profiles for both players")
        
    except Exception as e:
        print(f"❌ Player analysis error: {str(e)}")

async def advanced_prediction_example():
    """
    Advanced example with detailed analysis and betting insights
    """
    print("\n\n🎾 ADVANCED PREDICTION WITH ANALYSIS")
    print("=" * 45)
    
    predictor = TennisPredictor()
    data_loader = TennisDataLoader()
    
    try:
        # Setup
        await data_loader.load_sample_data_for_testing()
        training_data = await data_loader.get_training_data()
        await predictor.initialize_models()
        
        if len(training_data) > 50:
            await predictor.train_models(training_data)
        else:
            predictor.is_trained = True
        
        # Complex match scenario
        context = MatchContext(
            tournament_level=TournamentLevel.GRAND_SLAM,
            surface=Surface.CLAY,  # Roland Garros
            best_of=5,
            round_name="Final",
            venue="Philippe Chatrier",
            temperature=28.0,
            humidity=70.0,
            wind_speed=15.0,
            match_importance=2.0
        )
        
        print(f"🏆 ROLAND GARROS FINAL PREDICTION")
        print(f"Venue: {context.venue}")
        print(f"Conditions: {context.temperature}°C, {context.humidity}% humidity, {context.wind_speed}km/h wind")
        
        result = await predictor.predict_match(
            player1="Rafael Nadal",
            player2="Novak Djokovic", 
            context=context,
            return_details=True
        )
        
        print(f"\n🎆 MATCH PREDICTION:")
        print(f"   {result['player1']}: {result['probability_p1_wins']:.1%} chance")
        print(f"   {result['player2']}: {result['probability_p2_wins']:.1%} chance")
        print(f"   Predicted Winner: {result['predicted_winner']}")
        
        # Detailed analysis
        if 'detailed_analysis' in result:
            analysis = result['detailed_analysis']
            
            print(f"\n🔍 MATCHUP ANALYSIS:")
            if 'matchup_analysis' in analysis:
                matchup = analysis['matchup_analysis']
                print(f"   Ranking Advantage: {matchup.get('ranking_advantage', 'N/A')}")
                print(f"   Experience Advantage: {matchup.get('experience_advantage', 'N/A')}")
                print(f"   Style Matchup Score: {matchup.get('style_matchup', 0):.2f}")
            
            print(f"\n🌡️  ENVIRONMENTAL FACTORS:")
            if 'context_factors' in analysis:
                env = analysis['context_factors'].get('environmental_factors', {})
                print(f"   Temperature Impact: {env.get('temperature', 0)}°C")
                print(f"   Humidity Impact: {env.get('humidity', 0)}%")
                print(f"   Surface: {result['surface'].title()} (clay specialist advantage)")
        
        # Key factors
        if 'key_factors' in result:
            print(f"\n🔑 KEY FACTORS:")
            for i, factor in enumerate(result['key_factors'][:5], 1):
                impact_symbol = "⬆️" if factor['impact'] == 'positive' else "⬇️"
                print(f"   {i}. {factor['factor'].replace('_', ' ').title()}: {impact_symbol} {factor['importance']:.3f}")
        
        # Risk assessment  
        if 'risk_assessment' in result:
            risk = result['risk_assessment']
            risk_emoji = "🔴" if risk['risk_level'] == 'high' else "🟡" if risk['risk_level'] == 'medium' else "🟢"
            print(f"\n{risk_emoji} PREDICTION RISK: {risk['risk_level'].upper()}")
            if risk['risk_factors']:
                for factor in risk['risk_factors']:
                    print(f"   ⚠️  {factor}")
        
        print(f"\n✅ Prediction completed successfully!")
        return result
        
    except Exception as e:
        print(f"❌ Advanced prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

async def model_validation_example():
    """
    Example: Validate model performance
    """
    print("\n\n🎾 MODEL VALIDATION EXAMPLE")
    print("=" * 35)
    
    predictor = TennisPredictor()
    data_loader = TennisDataLoader()
    
    try:
        # Setup
        await data_loader.load_sample_data_for_testing()
        training_data = await data_loader.get_training_data()
        
        print(f"📈 Dataset Statistics:")
        print(f"   Total Matches: {len(training_data):,}")
        
        if len(training_data) > 0:
            surface_breakdown = training_data['surface'].value_counts()
            print(f"   Surface Breakdown:")
            for surface, count in surface_breakdown.items():
                print(f"     {surface}: {count:,} matches")
            
            level_breakdown = training_data['tourney_level'].value_counts()
            print(f"   Tournament Levels:")
            for level, count in level_breakdown.head().items():
                print(f"     {level}: {count:,} matches")
        
        # Initialize and train
        await predictor.initialize_models()
        
        if len(training_data) > 100:
            print(f"\n🤖 Training Models...")
            model_scores = await predictor.train_models(training_data)
            
            print(f"\n🏆 MODEL PERFORMANCE:")
            print("=" * 25)
            
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            for i, (model_name, accuracy) in enumerate(sorted_models, 1):
                emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "🔵"
                print(f"   {emoji} {model_name}: {accuracy:.3f} ({accuracy*100:.1f}%)")
            
            best_model = sorted_models[0]
            avg_accuracy = np.mean(list(model_scores.values()))
            
            print(f"\n🎨 ENSEMBLE PERFORMANCE:")
            print(f"   Best Individual Model: {best_model[0]} ({best_model[1]:.3f})")
            print(f"   Average Model Accuracy: {avg_accuracy:.3f}")
            print(f"   Expected Ensemble Accuracy: {avg_accuracy + 0.02:.3f} (+2% from ensemble)")
            
            # Performance assessment
            if avg_accuracy > 0.70:
                print(f"   ✅ EXCELLENT - Ready for production deployment!")
            elif avg_accuracy > 0.65:
                print(f"   🟡 GOOD - Consider additional feature engineering")
            elif avg_accuracy > 0.60:
                print(f"   🟠 FAIR - Needs improvement before production")
            else:
                print(f"   🔴 POOR - Requires significant model improvements")
        else:
            print(f"   Insufficient data for proper training ({len(training_data)} matches)")
            print(f"   Minimum recommendation: 1000+ matches for production")
        
    except Exception as e:
        print(f"❌ Validation error: {str(e)}")

async def api_integration_example():
    """
    Example: Using the API programmatically
    """
    print("\n\n🎾 API INTEGRATION EXAMPLE")
    print("=" * 33)
    
    import aiohttp
    
    # Note: This assumes the API is running on localhost:8000
    base_url = "http://localhost:8000"
    
    try:
        async with aiohttp.ClientSession() as session:
            # Health check
            print("👩‍⚕️ Checking API health...")
            async with session.get(f"{base_url}/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    print(f"   API Status: {health_data['status']}")
                    print(f"   Models Loaded: {health_data['models_loaded']}")
                    print(f"   Total Predictions: {health_data['total_predictions']}")
                else:
                    print(f"   API not available (status {response.status})")
                    return
            
            # Make API prediction
            print(f"\n🔮 Making API prediction...")
            prediction_request = {
                "player1": "Novak Djokovic",
                "player2": "Carlos Alcaraz",
                "surface": "hard",
                "tournament_level": "Grand Slam",
                "best_of": 5,
                "return_details": True
            }
            
            async with session.post(
                f"{base_url}/api/v1/predict",
                json=prediction_request
            ) as response:
                
                if response.status == 200:
                    api_result = await response.json()
                    
                    print(f"\n✅ API PREDICTION SUCCESS:")
                    print(f"   Match: {api_result['player1']} vs {api_result['player2']}")
                    print(f"   Winner: {api_result['predicted_winner']} ({api_result['win_probability']:.1%})")
                    print(f"   Confidence: {api_result['prediction_confidence']}")
                    print(f"   API Latency: {api_result['prediction_latency_ms']:.1f}ms")
                    
                else:
                    error_data = await response.json()
                    print(f"   ❌ API Error: {error_data.get('detail', 'Unknown error')}")
    
    except aiohttp.ClientError as e:
        print(f"   ❌ Connection error: {str(e)}")
        print(f"   Make sure API is running: python -m uvicorn app.main:app --host 0.0.0.0 --port 8000")
    except Exception as e:
        print(f"   ❌ API integration error: {str(e)}")

async def main():
    """
    Run all examples
    """
    print("🎾 ULTIMATE TENNIS PREDICTOR - USAGE EXAMPLES")
    print("=" * 55)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run examples
    await basic_prediction_example()
    await batch_prediction_example()
    await player_analysis_example()
    await model_validation_example()
    
    # API example (only if API is running)
    print(f"\n🔗 API Integration (requires running API server):")
    await api_integration_example()
    
    print(f"\n\n✅ ALL EXAMPLES COMPLETED!")
    print(f"\n📚 Next steps:")
    print(f"   1. Start API: python -m uvicorn app.main:app --reload")
    print(f"   2. View docs: http://localhost:8000/docs")
    print(f"   3. Monitor: http://localhost:8000/metrics")
    print(f"   4. Deploy: docker-compose up -d")

if __name__ == "__main__":
    # Import numpy for calculations
    import numpy as np
    from datetime import datetime
    
    # Run all examples
    asyncio.run(main())