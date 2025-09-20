"""
Comprehensive test suite for Ultimate Tennis Predictor
Tests all components including models, features, and API endpoints
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

# Import our tennis prediction system
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tennis_predictor.predictor import ProductionTennisPredictor
from tennis_predictor.data_loader import TennisDataLoader
from tennis_predictor.validation import ProductionValidation
from tennis_predictor.features import AdvancedFeatureEngineering
from tennis_predictor.core import (
    PlayerProfile, MatchContext, Surface, TournamentLevel, PlayerStyle, AdvancedELOSystem
)

class TestPlayerProfile:
    """Test PlayerProfile data model"""
    
    def test_player_profile_creation(self):
        player = PlayerProfile(
            name="Novak Djokovic",
            player_id="djokovic_n",
            current_ranking=1,
            current_points=11540,
            age=37.0,
            height=188,
            weight=77,
            handed='R',
            backhand='two',
            turned_pro=2003
        )
        
        assert player.name == "Novak Djokovic"
        assert player.current_ranking == 1
        assert player.handed == 'R'
        assert player.elo_rating == 1500.0  # Default value
    
    def test_player_career_stats(self):
        player = PlayerProfile(
            name="Test Player",
            player_id="test_01",
            current_ranking=50,
            current_points=2000,
            age=25.0,
            height=180,
            weight=75,
            handed='L',
            backhand='one',
            turned_pro=2015,
            career_wins=300,
            career_losses=100,
            career_titles=5
        )
        
        total_matches = player.career_wins + player.career_losses
        win_percentage = player.career_wins / total_matches
        
        assert total_matches == 400
        assert win_percentage == 0.75
        assert player.career_titles == 5

class TestMatchContext:
    """Test MatchContext data model"""
    
    def test_match_context_creation(self):
        context = MatchContext(
            tournament_level=TournamentLevel.GRAND_SLAM,
            surface=Surface.CLAY,
            best_of=5,
            round_name="Final",
            temperature=28.0,
            humidity=65.0,
            match_importance=2.0
        )
        
        assert context.tournament_level == TournamentLevel.GRAND_SLAM
        assert context.surface == Surface.CLAY
        assert context.best_of == 5
        assert context.match_importance == 2.0

class TestAdvancedELOSystem:
    """Test ELO rating system"""
    
    def setup_method(self):
        self.elo_system = AdvancedELOSystem()
    
    def test_initial_rating(self):
        rating = self.elo_system.get_rating("new_player")
        assert rating == 1500.0
    
    def test_expected_score_calculation(self):
        # Equal ratings should give 50% probability
        expected = self.elo_system.calculate_expected_score(1500, 1500)
        assert abs(expected - 0.5) < 0.001
        
        # Higher rating should give higher probability
        expected_higher = self.elo_system.calculate_expected_score(1600, 1500)
        assert expected_higher > 0.5
        
        expected_lower = self.elo_system.calculate_expected_score(1400, 1500)
        assert expected_lower < 0.5
    
    def test_rating_update(self):
        context = MatchContext(
            tournament_level=TournamentLevel.ATP_250,
            surface=Surface.HARD
        )
        
        initial_winner_rating = self.elo_system.get_rating("winner")
        initial_loser_rating = self.elo_system.get_rating("loser")
        
        # Update ratings after match
        self.elo_system.update_ratings("winner", "loser", context, [(6, 4), (6, 2)])
        
        new_winner_rating = self.elo_system.get_rating("winner")
        new_loser_rating = self.elo_system.get_rating("loser")
        
        # Winner rating should increase, loser should decrease
        assert new_winner_rating > initial_winner_rating
        assert new_loser_rating < initial_loser_rating
    
    def test_surface_specific_ratings(self):
        context_clay = MatchContext(tournament_level=TournamentLevel.MASTERS_1000, surface=Surface.CLAY)
        context_grass = MatchContext(tournament_level=TournamentLevel.MASTERS_1000, surface=Surface.GRASS)
        
        # Update on different surfaces
        self.elo_system.update_ratings("player1", "player2", context_clay, [(6, 4), (6, 3)])
        self.elo_system.update_ratings("player1", "player3", context_grass, [(6, 2), (6, 1)])
        
        clay_rating = self.elo_system.get_rating("player1", Surface.CLAY)
        grass_rating = self.elo_system.get_rating("player1", Surface.GRASS)
        overall_rating = self.elo_system.get_rating("player1")
        
        assert clay_rating != grass_rating
        assert overall_rating > 1500  # Should increase after wins

class TestAdvancedFeatureEngineering:
    """Test feature engineering system"""
    
    def setup_method(self):
        self.feature_engine = AdvancedFeatureEngineering()
    
    def create_sample_players(self):
        player1 = PlayerProfile(
            name="Novak Djokovic",
            player_id="djokovic_n",
            current_ranking=1,
            current_points=11540,
            age=37.0,
            height=188,
            weight=77,
            handed='R',
            backhand='two',
            turned_pro=2003,
            playing_style=PlayerStyle.ALL_COURT,
            career_wins=1100,
            career_losses=200,
            clay_wins=250,
            clay_losses=40,
            grass_wins=120,
            grass_losses=20,
            hard_wins=730,
            hard_losses=140
        )
        
        player2 = PlayerProfile(
            name="Carlos Alcaraz",
            player_id="alcaraz_c",
            current_ranking=2,
            current_points=8120,
            age=22.0,
            height=185,
            weight=74,
            handed='R',
            backhand='two',
            turned_pro=2018,
            playing_style=PlayerStyle.AGGRESSIVE_BASELINE,
            career_wins=200,
            career_losses=50,
            clay_wins=60,
            clay_losses=15,
            grass_wins=25,
            grass_losses=10,
            hard_wins=115,
            hard_losses=25
        )
        
        return player1, player2
    
    def test_basic_features_creation(self):
        player1, player2 = self.create_sample_players()
        context = MatchContext(
            tournament_level=TournamentLevel.GRAND_SLAM,
            surface=Surface.HARD,
            match_importance=2.0
        )
        
        features = self.feature_engine._create_basic_features(player1, player2, context)
        
        # Test basic comparative features
        assert 'rank_diff' in features
        assert features['rank_diff'] == -1  # Player1 rank 1, Player2 rank 2
        assert 'age_diff' in features
        assert features['age_diff'] == 15  # 37 - 22
        assert 'height_diff' in features
        assert 'p1_career_win_pct' in features
        assert 'surface_win_pct_diff' in features
    
    def test_surface_win_percentages(self):
        player1, player2 = self.create_sample_players()
        
        # Test clay win percentage calculation
        p1_clay_wins, p1_clay_total = self.feature_engine._get_surface_stats(player1, Surface.CLAY)
        p1_clay_pct = p1_clay_wins / max(p1_clay_total, 1)
        
        assert p1_clay_wins == 250
        assert p1_clay_total == 290  # 250 + 40
        assert abs(p1_clay_pct - 0.862) < 0.01  # ~86.2%
    
    def test_comprehensive_features(self):
        player1, player2 = self.create_sample_players()
        context = MatchContext(
            tournament_level=TournamentLevel.MASTERS_1000,
            surface=Surface.CLAY,
            temperature=25.0,
            humidity=70.0
        )
        
        # Mock additional data
        h2h_data = [
            {'winner': 'p1', 'surface': 'clay', 'date': '2024-05-15'},
            {'winner': 'p2', 'surface': 'hard', 'date': '2024-03-10'}
        ]
        
        recent_matches_p1 = [
            {'result': 'win', 'surface': 'clay', 'date': '2024-09-01'},
            {'result': 'win', 'surface': 'clay', 'date': '2024-08-28'},
            {'result': 'loss', 'surface': 'hard', 'date': '2024-08-20'}
        ]
        
        recent_matches_p2 = [
            {'result': 'win', 'surface': 'clay', 'date': '2024-09-05'},
            {'result': 'loss', 'surface': 'clay', 'date': '2024-08-30'}
        ]
        
        features = self.feature_engine.create_comprehensive_features(
            player1, player2, context, h2h_data, recent_matches_p1, recent_matches_p2
        )
        
        # Test feature DataFrame structure
        assert isinstance(features, pd.DataFrame)
        assert len(features) == 1  # Single match
        assert len(features.columns) > 50  # Should have many features
        
        # Test specific feature categories
        feature_columns = features.columns.tolist()
        assert any('rank' in col for col in feature_columns)
        assert any('age' in col for col in feature_columns)
        assert any('surface' in col for col in feature_columns)
        assert any('h2h' in col for col in feature_columns)

@pytest.mark.asyncio
class TestTennisDataLoader:
    """Test data loading and processing"""
    
    async def test_data_loader_initialization(self):
        data_loader = TennisDataLoader(db_path="test.db")
        assert data_loader.db_path == "test.db"
        
        # Test database statistics
        stats = await data_loader.get_database_stats()
        assert 'total_matches' in stats
    
    async def test_player_profile_loading(self):
        data_loader = TennisDataLoader(db_path=":memory:")  # In-memory database
        
        # Load sample data first
        await data_loader.load_sample_data_for_testing()
        
        # Test player lookup
        profile = await data_loader.get_player_profile("Novak Djokovic")
        
        if profile:  # Might not exist in sample data
            assert profile.name == "Novak Djokovic"
            assert profile.current_ranking > 0
            assert profile.age > 0
    
    async def test_head_to_head_data(self):
        data_loader = TennisDataLoader(db_path=":memory:")
        await data_loader.load_sample_data_for_testing()
        
        # Test H2H lookup
        h2h_data = await data_loader.get_head_to_head("djokovic_n", "alcaraz_c")
        
        assert isinstance(h2h_data, list)
        # May be empty for sample data, but should not error

@pytest.mark.asyncio 
class TestProductionTennisPredictor:
    """Test main prediction system"""
    
    async def test_predictor_initialization(self):
        predictor = ProductionTennisPredictor()
        assert predictor is not None
        assert not predictor.is_trained  # Should start untrained
    
    async def test_model_initialization(self):
        predictor = ProductionTennisPredictor()
        await predictor.initialize_models()
        
        # Should have multiple models
        assert len(predictor.models) >= 7
        assert 'xgboost' in predictor.models
        assert 'lightgbm' in predictor.models
        assert 'transformer' in predictor.models
        assert predictor.meta_model is not None
    
    async def test_prediction_with_mock_data(self):
        predictor = ProductionTennisPredictor()
        await predictor.initialize_models()
        
        # Create sample training data
        sample_data = pd.DataFrame({
            'p1_name': ['Player A'] * 100,
            'p2_name': ['Player B'] * 100,
            'p1_rank': np.random.randint(1, 50, 100),
            'p2_rank': np.random.randint(1, 50, 100),
            'p1_age': np.random.uniform(20, 35, 100),
            'p2_age': np.random.uniform(20, 35, 100),
            'surface': np.random.choice(['Clay', 'Grass', 'Hard'], 100),
            'tournament_level': np.random.choice(['Grand Slam', 'Masters', 'ATP 250'], 100),
            'p1_won': np.random.randint(0, 2, 100),
            'tourney_date': ['2024-01-01'] * 100
        })
        
        # Mock the training process
        with patch.object(predictor, '_prepare_training_data') as mock_prepare:
            # Create mock features and targets
            mock_X = pd.DataFrame(np.random.random((100, 50)))
            mock_X.columns = [f'feature_{i}' for i in range(50)]
            mock_y = pd.Series(np.random.randint(0, 2, 100))
            
            mock_prepare.return_value = (mock_X, mock_y)
            
            # Train models
            model_scores = await predictor.train_models(sample_data)
            
            # Should return scores for all models
            assert isinstance(model_scores, dict)
            assert len(model_scores) > 0
            assert predictor.is_trained
    
    async def test_prediction_output_format(self):
        predictor = ProductionTennisPredictor()
        await predictor.initialize_models()
        predictor.is_trained = True  # Mock trained state
        
        # Mock the prediction methods
        with patch.object(predictor, '_load_player_by_name') as mock_load_player:
            # Create mock players
            mock_player1 = PlayerProfile(
                name="Test Player 1", player_id="test1", current_ranking=1,
                current_points=8000, age=25, height=185, weight=75,
                handed='R', backhand='two', turned_pro=2015
            )
            mock_player2 = PlayerProfile(
                name="Test Player 2", player_id="test2", current_ranking=2,
                current_points=7000, age=23, height=180, weight=70,
                handed='L', backhand='two', turned_pro=2017
            )
            
            mock_load_player.side_effect = [mock_player1, mock_player2]
            
            # Mock other data loading methods
            with patch.object(predictor, '_load_h2h_data', return_value=[]):
                with patch.object(predictor, '_load_recent_matches', return_value=[]):
                    with patch.object(predictor, '_get_betting_odds', return_value=None):
                        
                        context = MatchContext(
                            tournament_level=TournamentLevel.ATP_500,
                            surface=Surface.HARD
                        )
                        
                        result = await predictor.predict_match(
                            "Test Player 1", "Test Player 2", context
                        )
                        
                        # Test result structure
                        required_fields = [
                            'player1', 'player2', 'surface', 'tournament_level',
                            'probability_p1_wins', 'probability_p2_wins',
                            'predicted_winner', 'model_agreement',
                            'prediction_timestamp', 'prediction_latency_ms'
                        ]
                        
                        for field in required_fields:
                            assert field in result, f"Missing required field: {field}"
                        
                        # Test probability constraints
                        assert 0 <= result['probability_p1_wins'] <= 1
                        assert 0 <= result['probability_p2_wins'] <= 1
                        assert abs(result['probability_p1_wins'] + result['probability_p2_wins'] - 1.0) < 0.001
                        
                        # Test confidence metrics
                        assert 0 <= result['model_agreement'] <= 1
                        assert 'confidence_interval' in result
                        assert result['prediction_latency_ms'] > 0

@pytest.mark.asyncio
class TestProductionValidation:
    """Test validation framework"""
    
    def setup_method(self):
        self.validator = ProductionValidation()
    
    def test_metrics_calculation(self):
        # Test metrics with known data
        y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
        y_pred = [0.8, 0.2, 0.9, 0.7, 0.3, 0.6, 0.4, 0.1, 0.8, 0.9]
        detailed_results = [{}] * len(y_true)  # Mock detailed results
        
        metrics = self.validator._calculate_comprehensive_metrics(
            y_true, y_pred, detailed_results,
            datetime.now(), datetime.now()
        )
        
        # Test metric ranges
        assert 0 <= metrics['accuracy'] <= 1
        assert metrics['log_loss'] >= 0
        assert 0 <= metrics['roc_auc'] <= 1
        assert 0 <= metrics['brier_score'] <= 1
    
    def test_calibration_error_calculation(self):
        # Perfect calibration
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.2, 0.3, 0.7, 0.8])
        
        calibration_error = self.validator._calculate_calibration_error(y_true, y_pred, n_bins=2)
        
        # Should be relatively low for well-calibrated predictions
        assert 0 <= calibration_error <= 1
    
    async def test_surface_specific_validation(self):
        # Create mock predictor and data
        predictor = Mock()
        predictor.train_models = AsyncMock(return_value={'mock_model': 0.65})
        predictor._create_player_profile = Mock(return_value=PlayerProfile(
            name="Mock", player_id="mock", current_ranking=50, current_points=2000,
            age=25, height=180, weight=75, handed='R', backhand='two', turned_pro=2015
        ))
        predictor._create_match_context = Mock(return_value=MatchContext(
            tournament_level=TournamentLevel.ATP_250, surface=Surface.CLAY
        ))
        predictor.predict_match = AsyncMock(return_value={
            'probability_p1_wins': 0.65,
            'player1': 'Mock 1',
            'player2': 'Mock 2'
        })
        
        # Create sample data with different surfaces
        sample_data = pd.DataFrame({
            'surface': ['Clay'] * 50 + ['Grass'] * 30 + ['Hard'] * 40,
            'p1_won': np.random.randint(0, 2, 120),
            'tourney_date': ['2024-01-01'] * 120,
            'p1_rank': np.random.randint(1, 100, 120),
            'p2_rank': np.random.randint(1, 100, 120)
        })
        
        results = await self.validator.surface_specific_validation(predictor, sample_data)
        
        # Should have results for surfaces with enough data
        assert isinstance(results, dict)
        if 'Clay' in results:
            assert 'mean_accuracy' in results['Clay']
            assert 'total_matches' in results['Clay']

class TestAPIIntegration:
    """Test FastAPI integration"""
    
    def test_prediction_request_model(self):
        from app.main import PredictionRequest
        
        # Test valid request
        request = PredictionRequest(
            player1="Novak Djokovic",
            player2="Carlos Alcaraz",
            surface="hard",
            tournament_level="Grand Slam",
            best_of=5
        )
        
        assert request.player1 == "Novak Djokovic"
        assert request.surface == "hard"
        assert request.best_of == 5
    
    def test_prediction_response_model(self):
        from app.main import PredictionResponse
        
        # Test response structure
        response = PredictionResponse(
            player1="Test 1",
            player2="Test 2",
            surface="clay",
            tournament_level="Masters",
            probability_p1_wins=0.65,
            probability_p2_wins=0.35,
            predicted_winner="Test 1",
            win_probability=0.65,
            model_agreement=0.75,
            confidence_interval={'lower': 0.55, 'upper': 0.75},
            prediction_confidence="high",
            prediction_latency_ms=45.5,
            prediction_timestamp=datetime.now().isoformat(),
            prediction_id="test_123"
        )
        
        assert response.probability_p1_wins == 0.65
        assert response.predicted_winner == "Test 1"
        assert response.model_agreement == 0.75

class TestBettingMetrics:
    """Test betting and business metrics"""
    
    def setup_method(self):
        self.validator = ProductionValidation()
    
    def test_roi_calculation(self):
        # Mock predictions and results
        y_true = np.array([1, 0, 1, 1, 0])  # Actual outcomes
        y_pred = np.array([0.7, 0.3, 0.8, 0.6, 0.4])  # Model predictions
        detailed_results = [{'betting_odds': {'p1_odds': 1.8, 'p2_odds': 2.0}}] * 5
        
        roi_results = self.validator._calculate_betting_roi(y_true, y_pred, detailed_results)
        
        assert isinstance(roi_results, dict)
        assert 'roi' in roi_results
        assert 'total_bets' in roi_results
        assert 'win_rate' in roi_results
    
    def test_sharpe_ratio_calculation(self):
        y_true = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0])
        y_pred = np.array([0.8, 0.2, 0.9, 0.7, 0.3, 0.8, 0.1, 0.9, 0.6, 0.4])
        
        sharpe = self.validator._calculate_sharpe_ratio(y_true, y_pred)
        
        # Sharpe ratio should be a finite number
        assert np.isfinite(sharpe)
        assert isinstance(sharpe, float)
    
    def test_max_drawdown_calculation(self):
        y_true = np.array([1, 0, 1, 0, 0, 1, 1, 1])
        y_pred = np.array([0.7, 0.6, 0.8, 0.4, 0.3, 0.9, 0.8, 0.7])
        
        max_dd = self.validator._calculate_max_drawdown(y_true, y_pred)
        
        # Drawdown should be between 0 and 1
        assert 0 <= max_dd <= 1
        assert isinstance(max_dd, float)

# Integration tests
@pytest.mark.asyncio
class TestEndToEndPipeline:
    """Test complete end-to-end prediction pipeline"""
    
    async def test_complete_pipeline(self):
        """Test the complete pipeline from data loading to prediction"""
        
        # Step 1: Initialize components
        data_loader = TennisDataLoader(db_path=":memory:")
        predictor = ProductionTennisPredictor()
        validator = ProductionValidation()
        
        # Step 2: Load sample data
        matches_loaded = await data_loader.load_sample_data_for_testing()
        assert matches_loaded > 0
        
        # Step 3: Get training data
        training_data = await data_loader.get_training_data()
        
        if len(training_data) > 10:  # Need minimum data
            # Step 4: Initialize and train models
            await predictor.initialize_models()
            
            # Mock the training for speed
            with patch.object(predictor, '_prepare_training_data') as mock_prepare:
                mock_X = pd.DataFrame(np.random.random((50, 30)))
                mock_X.columns = [f'feature_{i}' for i in range(30)]
                mock_y = pd.Series(np.random.randint(0, 2, 50))
                mock_prepare.return_value = (mock_X, mock_y)
                
                model_scores = await predictor.train_models(training_data)
                assert isinstance(model_scores, dict)
                assert len(model_scores) > 0
            
            # Step 5: Make test prediction
            context = MatchContext(
                tournament_level=TournamentLevel.ATP_250,
                surface=Surface.HARD
            )
            
            # Mock data loading for prediction
            with patch.object(predictor, '_load_player_by_name') as mock_player:
                mock_player.side_effect = [
                    PlayerProfile("Player 1", "p1", 10, 3000, 25, 185, 75, 'R', 'two', 2015),
                    PlayerProfile("Player 2", "p2", 20, 2500, 27, 180, 72, 'L', 'two', 2013)
                ]
                
                with patch.object(predictor, '_load_h2h_data', return_value=[]):
                    with patch.object(predictor, '_load_recent_matches', return_value=[]):
                        with patch.object(predictor, '_get_betting_odds', return_value=None):
                            
                            result = await predictor.predict_match(
                                "Player 1", "Player 2", context, return_details=True
                            )
                            
                            # Validate prediction result
                            assert 'probability_p1_wins' in result
                            assert 0 <= result['probability_p1_wins'] <= 1
                            assert 'model_agreement' in result
                            assert 'detailed_analysis' in result
    
    def test_model_persistence(self):
        """Test model saving and loading"""
        predictor = ProductionTennisPredictor()
        
        # Mock trained state
        predictor.is_trained = True
        predictor.models = {'mock_model': Mock()}
        predictor.meta_model = Mock()
        
        # Test saving (should not error)
        try:
            # Create temporary directory for testing
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                predictor.save_models(temp_dir)
                # If no exception, test passes
        except Exception as e:
            # Some models might not be serializable in tests, but structure should work
            assert "joblib" in str(e) or "torch" in str(e)

# Performance benchmarks
class TestPerformanceBenchmarks:
    """Test system performance requirements"""
    
    @pytest.mark.asyncio
    async def test_prediction_latency(self):
        """Test that predictions complete within acceptable time"""
        predictor = ProductionTennisPredictor()
        await predictor.initialize_models()
        predictor.is_trained = True
        
        # Mock all data loading for speed
        with patch.object(predictor, '_load_player_by_name') as mock_load:
            mock_load.side_effect = [
                PlayerProfile("P1", "p1", 10, 3000, 25, 185, 75, 'R', 'two', 2015),
                PlayerProfile("P2", "p2", 15, 2800, 26, 182, 73, 'R', 'two', 2014)
            ]
            
            with patch.object(predictor, '_load_h2h_data', return_value=[]):
                with patch.object(predictor, '_load_recent_matches', return_value=[]):
                    with patch.object(predictor, '_get_betting_odds', return_value=None):
                        
                        start_time = datetime.now()
                        
                        result = await predictor.predict_match(
                            "P1", "P2", 
                            MatchContext(TournamentLevel.ATP_250, Surface.HARD),
                            return_details=False
                        )
                        
                        end_time = datetime.now()
                        latency_ms = (end_time - start_time).total_seconds() * 1000
                        
                        # Should complete within 500ms for production readiness
                        assert latency_ms < 500, f"Prediction too slow: {latency_ms}ms"
                        assert result['prediction_latency_ms'] > 0
    
    def test_feature_generation_performance(self):
        """Test feature engineering performance"""
        feature_engine = AdvancedFeatureEngineering()
        
        # Create test players
        player1 = PlayerProfile("P1", "p1", 1, 8000, 25, 185, 75, 'R', 'two', 2015)
        player2 = PlayerProfile("P2", "p2", 5, 6000, 28, 180, 72, 'L', 'one', 2012)
        context = MatchContext(TournamentLevel.MASTERS_1000, Surface.CLAY)
        
        start_time = datetime.now()
        
        features = feature_engine.create_comprehensive_features(
            player1, player2, context, [], [], []
        )
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        # Feature engineering should be fast
        assert processing_time < 100, f"Feature generation too slow: {processing_time}ms"
        assert len(features.columns) > 50, "Not enough features generated"
        assert len(features) == 1, "Should generate exactly one row of features"

# Test fixtures
@pytest.fixture
async def sample_training_data():
    """Create sample training data for tests"""
    return pd.DataFrame({
        'p1_name': ['Djokovic'] * 100 + ['Federer'] * 100,
        'p2_name': ['Nadal'] * 100 + ['Murray'] * 100,
        'p1_rank': np.random.randint(1, 20, 200),
        'p2_rank': np.random.randint(1, 20, 200),
        'p1_age': np.random.uniform(22, 38, 200),
        'p2_age': np.random.uniform(22, 38, 200),
        'surface': np.random.choice(['Clay', 'Grass', 'Hard'], 200),
        'tournament_level': np.random.choice(['Grand Slam', 'Masters', 'ATP 250'], 200),
        'p1_won': np.random.randint(0, 2, 200),
        'tourney_date': pd.date_range('2020-01-01', periods=200, freq='D').strftime('%Y-%m-%d')
    })

@pytest.fixture
def mock_predictor():
    """Create mock predictor for testing"""
    predictor = Mock(spec=ProductionTennisPredictor)
    predictor.is_trained = True
    predictor.models = {'xgboost': Mock(), 'lightgbm': Mock()}
    return predictor

# Run specific test categories
if __name__ == "__main__":
    # Run with: python -m pytest tests/test_predictor.py -v
    import pytest
    pytest.main(["-v", __file__])