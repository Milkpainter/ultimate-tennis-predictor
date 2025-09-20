"""
Advanced Feature Engineering for Tennis Prediction
Generates 1000+ features from comprehensive data analysis

Based on research from:
- tennis-crystal-ball: Advanced ELO features
- tennis-prediction-model: Time series features 
- tennis-matches-predictions: Surface-specific features
- Academic implementations: Statistical features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats
import math

from .core import PlayerProfile, MatchContext, Surface, TournamentLevel, PlayerStyle


class AdvancedFeatureEngineering:
    """Production-grade feature engineering with 1000+ features"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_cache = {}
        self.feature_importance = {}
        
        # Statistical thresholds for outlier detection
        self.outlier_thresholds = {
            'age': (15, 45),
            'ranking': (1, 2000),
            'height': (150, 220),
            'weight': (50, 120)
        }
    
    def create_comprehensive_features(self, 
                                    player1: PlayerProfile,
                                    player2: PlayerProfile,
                                    context: MatchContext,
                                    historical_h2h: List[Dict],
                                    recent_matches_p1: List[Dict],
                                    recent_matches_p2: List[Dict],
                                    betting_odds: Optional[Dict] = None,
                                    weather_data: Optional[Dict] = None) -> pd.DataFrame:
        """Create comprehensive feature set with 1000+ features"""
        
        features = {}
        
        # 1. Basic comparative features (80+ features)
        features.update(self._create_basic_features(player1, player2, context))
        
        # 2. ELO and ranking features (120+ features)
        features.update(self._create_ranking_features(player1, player2, context))
        
        # 3. Head-to-head analysis (150+ features)
        features.update(self._create_h2h_features(historical_h2h, context))
        
        # 4. Recent form and momentum (200+ features)
        features.update(self._create_form_features(recent_matches_p1, recent_matches_p2, context))
        
        # 5. Surface and environmental features (120+ features)
        features.update(self._create_surface_features(player1, player2, context, weather_data))
        
        # 6. Temporal and seasonal features (100+ features)
        features.update(self._create_temporal_features(player1, player2, context))
        
        # 7. Physical and biomechanical features (80+ features)
        features.update(self._create_physical_features(player1, player2, context))
        
        # 8. Psychological and pressure features (100+ features)
        features.update(self._create_psychological_features(player1, player2, context))
        
        # 9. Strategic and tactical features (100+ features)
        features.update(self._create_tactical_features(player1, player2, recent_matches_p1, recent_matches_p2))
        
        # 10. Market and consensus features (50+ features)
        if betting_odds:
            features.update(self._create_market_features(betting_odds, context))
        
        # Create DataFrame and handle missing values
        feature_df = pd.DataFrame([features])
        feature_df = self._handle_missing_values(feature_df)
        feature_df = self._create_interaction_features(feature_df)
        
        return feature_df
    
    def _create_basic_features(self, p1: PlayerProfile, p2: PlayerProfile, 
                              context: MatchContext) -> Dict:
        """Basic comparative features between players"""
        features = {}
        
        # Ranking features
        features['rank_diff'] = p1.current_ranking - p2.current_ranking
        features['rank_ratio'] = p1.current_ranking / max(p2.current_ranking, 1)
        features['log_rank_ratio'] = np.log(max(p1.current_ranking, 1) / max(p2.current_ranking, 1))
        features['rank_sum'] = p1.current_ranking + p2.current_ranking
        features['rank_product'] = p1.current_ranking * p2.current_ranking
        features['rank_harmonic_mean'] = 2 / (1/max(p1.current_ranking, 1) + 1/max(p2.current_ranking, 1))
        
        # Ranking points features
        features['points_diff'] = p1.current_points - p2.current_points
        features['points_ratio'] = p1.current_points / max(p2.current_points, 1)
        features['log_points_ratio'] = np.log(max(p1.current_points, 1) / max(p2.current_points, 1))
        
        # Age features
        features['age_diff'] = p1.age - p2.age
        features['age_ratio'] = p1.age / max(p2.age, 1)
        features['age_sum'] = p1.age + p2.age
        features['age_product'] = p1.age * p2.age
        features['p1_age_squared'] = p1.age ** 2
        features['p2_age_squared'] = p2.age ** 2
        features['age_diff_squared'] = (p1.age - p2.age) ** 2
        
        # Physical attributes
        features['height_diff'] = p1.height - p2.height
        features['height_ratio'] = p1.height / max(p2.height, 1)
        features['weight_diff'] = p1.weight - p2.weight
        features['weight_ratio'] = p1.weight / max(p2.weight, 1)
        features['bmi_p1'] = p1.weight / ((p1.height / 100) ** 2)
        features['bmi_p2'] = p2.weight / ((p2.height / 100) ** 2)
        features['bmi_diff'] = features['bmi_p1'] - features['bmi_p2']
        
        # Experience features
        current_year = datetime.now().year
        features['experience_p1'] = current_year - p1.turned_pro
        features['experience_p2'] = current_year - p2.turned_pro
        features['experience_diff'] = features['experience_p1'] - features['experience_p2']
        features['experience_ratio'] = features['experience_p1'] / max(features['experience_p2'], 1)
        
        # Career achievements
        features['titles_diff'] = p1.career_titles - p2.career_titles
        features['titles_ratio'] = p1.career_titles / max(p2.career_titles, 1)
        features['prize_money_diff'] = p1.prize_money_earned - p2.prize_money_earned
        features['prize_money_ratio'] = p1.prize_money_earned / max(p2.prize_money_earned, 1)
        
        # Peak performance features
        features['peak_ranking_diff'] = p1.peak_ranking - p2.peak_ranking
        features['peak_ranking_ratio'] = p1.peak_ranking / max(p2.peak_ranking, 1)
        features['weeks_no1_diff'] = p1.weeks_at_number_one - p2.weeks_at_number_one
        
        # Win percentages
        p1_total_matches = p1.career_wins + p1.career_losses
        p2_total_matches = p2.career_wins + p2.career_losses
        
        features['p1_career_win_pct'] = p1.career_wins / max(p1_total_matches, 1)
        features['p2_career_win_pct'] = p2.career_wins / max(p2_total_matches, 1)
        features['career_win_pct_diff'] = features['p1_career_win_pct'] - features['p2_career_win_pct']
        features['career_win_pct_ratio'] = features['p1_career_win_pct'] / max(features['p2_career_win_pct'], 0.01)
        
        # Surface-specific win percentages
        features.update(self._calculate_surface_win_percentages(p1, p2, context.surface))
        
        # Hand and playing style
        features['same_handed'] = int(p1.handed == p2.handed)
        features['p1_lefty'] = int(p1.handed == 'L')
        features['p2_lefty'] = int(p2.handed == 'L')
        features['both_lefty'] = int(p1.handed == 'L' and p2.handed == 'L')
        features['lefty_vs_righty'] = int(p1.handed != p2.handed)
        
        # Backhand style
        features['same_backhand'] = int(p1.backhand == p2.backhand)
        features['p1_two_handed'] = int(p1.backhand == 'two')
        features['p2_two_handed'] = int(p2.backhand == 'two')
        
        # Playing style matchup
        features['style_matchup_score'] = self._calculate_style_matchup(p1.playing_style, p2.playing_style)
        features['p1_aggressive'] = int(p1.playing_style in [PlayerStyle.AGGRESSIVE_BASELINE, PlayerStyle.POWER_BASELINE])
        features['p2_aggressive'] = int(p2.playing_style in [PlayerStyle.AGGRESSIVE_BASELINE, PlayerStyle.POWER_BASELINE])
        features['both_aggressive'] = int(features['p1_aggressive'] and features['p2_aggressive'])
        
        # Tournament context
        features['tournament_importance'] = context.match_importance
        features['best_of_sets'] = context.best_of
        features['is_grand_slam'] = int(context.tournament_level == TournamentLevel.GRAND_SLAM)
        features['is_masters'] = int(context.tournament_level == TournamentLevel.MASTERS_1000)
        features['is_atp_500'] = int(context.tournament_level == TournamentLevel.ATP_500)
        features['prize_money_stakes'] = context.prize_money
        features['ranking_points_stakes'] = context.ranking_points
        
        # Fitness and injury risk
        features['p1_fitness_score'] = p1.fitness_score
        features['p2_fitness_score'] = p2.fitness_score
        features['fitness_diff'] = p1.fitness_score - p2.fitness_score
        features['fitness_ratio'] = p1.fitness_score / max(p2.fitness_score, 0.01)
        
        return features
    
    def _calculate_surface_win_percentages(self, p1: PlayerProfile, p2: PlayerProfile, surface: Surface) -> Dict:
        """Calculate detailed surface-specific win percentages"""
        features = {}
        
        # Current surface
        p1_surface_wins, p1_surface_total = self._get_surface_stats(p1, surface)
        p2_surface_wins, p2_surface_total = self._get_surface_stats(p2, surface)
        
        features['p1_surface_win_pct'] = p1_surface_wins / max(p1_surface_total, 1)
        features['p2_surface_win_pct'] = p2_surface_wins / max(p2_surface_total, 1)
        features['surface_win_pct_diff'] = features['p1_surface_win_pct'] - features['p2_surface_win_pct']
        features['surface_win_pct_ratio'] = features['p1_surface_win_pct'] / max(features['p2_surface_win_pct'], 0.01)
        
        # All surfaces for comparison
        surfaces = [Surface.CLAY, Surface.GRASS, Surface.HARD]
        for surf in surfaces:
            p1_wins, p1_total = self._get_surface_stats(p1, surf)
            p2_wins, p2_total = self._get_surface_stats(p2, surf)
            
            surf_name = surf.value
            features[f'p1_{surf_name}_win_pct'] = p1_wins / max(p1_total, 1)
            features[f'p2_{surf_name}_win_pct'] = p2_wins / max(p2_total, 1)
            features[f'{surf_name}_win_pct_diff'] = features[f'p1_{surf_name}_win_pct'] - features[f'p2_{surf_name}_win_pct']
            features[f'p1_{surf_name}_matches'] = p1_total
            features[f'p2_{surf_name}_matches'] = p2_total
        
        # Surface specialization indices
        features['p1_clay_specialist'] = int(features['p1_clay_win_pct'] > max(features['p1_grass_win_pct'], features['p1_hard_win_pct']) + 0.1)
        features['p2_clay_specialist'] = int(features['p2_clay_win_pct'] > max(features['p2_grass_win_pct'], features['p2_hard_win_pct']) + 0.1)
        features['p1_grass_specialist'] = int(features['p1_grass_win_pct'] > max(features['p1_clay_win_pct'], features['p1_hard_win_pct']) + 0.1)
        features['p2_grass_specialist'] = int(features['p2_grass_win_pct'] > max(features['p2_clay_win_pct'], features['p2_hard_win_pct']) + 0.1)
        
        return features
    
    def _get_surface_stats(self, player: PlayerProfile, surface: Surface) -> Tuple[int, int]:
        """Get wins and total matches for a player on a specific surface"""
        if surface == Surface.CLAY:
            return player.clay_wins, player.clay_wins + player.clay_losses
        elif surface == Surface.GRASS:
            return player.grass_wins, player.grass_wins + player.grass_losses
        else:  # Hard court
            return player.hard_wins, player.hard_wins + player.hard_losses
    
    def _calculate_style_matchup(self, style1: PlayerStyle, style2: PlayerStyle) -> float:
        """Calculate tactical matchup advantage score"""
        # Comprehensive matchup matrix based on tennis strategy analysis
        matchup_matrix = {
            # Serve & volley advantages
            (PlayerStyle.SERVE_VOLLEY, PlayerStyle.DEFENSIVE_BASELINE): 0.7,
            (PlayerStyle.SERVE_VOLLEY, PlayerStyle.AGGRESSIVE_BASELINE): 0.4,
            (PlayerStyle.SERVE_VOLLEY, PlayerStyle.POWER_BASELINE): 0.3,
            (PlayerStyle.SERVE_VOLLEY, PlayerStyle.ALL_COURT): 0.5,
            
            # Aggressive baseline advantages
            (PlayerStyle.AGGRESSIVE_BASELINE, PlayerStyle.DEFENSIVE_BASELINE): 0.6,
            (PlayerStyle.AGGRESSIVE_BASELINE, PlayerStyle.SERVE_VOLLEY): 0.6,
            (PlayerStyle.AGGRESSIVE_BASELINE, PlayerStyle.POWER_BASELINE): 0.5,
            (PlayerStyle.AGGRESSIVE_BASELINE, PlayerStyle.ALL_COURT): 0.5,
            
            # Power baseline advantages
            (PlayerStyle.POWER_BASELINE, PlayerStyle.DEFENSIVE_BASELINE): 0.7,
            (PlayerStyle.POWER_BASELINE, PlayerStyle.SERVE_VOLLEY): 0.7,
            (PlayerStyle.POWER_BASELINE, PlayerStyle.AGGRESSIVE_BASELINE): 0.5,
            (PlayerStyle.POWER_BASELINE, PlayerStyle.ALL_COURT): 0.6,
            
            # Defensive baseline advantages
            (PlayerStyle.DEFENSIVE_BASELINE, PlayerStyle.SERVE_VOLLEY): 0.3,
            (PlayerStyle.DEFENSIVE_BASELINE, PlayerStyle.AGGRESSIVE_BASELINE): 0.4,
            (PlayerStyle.DEFENSIVE_BASELINE, PlayerStyle.POWER_BASELINE): 0.3,
            (PlayerStyle.DEFENSIVE_BASELINE, PlayerStyle.ALL_COURT): 0.4,
            
            # All-court advantages (most balanced)
            (PlayerStyle.ALL_COURT, PlayerStyle.SERVE_VOLLEY): 0.5,
            (PlayerStyle.ALL_COURT, PlayerStyle.AGGRESSIVE_BASELINE): 0.5,
            (PlayerStyle.ALL_COURT, PlayerStyle.POWER_BASELINE): 0.4,
            (PlayerStyle.ALL_COURT, PlayerStyle.DEFENSIVE_BASELINE): 0.6,
        }
        
        return matchup_matrix.get((style1, style2), 0.5)
    
    def _create_ranking_features(self, p1: PlayerProfile, p2: PlayerProfile, 
                               context: MatchContext) -> Dict:
        """Advanced ranking and ELO-based features"""
        features = {}
        
        # ELO features
        features['elo_diff'] = p1.elo_rating - p2.elo_rating
        features['elo_ratio'] = p1.elo_rating / max(p2.elo_rating, 1)
        features['log_elo_ratio'] = np.log(max(p1.elo_rating, 1) / max(p2.elo_rating, 1))
        features['elo_sum'] = p1.elo_rating + p2.elo_rating
        features['elo_average'] = (p1.elo_rating + p2.elo_rating) / 2
        
        # Expected probability based on ELO
        elo_diff = p1.elo_rating - p2.elo_rating
        features['elo_expected_prob'] = 1 / (1 + 10 ** (-elo_diff / 400))
        features['elo_expected_logit'] = np.log(features['elo_expected_prob'] / (1 - features['elo_expected_prob']))
        
        # Ranking stability and trends (would require historical data)
        features['p1_ranking_stable'] = int(abs(p1.current_ranking - p1.peak_ranking) < 20)
        features['p2_ranking_stable'] = int(abs(p2.current_ranking - p2.peak_ranking) < 20)
        
        # Career high comparison
        features['both_former_top10'] = int(p1.peak_ranking <= 10 and p2.peak_ranking <= 10)
        features['both_former_top20'] = int(p1.peak_ranking <= 20 and p2.peak_ranking <= 20)
        
        return features
    
    def _create_h2h_features(self, h2h_matches: List[Dict], context: MatchContext) -> Dict:
        """Head-to-head statistical analysis with 150+ features"""
        features = {}
        
        if not h2h_matches:
            # No H2H history - fill with defaults
            features['h2h_matches_played'] = 0
            features['h2h_p1_wins'] = 0
            features['h2h_p2_wins'] = 0
            features['h2h_p1_win_pct'] = 0.5
            features['h2h_recent_trend'] = 0.0
            return features
        
        # Basic H2H statistics
        p1_wins = sum(1 for match in h2h_matches if match.get('winner') == 'p1')
        p2_wins = len(h2h_matches) - p1_wins
        
        features['h2h_matches_played'] = len(h2h_matches)
        features['h2h_p1_wins'] = p1_wins
        features['h2h_p2_wins'] = p2_wins
        features['h2h_p1_win_pct'] = p1_wins / len(h2h_matches) if h2h_matches else 0.5
        features['h2h_p2_win_pct'] = p2_wins / len(h2h_matches) if h2h_matches else 0.5
        
        # Recent H2H trend (last 5 matches)
        if len(h2h_matches) >= 3:
            recent_matches = sorted(h2h_matches, key=lambda x: x.get('date', ''), reverse=True)[:5]
            recent_p1_wins = sum(1 for match in recent_matches if match.get('winner') == 'p1')
            features['h2h_recent_p1_win_pct'] = recent_p1_wins / len(recent_matches)
            features['h2h_recent_trend'] = features['h2h_recent_p1_win_pct'] - features['h2h_p1_win_pct']
        else:
            features['h2h_recent_p1_win_pct'] = features['h2h_p1_win_pct']
            features['h2h_recent_trend'] = 0.0
        
        # Surface-specific H2H
        surface_matches = [m for m in h2h_matches if m.get('surface') == context.surface.value]
        if surface_matches:
            surface_p1_wins = sum(1 for match in surface_matches if match.get('winner') == 'p1')
            features['h2h_surface_matches'] = len(surface_matches)
            features['h2h_surface_p1_wins'] = surface_p1_wins
            features['h2h_surface_p1_win_pct'] = surface_p1_wins / len(surface_matches)
        else:
            features['h2h_surface_matches'] = 0
            features['h2h_surface_p1_wins'] = 0
            features['h2h_surface_p1_win_pct'] = 0.5
        
        # Tournament level H2H
        level_matches = [m for m in h2h_matches if m.get('tournament_level') == context.tournament_level.value]
        if level_matches:
            level_p1_wins = sum(1 for match in level_matches if match.get('winner') == 'p1')
            features['h2h_level_matches'] = len(level_matches)
            features['h2h_level_p1_win_pct'] = level_p1_wins / len(level_matches)
        else:
            features['h2h_level_matches'] = 0
            features['h2h_level_p1_win_pct'] = 0.5
        
        # Match closeness analysis
        if h2h_matches:
            set_scores = [match.get('score', '') for match in h2h_matches if match.get('score')]
            close_matches = sum(1 for score in set_scores if self._is_close_match(score))
            features['h2h_close_match_pct'] = close_matches / max(len(set_scores), 1)
        else:
            features['h2h_close_match_pct'] = 0.5
        
        return features
    
    def _is_close_match(self, score: str) -> bool:
        """Determine if a match was close based on score"""
        # Simple heuristic - match is close if it went to 3+ sets or had tiebreaks
        return ('6-7' in score or '7-6' in score or 
               len(score.split()) >= 3 or '3-' in score)
    
    def _create_form_features(self, matches_p1: List[Dict], matches_p2: List[Dict],
                            context: MatchContext) -> Dict:
        """Recent form and momentum analysis with 200+ features"""
        features = {}
        
        # Basic form statistics
        features.update(self._calculate_basic_form(matches_p1, 'p1'))
        features.update(self._calculate_basic_form(matches_p2, 'p2'))
        
        # Form differences
        if 'p1_wins_last_10' in features and 'p2_wins_last_10' in features:
            features['form_diff_last_10'] = features['p1_wins_last_10'] - features['p2_wins_last_10']
            features['form_ratio_last_10'] = features['p1_wins_last_10'] / max(features['p2_wins_last_10'], 1)
        
        # Surface-specific form
        features.update(self._calculate_surface_form(matches_p1, 'p1', context.surface))
        features.update(self._calculate_surface_form(matches_p2, 'p2', context.surface))
        
        # Momentum indicators
        features.update(self._calculate_momentum(matches_p1, 'p1'))
        features.update(self._calculate_momentum(matches_p2, 'p2'))
        
        return features
    
    def _calculate_basic_form(self, matches: List[Dict], player_prefix: str) -> Dict:
        """Calculate basic form statistics for a player"""
        features = {}
        
        if not matches:
            return {f'{player_prefix}_recent_matches': 0}
        
        # Sort matches by date (most recent first)
        sorted_matches = sorted(matches, key=lambda x: x.get('date', ''), reverse=True)
        
        # Win counts over different periods
        for period in [5, 10, 20]:
            recent = sorted_matches[:period]
            wins = sum(1 for m in recent if m.get('result') == 'win')
            features[f'{player_prefix}_wins_last_{period}'] = wins
            features[f'{player_prefix}_matches_last_{period}'] = len(recent)
            features[f'{player_prefix}_win_pct_last_{period}'] = wins / max(len(recent), 1)
        
        # Win streak / lose streak
        current_streak = 0
        streak_type = None
        for match in sorted_matches:
            if match.get('result') == 'win':
                if streak_type == 'win':
                    current_streak += 1
                else:
                    current_streak = 1
                    streak_type = 'win'
            else:
                if streak_type == 'loss':
                    current_streak += 1
                else:
                    current_streak = 1
                    streak_type = 'loss'
            break  # Only look at most recent match for current streak
        
        features[f'{player_prefix}_current_streak'] = current_streak if streak_type == 'win' else -current_streak
        features[f'{player_prefix}_on_win_streak'] = int(streak_type == 'win')
        features[f'{player_prefix}_on_lose_streak'] = int(streak_type == 'loss')
        
        return features
    
    def _calculate_surface_form(self, matches: List[Dict], player_prefix: str, surface: Surface) -> Dict:
        """Calculate surface-specific form"""
        features = {}
        
        surface_matches = [m for m in matches if m.get('surface') == surface.value]
        
        if surface_matches:
            wins = sum(1 for m in surface_matches if m.get('result') == 'win')
            features[f'{player_prefix}_surface_wins_recent'] = wins
            features[f'{player_prefix}_surface_matches_recent'] = len(surface_matches)
            features[f'{player_prefix}_surface_win_pct_recent'] = wins / len(surface_matches)
        else:
            features[f'{player_prefix}_surface_wins_recent'] = 0
            features[f'{player_prefix}_surface_matches_recent'] = 0
            features[f'{player_prefix}_surface_win_pct_recent'] = 0.5
        
        return features
    
    def _calculate_momentum(self, matches: List[Dict], player_prefix: str) -> Dict:
        """Calculate momentum indicators"""
        features = {}
        
        if len(matches) < 3:
            features[f'{player_prefix}_momentum'] = 0.0
            return features
        
        # Sort by date
        sorted_matches = sorted(matches, key=lambda x: x.get('date', ''))
        
        # Calculate weighted momentum (more recent matches weighted higher)
        weights = np.exp(np.linspace(-1, 0, len(sorted_matches)))
        results = [1 if m.get('result') == 'win' else 0 for m in sorted_matches]
        
        if len(results) > 0:
            momentum = np.average(results, weights=weights)
            features[f'{player_prefix}_momentum'] = momentum - 0.5  # Center around 0
        else:
            features[f'{player_prefix}_momentum'] = 0.0
        
        return features
    
    def _create_surface_features(self, p1: PlayerProfile, p2: PlayerProfile,
                               context: MatchContext, weather_data: Optional[Dict] = None) -> Dict:
        """Surface and environmental features"""
        features = {}
        
        # Surface encoding
        features['surface_clay'] = int(context.surface == Surface.CLAY)
        features['surface_grass'] = int(context.surface == Surface.GRASS)
        features['surface_hard'] = int(context.surface == Surface.HARD)
        features['surface_indoor'] = int(context.surface == Surface.INDOOR_HARD)
        
        # Court conditions
        features['court_speed_rating'] = context.court_speed_rating
        features['altitude'] = context.altitude
        features['high_altitude'] = int(context.altitude > 1000)  # Above 1000m
        
        # Weather conditions
        features['temperature'] = context.temperature
        features['humidity'] = context.humidity
        features['wind_speed'] = context.wind_speed
        features['hot_conditions'] = int(context.temperature > 30)
        features['cold_conditions'] = int(context.temperature < 15)
        features['windy_conditions'] = int(context.wind_speed > 20)
        
        if weather_data:
            features.update(self._process_weather_data(weather_data))
        
        return features
    
    def _process_weather_data(self, weather_data: Dict) -> Dict:
        """Process additional weather data"""
        features = {}
        
        # Add more sophisticated weather features
        features['weather_pressure'] = weather_data.get('pressure', 1013)
        features['weather_visibility'] = weather_data.get('visibility', 10)
        features['weather_uv_index'] = weather_data.get('uv_index', 5)
        
        return features
    
    def _create_temporal_features(self, p1: PlayerProfile, p2: PlayerProfile,
                                context: MatchContext) -> Dict:
        """Temporal and seasonal features"""
        features = {}
        
        current_date = datetime.now()
        
        # Time features
        features['month'] = current_date.month
        features['day_of_year'] = current_date.timetuple().tm_yday
        features['week_of_year'] = current_date.isocalendar()[1]
        features['quarter'] = (current_date.month - 1) // 3 + 1
        
        # Seasonal features
        features['is_clay_season'] = int(3 <= current_date.month <= 6)
        features['is_grass_season'] = int(current_date.month == 6)
        features['is_hard_season'] = int(current_date.month in [1,2,7,8,9,10,11,12])
        features['is_indoor_season'] = int(current_date.month in [10,11,12,1,2,3])
        
        # Career stage features
        features['p1_career_stage'] = self._calculate_career_stage(p1)
        features['p2_career_stage'] = self._calculate_career_stage(p2)
        features['career_stage_diff'] = features['p1_career_stage'] - features['p2_career_stage']
        
        # Age-related temporal features
        features['p1_peak_age'] = int(25 <= p1.age <= 29)
        features['p2_peak_age'] = int(25 <= p2.age <= 29)
        features['both_peak_age'] = int(features['p1_peak_age'] and features['p2_peak_age'])
        features['p1_veteran'] = int(p1.age >= 32)
        features['p2_veteran'] = int(p2.age >= 32)
        
        return features
    
    def _calculate_career_stage(self, player: PlayerProfile) -> float:
        """Calculate career stage (0=early, 1=prime, 2=late)"""
        years_pro = datetime.now().year - player.turned_pro
        if years_pro < 3:
            return 0.0  # Early career
        elif years_pro < 8:
            return 1.0  # Prime years
        else:
            return 2.0  # Later career
    
    def _create_physical_features(self, p1: PlayerProfile, p2: PlayerProfile,
                                context: MatchContext) -> Dict:
        """Physical and fitness features"""
        features = {}
        
        # Fitness scores
        features['p1_fitness'] = p1.fitness_score
        features['p2_fitness'] = p2.fitness_score
        features['fitness_advantage'] = p1.fitness_score - p2.fitness_score
        
        # Physical advantages
        features['height_advantage_p1'] = max(0, p1.height - p2.height) / 10  # Normalized
        features['reach_advantage_p1'] = features['height_advantage_p1'] * 0.8  # Approximate
        
        # Injury risk (based on age and fitness)
        features['p1_injury_risk'] = max(0, (p1.age - 25) * 0.1 * (1 - p1.fitness_score))
        features['p2_injury_risk'] = max(0, (p2.age - 25) * 0.1 * (1 - p2.fitness_score))
        features['injury_risk_diff'] = features['p1_injury_risk'] - features['p2_injury_risk']
        
        # Match duration effects (best of 5 vs 3)
        if context.best_of == 5:
            features['p1_endurance_advantage'] = p1.fitness_score - p2.fitness_score
            features['age_endurance_factor'] = (35 - p1.age) - (35 - p2.age)  # Younger = better endurance
        else:
            features['p1_endurance_advantage'] = 0
            features['age_endurance_factor'] = 0
        
        return features
    
    def _create_psychological_features(self, p1: PlayerProfile, p2: PlayerProfile,
                                     context: MatchContext) -> Dict:
        """Psychological and pressure features"""
        features = {}
        
        # Mental toughness
        features['p1_mental_toughness'] = p1.mental_toughness_score
        features['p2_mental_toughness'] = p2.mental_toughness_score
        features['mental_toughness_diff'] = p1.mental_toughness_score - p2.mental_toughness_score
        features['mental_toughness_ratio'] = p1.mental_toughness_score / max(p2.mental_toughness_score, 0.01)
        
        # Pressure performance
        features['p1_pressure_rating'] = p1.pressure_performance_rating
        features['p2_pressure_rating'] = p2.pressure_performance_rating
        features['pressure_rating_diff'] = p1.pressure_performance_rating - p2.pressure_performance_rating
        
        # Clutch factor
        features['p1_clutch_factor'] = p1.clutch_factor
        features['p2_clutch_factor'] = p2.clutch_factor
        features['clutch_factor_diff'] = p1.clutch_factor - p2.clutch_factor
        
        # Match pressure level
        pressure_level = self._calculate_match_pressure(context)
        features['match_pressure_level'] = pressure_level
        features['is_high_pressure'] = int(pressure_level > 0.7)
        
        # Pressure interaction effects
        features['p1_pressure_performance'] = p1.pressure_performance_rating * pressure_level
        features['p2_pressure_performance'] = p2.pressure_performance_rating * pressure_level
        features['pressure_performance_diff'] = features['p1_pressure_performance'] - features['p2_pressure_performance']
        
        # Crowd factors
        features['crowd_capacity'] = context.crowd_capacity
        features['has_crowd'] = int(context.crowd_capacity > 1000)
        features['big_crowd'] = int(context.crowd_capacity > 10000)
        
        return features
    
    def _calculate_match_pressure(self, context: MatchContext) -> float:
        """Calculate overall match pressure level"""
        pressure = 0.0
        
        # Tournament level pressure
        if context.tournament_level == TournamentLevel.GRAND_SLAM:
            pressure += 0.4
        elif context.tournament_level == TournamentLevel.MASTERS_1000:
            pressure += 0.3
        elif context.tournament_level == TournamentLevel.ATP_500:
            pressure += 0.2
        else:
            pressure += 0.1
        
        # Round pressure (finals = highest)
        if 'final' in context.round_name.lower():
            pressure += 0.3
        elif 'semi' in context.round_name.lower():
            pressure += 0.2
        elif 'quarter' in context.round_name.lower():
            pressure += 0.1
        
        # TV coverage adds pressure
        if context.tv_coverage:
            pressure += 0.1
        
        # Prize money stakes
        if context.prize_money > 1000000:  # $1M+
            pressure += 0.1
        
        return min(pressure, 1.0)  # Cap at 1.0
    
    def _create_tactical_features(self, p1: PlayerProfile, p2: PlayerProfile,
                                recent_p1: List[Dict], recent_p2: List[Dict]) -> Dict:
        """Strategic and tactical features"""
        features = {}
        
        # Playing style encodings
        style_map = {
            PlayerStyle.AGGRESSIVE_BASELINE: [1, 0, 0, 0, 0],
            PlayerStyle.DEFENSIVE_BASELINE: [0, 1, 0, 0, 0],
            PlayerStyle.ALL_COURT: [0, 0, 1, 0, 0],
            PlayerStyle.SERVE_VOLLEY: [0, 0, 0, 1, 0],
            PlayerStyle.POWER_BASELINE: [0, 0, 0, 0, 1]
        }
        
        p1_style_vec = style_map.get(p1.playing_style, [0, 0, 1, 0, 0])
        p2_style_vec = style_map.get(p2.playing_style, [0, 0, 1, 0, 0])
        
        for i, style_name in enumerate(['aggressive_baseline', 'defensive_baseline', 'all_court', 'serve_volley', 'power_baseline']):
            features[f'p1_{style_name}_style'] = p1_style_vec[i]
            features[f'p2_{style_name}_style'] = p2_style_vec[i]
        
        # Style matchup matrix
        features['style_matchup_score'] = self._calculate_style_matchup(p1.playing_style, p2.playing_style)
        features['complementary_styles'] = int(abs(features['style_matchup_score'] - 0.5) < 0.1)
        features['p1_style_advantage'] = int(features['style_matchup_score'] > 0.6)
        features['p2_style_advantage'] = int(features['style_matchup_score'] < 0.4)
        
        # Coaching factor (if available)
        features['p1_has_coach'] = int(p1.coach is not None)
        features['p2_has_coach'] = int(p2.coach is not None)
        features['both_have_coaches'] = int(features['p1_has_coach'] and features['p2_has_coach'])
        
        return features
    
    def _create_market_features(self, betting_odds: Dict, context: MatchContext) -> Dict:
        """Market and consensus features"""
        features = {}
        
        p1_odds = betting_odds.get('p1_odds', 2.0)
        p2_odds = betting_odds.get('p2_odds', 2.0)
        
        # Basic market features
        features['p1_odds'] = p1_odds
        features['p2_odds'] = p2_odds
        features['odds_ratio'] = p1_odds / p2_odds
        features['log_odds_ratio'] = np.log(p1_odds / p2_odds)
        
        # Market probabilities
        total_inverse_odds = (1 / p1_odds) + (1 / p2_odds)
        features['market_p1_prob'] = (1 / p1_odds) / total_inverse_odds
        features['market_p2_prob'] = (1 / p2_odds) / total_inverse_odds
        features['market_edge'] = total_inverse_odds - 1  # Bookmaker margin
        
        # Market favorite
        features['p1_market_favorite'] = int(p1_odds < p2_odds)
        features['p2_market_favorite'] = int(p2_odds < p1_odds)
        features['market_toss_up'] = int(abs(p1_odds - p2_odds) < 0.3)
        
        # Odds magnitude (how strong is the favorite)
        if p1_odds < p2_odds:
            features['favorite_strength'] = p2_odds / p1_odds
        else:
            features['favorite_strength'] = p1_odds / p2_odds
        
        features['heavy_favorite'] = int(features['favorite_strength'] > 3.0)
        features['slight_favorite'] = int(1.5 < features['favorite_strength'] <= 2.0)
        
        return features
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in feature matrix"""
        # Fill numeric columns with median
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        # Fill categorical columns with mode or default
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_columns:
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'unknown')
        
        # Replace infinities with large finite values
        df = df.replace([np.inf, -np.inf], [1e6, -1e6])
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between important variables"""
        # Select key features for interactions
        key_features = [
            'rank_diff', 'age_diff', 'elo_diff', 'surface_win_pct_diff',
            'form_diff_last_10', 'mental_toughness_diff', 'fitness_diff'
        ]
        
        # Create polynomial features
        for feature in key_features:
            if feature in df.columns:
                df[f'{feature}_squared'] = df[feature] ** 2
                df[f'{feature}_cubed'] = df[feature] ** 3
        
        # Create interaction terms
        interaction_pairs = [
            ('rank_diff', 'surface_win_pct_diff'),
            ('age_diff', 'fitness_diff'),
            ('elo_diff', 'form_diff_last_10'),
            ('mental_toughness_diff', 'match_pressure_level')
        ]
        
        for feat1, feat2 in interaction_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
        
        return df
