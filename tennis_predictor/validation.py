"""
Production Validation Framework for Tennis Prediction Models
Comprehensive testing, validation, and performance monitoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, brier_score_loss, roc_auc_score, classification_report,
    confusion_matrix
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class ProductionValidation:
    """
    Comprehensive validation system for tennis prediction models
    Includes backtesting, performance tracking, and business metrics
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validation_results = {}
        self.performance_history = []
        
    async def walk_forward_validation(self, 
                                    predictor,
                                    data: pd.DataFrame,
                                    initial_train_years: int = 3,
                                    retrain_frequency_days: int = 30,
                                    test_period_days: int = 90) -> Dict:
        """
        Walk-forward validation with expanding window
        Simulates real-world deployment with periodic retraining
        """
        self.logger.info(f"Starting walk-forward validation on {len(data)} matches")
        
        # Sort data by date
        data_sorted = data.sort_values('tourney_date').reset_index(drop=True)
        
        # Convert dates to datetime
        data_sorted['date'] = pd.to_datetime(data_sorted['tourney_date'])
        
        # Determine validation periods
        start_date = data_sorted['date'].min()
        end_date = data_sorted['date'].max()
        initial_train_end = start_date + timedelta(days=365 * initial_train_years)
        
        results = []
        current_date = initial_train_end
        
        while current_date + timedelta(days=test_period_days) <= end_date:
            self.logger.info(f"Validation period: {current_date} to {current_date + timedelta(days=test_period_days)}")
            
            # Define train and test periods
            train_mask = data_sorted['date'] < current_date
            test_start = current_date
            test_end = current_date + timedelta(days=test_period_days)
            test_mask = (data_sorted['date'] >= test_start) & (data_sorted['date'] < test_end)
            
            train_data = data_sorted[train_mask]
            test_data = data_sorted[test_mask]
            
            if len(test_data) == 0:
                self.logger.warning(f"No test data for period {current_date}")
                current_date += timedelta(days=retrain_frequency_days)
                continue
            
            try:
                # Train models on expanding window
                self.logger.info(f"Training on {len(train_data)} matches...")
                model_scores = await predictor.train_models(train_data)
                
                # Make predictions on test period
                predictions = []
                actuals = []
                detailed_results = []
                
                for _, test_row in test_data.iterrows():
                    try:
                        # Create prediction input
                        player1 = predictor._create_player_profile(test_row, 'p1')
                        player2 = predictor._create_player_profile(test_row, 'p2')
                        context = predictor._create_match_context(test_row)
                        
                        # Make prediction
                        result = await predictor.predict_match(player1, player2, context, return_details=False)
                        
                        predictions.append(result['probability_p1_wins'])
                        actuals.append(int(test_row['p1_won']))
                        detailed_results.append(result)
                        
                    except Exception as e:
                        self.logger.warning(f"Prediction failed for match: {str(e)}")
                        continue
                
                if len(predictions) == 0:
                    self.logger.warning(f"No valid predictions for period {current_date}")
                    current_date += timedelta(days=retrain_frequency_days)
                    continue
                
                # Calculate metrics
                period_metrics = self._calculate_comprehensive_metrics(
                    actuals, predictions, detailed_results, test_start, test_end
                )
                
                # Add period info
                period_metrics.update({
                    'train_start': start_date,
                    'train_end': current_date,
                    'test_start': test_start,
                    'test_end': test_end,
                    'train_samples': len(train_data),
                    'test_samples': len(test_data),
                    'valid_predictions': len(predictions)
                })
                
                results.append(period_metrics)
                
            except Exception as e:
                self.logger.error(f"Validation failed for period {current_date}: {str(e)}")
            
            # Move to next validation period
            current_date += timedelta(days=retrain_frequency_days)
        
        # Aggregate results
        validation_summary = self._aggregate_validation_results(results)
        
        self.validation_results['walk_forward'] = {
            'summary': validation_summary,
            'detailed_periods': results,
            'validation_completed': datetime.now().isoformat()
        }
        
        self.logger.info(f"Walk-forward validation completed. Average accuracy: {validation_summary['mean_accuracy']:.3f}")
        
        return self.validation_results['walk_forward']
    
    def _calculate_comprehensive_metrics(self, 
                                       y_true: List[int], 
                                       y_pred_proba: List[float],
                                       detailed_results: List[Dict],
                                       period_start: datetime,
                                       period_end: datetime) -> Dict:
        """
        Calculate comprehensive validation metrics
        """
        y_true_array = np.array(y_true)
        y_pred_array = np.array(y_pred_proba)
        y_pred_binary = (y_pred_array > 0.5).astype(int)
        
        metrics = {
            # Basic classification metrics
            'accuracy': accuracy_score(y_true_array, y_pred_binary),
            'precision': precision_score(y_true_array, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true_array, y_pred_binary, zero_division=0),
            'f1_score': f1_score(y_true_array, y_pred_binary, zero_division=0),
            
            # Probabilistic metrics
            'log_loss': log_loss(y_true_array, y_pred_array),
            'brier_score': brier_score_loss(y_true_array, y_pred_array),
            'roc_auc': roc_auc_score(y_true_array, y_pred_array),
            
            # Tennis-specific metrics
            'baseline_accuracy': max(np.mean(y_true_array), 1 - np.mean(y_true_array)),  # Always predict majority class
            'improvement_over_baseline': accuracy_score(y_true_array, y_pred_binary) - max(np.mean(y_true_array), 1 - np.mean(y_true_array)),
            
            # Confidence and calibration
            'mean_confidence': np.mean([abs(p - 0.5) for p in y_pred_array]),
            'calibration_error': self._calculate_calibration_error(y_true_array, y_pred_array),
            
            # Business metrics for betting
            'betting_roi': self._calculate_betting_roi(y_true_array, y_pred_array, detailed_results),
            'sharpe_ratio': self._calculate_sharpe_ratio(y_true_array, y_pred_array),
            'max_drawdown': self._calculate_max_drawdown(y_true_array, y_pred_array),
            'kelly_growth_rate': self._calculate_kelly_growth(y_true_array, y_pred_array),
            
            # Period info
            'period_start': period_start.isoformat(),
            'period_end': period_end.isoformat(),
            'total_predictions': len(y_true_array),
            
            # Prediction distribution
            'prediction_mean': float(np.mean(y_pred_array)),
            'prediction_std': float(np.std(y_pred_array)),
            'prediction_min': float(np.min(y_pred_array)),
            'prediction_max': float(np.max(y_pred_array))
        }
        
        return metrics
    
    def _calculate_calibration_error(self, y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
        """
        Calculate calibration error (reliability of probability predictions)
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Determine if prediction is in bin
            in_bin = (y_pred > bin_lower) & (y_pred <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_pred[in_bin].mean()
                
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return float(ece)
    
    def _calculate_betting_roi(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             detailed_results: List[Dict]) -> Dict:
        """
        Calculate betting ROI assuming we bet when we have edge
        """
        total_bets = 0
        total_wagered = 0
        total_won = 0
        winning_bets = 0
        
        # Mock betting odds for ROI calculation
        for i, (actual, predicted, result) in enumerate(zip(y_true, y_pred, detailed_results)):
            # Simulate betting odds (in real system would use actual odds)
            implied_prob = 0.5 + np.random.normal(0, 0.1)  # Market with noise
            implied_prob = max(0.1, min(0.9, implied_prob))
            
            # Calculate edge
            edge = predicted - implied_prob
            
            # Only bet if we have 5%+ edge and high confidence
            if abs(edge) > 0.05 and abs(predicted - 0.5) > 0.1:
                total_bets += 1
                bet_amount = abs(edge) * 100  # Kelly-style sizing
                total_wagered += bet_amount
                
                # Determine if bet won
                bet_on_p1 = edge > 0
                bet_won = (bet_on_p1 and actual == 1) or (not bet_on_p1 and actual == 0)
                
                if bet_won:
                    winning_bets += 1
                    # Calculate winnings (simplified)
                    odds = 1 / implied_prob if bet_on_p1 else 1 / (1 - implied_prob)
                    total_won += bet_amount * (odds - 1)
        
        if total_wagered == 0:
            return {'roi': 0, 'total_bets': 0, 'win_rate': 0, 'profit': 0}
        
        roi = (total_won - total_wagered) / total_wagered * 100
        win_rate = winning_bets / total_bets if total_bets > 0 else 0
        
        return {
            'roi': float(roi),
            'total_bets': total_bets,
            'win_rate': float(win_rate),
            'profit': float(total_won - total_wagered),
            'total_wagered': float(total_wagered)
        }
    
    def _calculate_sharpe_ratio(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Sharpe ratio for betting strategy
        """
        # Simulate daily returns from betting strategy
        returns = []
        
        for actual, predicted in zip(y_true, y_pred):
            # Simulate betting return
            edge = abs(predicted - 0.5)
            if edge > 0.1:  # High confidence bets only
                if (predicted > 0.5 and actual == 1) or (predicted < 0.5 and actual == 0):
                    returns.append(edge * 2)  # Win
                else:
                    returns.append(-edge)  # Loss
            else:
                returns.append(0)  # No bet
        
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Annualized Sharpe ratio (assuming ~100 betting opportunities per year)
        sharpe = (mean_return * np.sqrt(100)) / (std_return * np.sqrt(100))
        
        return float(sharpe)
    
    def _calculate_max_drawdown(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate maximum drawdown for betting strategy
        """
        cumulative_returns = []
        cumulative = 0
        
        for actual, predicted in zip(y_true, y_pred):
            edge = abs(predicted - 0.5)
            if edge > 0.1:
                if (predicted > 0.5 and actual == 1) or (predicted < 0.5 and actual == 0):
                    cumulative += edge * 2
                else:
                    cumulative -= edge
            
            cumulative_returns.append(cumulative)
        
        if len(cumulative_returns) == 0:
            return 0.0
        
        # Calculate maximum drawdown
        peak = cumulative_returns[0]
        max_dd = 0
        
        for value in cumulative_returns:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / max(peak, 1)
            max_dd = max(max_dd, drawdown)
        
        return float(max_dd)
    
    def _calculate_kelly_growth(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Kelly criterion growth rate
        """
        growth_rates = []
        
        for actual, predicted in zip(y_true, y_pred):
            # Simulate Kelly betting
            p = predicted  # Estimated win probability
            
            # Simulate odds (b-1 is profit per unit bet)
            implied_prob = 0.5 + np.random.normal(0, 0.05)
            implied_prob = max(0.1, min(0.9, implied_prob))
            b = 1 / implied_prob
            
            # Kelly formula: f = (bp - q) / b where q = 1-p
            kelly_fraction = (b * p - (1 - p)) / b
            
            # Only bet if Kelly fraction is positive and reasonable
            if 0 < kelly_fraction < 0.1:
                if actual == 1:
                    growth_rate = kelly_fraction * (b - 1)  # Win
                else:
                    growth_rate = -kelly_fraction  # Loss
                
                growth_rates.append(growth_rate)
        
        return float(np.mean(growth_rates)) if growth_rates else 0.0
    
    async def surface_specific_validation(self, predictor, data: pd.DataFrame) -> Dict:
        """
        Validate model performance on different court surfaces
        """
        self.logger.info("Starting surface-specific validation...")
        
        surfaces = ['Clay', 'Grass', 'Hard']
        surface_results = {}
        
        for surface in surfaces:
            surface_data = data[data['surface'] == surface].copy()
            
            if len(surface_data) < 100:
                self.logger.warning(f"Insufficient data for {surface} validation ({len(surface_data)} matches)")
                continue
            
            self.logger.info(f"Validating on {surface} ({len(surface_data)} matches)...")
            
            # Time series split for surface-specific data
            tscv = TimeSeriesSplit(n_splits=3)
            surface_metrics = []
            
            for train_idx, test_idx in tscv.split(surface_data):
                train_surface = surface_data.iloc[train_idx]
                test_surface = surface_data.iloc[test_idx]
                
                try:
                    # Train on surface-specific data
                    await predictor.train_models(train_surface)
                    
                    # Test predictions
                    predictions = []
                    actuals = []
                    
                    for _, test_row in test_surface.iterrows():
                        try:
                            player1 = predictor._create_player_profile(test_row, 'p1')
                            player2 = predictor._create_player_profile(test_row, 'p2')
                            context = predictor._create_match_context(test_row)
                            
                            result = await predictor.predict_match(player1, player2, context, return_details=False)
                            
                            predictions.append(result['probability_p1_wins'])
                            actuals.append(int(test_row['p1_won']))
                        except:
                            continue
                    
                    if len(predictions) > 0:
                        fold_metrics = self._calculate_comprehensive_metrics(
                            actuals, predictions, [], period_start=datetime.now(), period_end=datetime.now()
                        )
                        surface_metrics.append(fold_metrics)
                
                except Exception as e:
                    self.logger.warning(f"Surface validation fold failed: {str(e)}")
                    continue
            
            # Aggregate surface results
            if surface_metrics:
                surface_results[surface] = {
                    'mean_accuracy': np.mean([m['accuracy'] for m in surface_metrics]),
                    'std_accuracy': np.std([m['accuracy'] for m in surface_metrics]),
                    'mean_log_loss': np.mean([m['log_loss'] for m in surface_metrics]),
                    'mean_roc_auc': np.mean([m['roc_auc'] for m in surface_metrics]),
                    'total_matches': len(surface_data),
                    'cv_folds': len(surface_metrics)
                }
        
        self.validation_results['surface_specific'] = surface_results
        return surface_results
    
    async def pressure_situation_validation(self, predictor, data: pd.DataFrame) -> Dict:
        """
        Validate model performance in high-pressure situations
        """
        self.logger.info("Validating pressure situation performance...")
        
        pressure_scenarios = {
            'grand_slam_finals': data[
                (data['tourney_level'] == 'Grand Slam') & 
                (data['round'].str.contains('F', case=False, na=False))
            ],
            'masters_finals': data[
                (data['tourney_level'] == 'Masters') & 
                (data['round'].str.contains('F', case=False, na=False))
            ],
            'top10_matchups': data[
                (data['p1_rank'] <= 10) & (data['p2_rank'] <= 10)
            ],
            'close_ranking_matches': data[
                abs(data['p1_rank'] - data['p2_rank']) <= 5
            ]
        }
        
        pressure_results = {}
        
        for scenario_name, scenario_data in pressure_scenarios.items():
            if len(scenario_data) < 20:
                self.logger.warning(f"Insufficient data for {scenario_name} ({len(scenario_data)} matches)")
                continue
            
            self.logger.info(f"Testing {scenario_name} ({len(scenario_data)} matches)...")
            
            try:
                # Use most recent data for training, oldest for testing
                split_point = int(len(scenario_data) * 0.7)
                train_data = scenario_data.iloc[:split_point]
                test_data = scenario_data.iloc[split_point:]
                
                # Train on scenario-specific data
                await predictor.train_models(train_data)
                
                # Make predictions
                predictions = []
                actuals = []
                
                for _, test_row in test_data.iterrows():
                    try:
                        player1 = predictor._create_player_profile(test_row, 'p1')
                        player2 = predictor._create_player_profile(test_row, 'p2')
                        context = predictor._create_match_context(test_row)
                        
                        # Increase match importance for pressure scenarios
                        if 'final' in scenario_name:
                            context.match_importance = 2.0
                        elif 'top10' in scenario_name:
                            context.match_importance = 1.5
                        
                        result = await predictor.predict_match(player1, player2, context, return_details=False)
                        
                        predictions.append(result['probability_p1_wins'])
                        actuals.append(int(test_row['p1_won']))
                    except:
                        continue
                
                if len(predictions) > 10:  # Minimum for meaningful results
                    scenario_metrics = self._calculate_comprehensive_metrics(
                        actuals, predictions, [], datetime.now(), datetime.now()
                    )
                    
                    pressure_results[scenario_name] = {
                        'accuracy': scenario_metrics['accuracy'],
                        'log_loss': scenario_metrics['log_loss'],
                        'roc_auc': scenario_metrics['roc_auc'],
                        'calibration_error': scenario_metrics['calibration_error'],
                        'total_matches': len(scenario_data),
                        'test_matches': len(predictions),
                        'difficulty_rating': self._assess_prediction_difficulty(scenario_name)
                    }
            
            except Exception as e:
                self.logger.error(f"Pressure validation failed for {scenario_name}: {str(e)}")
        
        self.validation_results['pressure_situations'] = pressure_results
        return pressure_results
    
    def _assess_prediction_difficulty(self, scenario_name: str) -> str:
        """
        Assess how difficult a prediction scenario is
        """
        difficulty_map = {
            'grand_slam_finals': 'very_hard',
            'masters_finals': 'hard', 
            'top10_matchups': 'hard',
            'close_ranking_matches': 'medium'
        }
        return difficulty_map.get(scenario_name, 'medium')
    
    def _aggregate_validation_results(self, results: List[Dict]) -> Dict:
        """
        Aggregate results across all validation periods
        """
        if not results:
            return {}
        
        metrics_to_aggregate = ['accuracy', 'log_loss', 'brier_score', 'roc_auc', 'betting_roi']
        
        aggregated = {}
        for metric in metrics_to_aggregate:
            values = [r[metric] for r in results if metric in r and not np.isnan(r[metric])]
            if values:
                aggregated[f'mean_{metric}'] = np.mean(values)
                aggregated[f'std_{metric}'] = np.std(values)
                aggregated[f'min_{metric}'] = np.min(values)
                aggregated[f'max_{metric}'] = np.max(values)
        
        # Overall statistics
        aggregated.update({
            'total_periods': len(results),
            'total_predictions': sum(r['total_predictions'] for r in results),
            'validation_start': min(r['period_start'] for r in results),
            'validation_end': max(r['period_end'] for r in results)
        })
        
        return aggregated
    
    async def generate_validation_report(self, output_dir: str = "reports/") -> str:
        """
        Generate comprehensive validation report
        """
        if not self.validation_results:
            raise ValueError("No validation results available. Run validation first.")
        
        report_path = Path(output_dir) / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        report_content = [
            "# Tennis Prediction Model Validation Report",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n## Executive Summary"
        ]
        
        # Walk-forward results
        if 'walk_forward' in self.validation_results:
            wf_results = self.validation_results['walk_forward']['summary']
            report_content.extend([
                "\n### Walk-Forward Validation Results",
                f"- **Mean Accuracy**: {wf_results.get('mean_accuracy', 0):.3f} ± {wf_results.get('std_accuracy', 0):.3f}",
                f"- **Mean Log Loss**: {wf_results.get('mean_log_loss', 0):.3f}",
                f"- **Mean ROC AUC**: {wf_results.get('mean_roc_auc', 0):.3f}",
                f"- **Total Predictions**: {wf_results.get('total_predictions', 0):,}",
                f"- **Validation Periods**: {wf_results.get('total_periods', 0)}"
            ])
        
        # Surface-specific results
        if 'surface_specific' in self.validation_results:
            report_content.append("\n### Surface-Specific Performance")
            
            for surface, metrics in self.validation_results['surface_specific'].items():
                report_content.extend([
                    f"\n#### {surface} Court",
                    f"- **Accuracy**: {metrics['mean_accuracy']:.3f} ± {metrics['std_accuracy']:.3f}",
                    f"- **Log Loss**: {metrics['mean_log_loss']:.3f}",
                    f"- **ROC AUC**: {metrics['mean_roc_auc']:.3f}",
                    f"- **Total Matches**: {metrics['total_matches']:,}"
                ])
        
        # Pressure situations
        if 'pressure_situations' in self.validation_results:
            report_content.append("\n### High-Pressure Situation Performance")
            
            for scenario, metrics in self.validation_results['pressure_situations'].items():
                report_content.extend([
                    f"\n#### {scenario.replace('_', ' ').title()}",
                    f"- **Accuracy**: {metrics['accuracy']:.3f}",
                    f"- **Difficulty**: {metrics['difficulty_rating']}",
                    f"- **Test Matches**: {metrics['test_matches']:,}"
                ])
        
        # Business metrics
        if 'walk_forward' in self.validation_results:
            wf_summary = self.validation_results['walk_forward']['summary']
            if 'mean_betting_roi' in wf_summary:
                report_content.extend([
                    "\n### Business Performance",
                    f"- **Betting ROI**: {wf_summary['mean_betting_roi']:.2f}%",
                    f"- **Sharpe Ratio**: {wf_summary.get('mean_sharpe_ratio', 0):.2f}",
                    f"- **Max Drawdown**: {wf_summary.get('mean_max_drawdown', 0):.2f}%"
                ])
        
        # Recommendations
        report_content.extend([
            "\n## Recommendations",
            "\nBased on validation results:",
            "\n1. **Model Performance**: "
        ])
        
        if 'walk_forward' in self.validation_results:
            accuracy = self.validation_results['walk_forward']['summary'].get('mean_accuracy', 0)
            if accuracy > 0.70:
                report_content.append("Excellent accuracy achieved (>70%). Ready for production deployment.")
            elif accuracy > 0.65:
                report_content.append("Good accuracy (65-70%). Consider additional feature engineering.")
            else:
                report_content.append("Below-target accuracy (<65%). Requires model improvements.")
        
        report_content.extend([
            "\n2. **Surface Specialization**: Consider separate models for Clay/Grass specialists.",
            "\n3. **Pressure Situations**: Additional training data needed for high-pressure matches.",
            "\n4. **Business Viability**: "
        ])
        
        # Write report
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_content))
        
        self.logger.info(f"Validation report saved to {report_path}")
        return str(report_path)
    
    async def benchmark_against_baselines(self, predictor, test_data: pd.DataFrame) -> Dict:
        """
        Benchmark our model against simple baseline approaches
        """
        self.logger.info("Benchmarking against baseline models...")
        
        # Get our model predictions
        our_predictions = []
        actuals = []
        
        for _, row in test_data.iterrows():
            try:
                player1 = predictor._create_player_profile(row, 'p1')
                player2 = predictor._create_player_profile(row, 'p2')
                context = predictor._create_match_context(row)
                
                result = await predictor.predict_match(player1, player2, context, return_details=False)
                our_predictions.append(result['probability_p1_wins'])
                actuals.append(int(row['p1_won']))
            except:
                continue
        
        if len(our_predictions) == 0:
            return {}
        
        # Baseline 1: Always predict higher-ranked player
        ranking_predictions = []
        for _, row in test_data.iterrows():
            p1_rank = row.get('p1_rank', 999)
            p2_rank = row.get('p2_rank', 999)
            # Lower rank number = better player
            prob = 0.7 if p1_rank < p2_rank else 0.3 if p1_rank > p2_rank else 0.5
            ranking_predictions.append(prob)
        
        # Baseline 2: Simple ELO
        simple_elo_predictions = []
        for _, row in test_data.iterrows():
            # Simplified ELO based on ranking points
            p1_points = row.get('p1_rank_points', 1000)
            p2_points = row.get('p2_rank_points', 1000)
            point_diff = p1_points - p2_points
            prob = 1 / (1 + 10**(-point_diff / 2000))  # Simple sigmoid
            simple_elo_predictions.append(prob)
        
        # Baseline 3: Random predictions
        random_predictions = [np.random.random() for _ in range(len(actuals))]
        
        # Calculate metrics for each approach
        approaches = {
            'our_model': our_predictions,
            'ranking_favorite': ranking_predictions,
            'simple_elo': simple_elo_predictions,
            'random': random_predictions
        }
        
        benchmark_results = {}
        
        for approach_name, predictions in approaches.items():
            if len(predictions) == len(actuals):
                pred_binary = (np.array(predictions) > 0.5).astype(int)
                
                benchmark_results[approach_name] = {
                    'accuracy': accuracy_score(actuals, pred_binary),
                    'log_loss': log_loss(actuals, predictions),
                    'roc_auc': roc_auc_score(actuals, predictions),
                    'brier_score': brier_score_loss(actuals, predictions)
                }
        
        # Calculate improvements
        if 'our_model' in benchmark_results and 'ranking_favorite' in benchmark_results:
            our_acc = benchmark_results['our_model']['accuracy']
            baseline_acc = benchmark_results['ranking_favorite']['accuracy']
            benchmark_results['improvement_over_ranking'] = our_acc - baseline_acc
            benchmark_results['relative_improvement'] = (our_acc - baseline_acc) / baseline_acc * 100
        
        self.validation_results['benchmarks'] = benchmark_results
        return benchmark_results
    
    def get_validation_summary(self) -> Dict:
        """
        Get summary of all validation results
        """
        if not self.validation_results:
            return {'status': 'No validation completed'}
        
        summary = {
            'validation_completed': True,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add key metrics from each validation type
        if 'walk_forward' in self.validation_results:
            wf = self.validation_results['walk_forward']['summary']
            summary['overall_accuracy'] = wf.get('mean_accuracy', 0)
            summary['overall_log_loss'] = wf.get('mean_log_loss', 0)
            summary['total_predictions'] = wf.get('total_predictions', 0)
        
        if 'surface_specific' in self.validation_results:
            surface_accs = [metrics['mean_accuracy'] for metrics in self.validation_results['surface_specific'].values()]
            if surface_accs:
                summary['best_surface_accuracy'] = max(surface_accs)
                summary['worst_surface_accuracy'] = min(surface_accs)
        
        if 'benchmarks' in self.validation_results:
            benchmarks = self.validation_results['benchmarks']
            if 'improvement_over_ranking' in benchmarks:
                summary['improvement_over_baseline'] = benchmarks['improvement_over_ranking']
        
        return summary