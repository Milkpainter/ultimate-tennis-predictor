#!/usr/bin/env python3
"""
Complete Model Training Script for Tennis Prediction System
Handles data loading, feature engineering, model training, and validation
"""

import asyncio
import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tennis_predictor.predictor import ProductionTennisPredictor
from tennis_predictor.data_loader import TennisDataLoader
from tennis_predictor.validation import ProductionValidation
from tennis_predictor.core import Surface, TournamentLevel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/training.log')
    ]
)
logger = logging.getLogger(__name__)

async def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='Train Ultimate Tennis Predictor models')
    parser.add_argument('--data-years', nargs='+', type=int, 
                       default=list(range(2015, 2025)),
                       help='Years of data to load')
    parser.add_argument('--min-matches', type=int, default=1000,
                       help='Minimum matches required for training')
    parser.add_argument('--surfaces', nargs='+', 
                       choices=['Clay', 'Grass', 'Hard'],
                       help='Surfaces to include (default: all)')
    parser.add_argument('--validation', action='store_true',
                       help='Run comprehensive validation')
    parser.add_argument('--load-atp', action='store_true',
                       help='Download fresh ATP data')
    parser.add_argument('--load-wta', action='store_true', 
                       help='Download fresh WTA data')
    parser.add_argument('--output-dir', default='models/trained',
                       help='Output directory for trained models')
    
    args = parser.parse_args()
    
    logger.info("🎾 Starting Ultimate Tennis Predictor Training Pipeline")
    logger.info(f"Configuration: {vars(args)}")
    
    try:
        # Step 1: Initialize data loader
        logger.info("📊 Initializing data loader...")
        data_loader = TennisDataLoader()
        
        # Step 2: Load tennis data
        total_matches = 0
        
        if args.load_atp:
            logger.info("⬇️  Loading ATP match data...")
            atp_matches = await data_loader.load_atp_data(years=args.data_years)
            total_matches += atp_matches
            logger.info(f"Loaded {atp_matches:,} ATP matches")
        
        if args.load_wta:
            logger.info("⬇️  Loading WTA match data...")
            wta_matches = await data_loader.load_wta_data(years=args.data_years)
            total_matches += wta_matches
            logger.info(f"Loaded {wta_matches:,} WTA matches")
        
        # Load sample data if no fresh data
        if total_matches == 0:
            logger.info("📝 Loading sample data for testing...")
            sample_matches = await data_loader.load_sample_data_for_testing()
            logger.info(f"Created {sample_matches:,} sample matches")
        
        # Step 3: Get training dataset
        logger.info("🔄 Preparing training dataset...")
        training_data = await data_loader.get_training_data(
            min_date='2015-01-01',
            surfaces=args.surfaces
        )
        
        if len(training_data) < args.min_matches:
            logger.error(f"Insufficient training data: {len(training_data)} < {args.min_matches}")
            return 1
        
        logger.info(f"Training dataset ready: {len(training_data):,} matches")
        
        # Display dataset statistics
        stats = training_data.groupby(['surface', 'tourney_level']).size().unstack(fill_value=0)
        logger.info(f"Dataset breakdown:\n{stats}")
        
        # Step 4: Initialize predictor
        logger.info("🤖 Initializing prediction models...")
        predictor = ProductionTennisPredictor()
        await predictor.initialize_models()
        
        # Step 5: Train models
        logger.info("🚀 Starting model training...")
        training_start = datetime.now()
        
        model_scores = await predictor.train_models(training_data, validation_split=0.2)
        
        training_time = (datetime.now() - training_start).total_seconds()
        logger.info(f"✅ Model training completed in {training_time:.1f} seconds")
        
        # Display model performance
        logger.info("📈 Model Performance Results:")
        for model_name, score in model_scores.items():
            logger.info(f"   {model_name}: {score:.3f} accuracy")
        
        avg_accuracy = sum(model_scores.values()) / len(model_scores)
        logger.info(f"   Average Accuracy: {avg_accuracy:.3f}")
        
        # Step 6: Save trained models
        logger.info(f"💾 Saving models to {args.output_dir}...")
        predictor.save_models(args.output_dir)
        
        # Step 7: Validation (optional)
        if args.validation:
            logger.info("🔍 Running comprehensive validation...")
            validator = ProductionValidation()
            
            # Walk-forward validation
            logger.info("Running walk-forward validation...")
            walk_forward_results = await validator.walk_forward_validation(
                predictor, training_data, initial_train_years=2
            )
            
            # Surface-specific validation
            logger.info("Running surface-specific validation...")
            surface_results = await validator.surface_specific_validation(
                predictor, training_data
            )
            
            # Pressure situation validation
            logger.info("Running pressure situation validation...")
            pressure_results = await validator.pressure_situation_validation(
                predictor, training_data
            )
            
            # Baseline comparison
            logger.info("Benchmarking against baselines...")
            benchmark_results = await validator.benchmark_against_baselines(
                predictor, training_data[-1000:]  # Use recent data for benchmarking
            )
            
            # Generate validation report
            report_path = await validator.generate_validation_report()
            logger.info(f"📄 Validation report saved to {report_path}")
            
            # Display key validation results
            logger.info("🎯 Validation Results Summary:")
            
            if 'walk_forward' in validator.validation_results:
                wf_summary = validator.validation_results['walk_forward']['summary']
                logger.info(f"   Overall Accuracy: {wf_summary.get('mean_accuracy', 0):.3f} ± {wf_summary.get('std_accuracy', 0):.3f}")
                logger.info(f"   Total Predictions: {wf_summary.get('total_predictions', 0):,}")
            
            if surface_results:
                logger.info("   Surface-Specific Results:")
                for surface, metrics in surface_results.items():
                    logger.info(f"     {surface}: {metrics['mean_accuracy']:.3f} accuracy")
            
            if benchmark_results and 'improvement_over_ranking' in benchmark_results:
                improvement = benchmark_results['improvement_over_ranking']
                logger.info(f"   Improvement over baseline: {improvement:+.3f} accuracy points")
        
        # Step 8: Test prediction functionality
        logger.info("🧪 Testing prediction functionality...")
        
        # Test with sample match
        from tennis_predictor.core import PlayerProfile, MatchContext
        
        sample_context = MatchContext(
            tournament_level=TournamentLevel.GRAND_SLAM,
            surface=Surface.HARD,
            best_of=5,
            round_name="Final",
            match_importance=2.0
        )
        
        test_result = await predictor.predict_match(
            "Novak Djokovic", "Carlos Alcaraz", sample_context
        )
        
        logger.info("✅ Sample Prediction Test:")
        logger.info(f"   {test_result['player1']} vs {test_result['player2']}")
        logger.info(f"   Winner Probability: {test_result['win_probability']:.1%}")
        logger.info(f"   Model Agreement: {test_result['model_agreement']:.1%}")
        logger.info(f"   Prediction Latency: {test_result['prediction_latency_ms']:.1f}ms")
        
        # Final summary
        logger.info("\n🏆 TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"✅ Models trained on {len(training_data):,} matches")
        logger.info(f"✅ Average model accuracy: {avg_accuracy:.3f}")
        logger.info(f"✅ Training completed in {training_time:.1f} seconds")
        logger.info(f"✅ Models saved to {args.output_dir}")
        
        if args.validation:
            logger.info(f"✅ Comprehensive validation completed")
            logger.info(f"✅ Validation report generated")
        
        logger.info(f"✅ System ready for predictions!")
        logger.info(f"\n🚀 Start API server with: python -m uvicorn app.main:app --reload")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Training pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

def create_training_environment():
    """Create necessary directories and files for training"""
    directories = [
        'data',
        'models/trained',
        'logs',
        'reports',
        'config'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Create basic config file
    config_content = """
# Ultimate Tennis Predictor Configuration

models:
  ensemble_weights:
    xgboost: 0.25
    lightgbm: 0.20
    catboost: 0.15
    random_forest: 0.15
    logistic: 0.10
    mlp: 0.10
    gaussian_process: 0.05
  
  hyperparameters:
    xgboost:
      n_estimators: 500
      max_depth: 8
      learning_rate: 0.1
      subsample: 0.8
    
    lightgbm:
      n_estimators: 500
      max_depth: 8
      learning_rate: 0.1

validation:
  time_series_splits: 5
  min_train_years: 2
  test_period_months: 3
  min_accuracy_threshold: 0.65

data:
  sources:
    - jeff_sackmann_atp
    - jeff_sackmann_wta
  min_matches_per_player: 10
  feature_selection_threshold: 0.01
  
api:
  host: "0.0.0.0"
  port: 8000
  max_batch_size: 50
  rate_limit_per_minute: 1000
"""
    
    config_path = Path('config/production.yaml')
    if not config_path.exists():
        with open(config_path, 'w') as f:
            f.write(config_content)
        logger.info(f"Created configuration file: {config_path}")

if __name__ == "__main__":
    # Create necessary directories
    create_training_environment()
    
    # Run main training pipeline
    exit_code = asyncio.run(main())
    sys.exit(exit_code)