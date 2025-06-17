# Utils/optuna/optimize.py
import optuna
import logging
import sys
import hydra # Added for Hydra

from hydra.core.config_store import ConfigStore

from cultivation.utils.logging_config import setup_logging
from cultivation.systems.arc_reactor.jarc_reactor.config_schema import JARCReactorConfigSchema
from cultivation.systems.arc_reactor.jarc_reactor.optimization.objective import create_objective
from cultivation.systems.arc_reactor.jarc_reactor.data.data_preparation import prepare_data

# --- Hydra Config Store Setup ---
cs = ConfigStore.instance()
cs.store(name="jarc_app", group="schema", node=JARCReactorConfigSchema)

# Setup logging - Note: This might be overridden or enhanced by Hydra's own logging setup
# For now, we keep it, but ensure Hydra's config can also control logging levels.
setup_logging(log_file='optuna_optimization.log')
logger = logging.getLogger(__name__) 

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def run_optimization(cfg: JARCReactorConfigSchema):
    """Main optimization entry point, configured by Hydra."""
    try:
        logger.info("Starting optimization")
        # Accessing Optuna config directly, Hydra handles structure
        logger.debug(f"Using Optuna config: {cfg.optuna}")
        logger.debug(f"Using Training config: {cfg.training}")

        # The decision to use synthetic data should be part of the input cfg.
        # If this script *always* uses synthetic data, it should be launched with:
        # python optimize.py training.include_synthetic_training_data=true
        # Avoid direct mutation of cfg.training.include_synthetic_training_data here.
        if cfg.training.include_synthetic_training_data:
            logger.info("Using synthetic data for hyperparameter optimization as per configuration.")
        else:
            logger.info("Using real data for hyperparameter optimization as per configuration.")

        if cfg.optuna.delete_study:
            logger.info(f"Deleting existing study '{cfg.optuna.study_name}' as per configuration (optuna.delete_study=True).")
            try:
                optuna.delete_study(study_name=cfg.optuna.study_name, storage=cfg.optuna.storage_url)
                logger.info(f"Successfully deleted study '{cfg.optuna.study_name}'.")
            except Exception as e_del:
                logger.warning(f"Could not delete study '{cfg.optuna.study_name}': {e_del}. It might not exist.")

        # Load datasets once before defining the objective
        logger.info("Loading datasets once for all trials.")
        # prepare_data now expects the full cfg object
        train_dataset, val_dataset = prepare_data(cfg=cfg, return_datasets=True)
        
        logger.info(f"Loaded {len(train_dataset)} training examples")
        logger.info(f"Loaded {len(val_dataset)} validation examples")

        study = optuna.create_study(
            study_name=cfg.optuna.study_name,
            storage=cfg.optuna.storage_url,
            direction="minimize",
            load_if_exists=True # This will load if exists, or create if not.
        )

        # Pass preloaded datasets and cfg to create_objective
        objective = create_objective(cfg, train_dataset, val_dataset)

        logger.info(f"Running {cfg.optuna.n_trials} trials")
        study.optimize(objective, n_trials=cfg.optuna.n_trials)

        logger.info("Optimization completed")
        best_trial = study.best_trial
        logger.info(f"Best trial: {best_trial.number}")
        logger.info(f"Best value: {best_trial.value}")
        logger.info("Best params:")
        for key, value in best_trial.params.items():
            logger.info(f"  {key}: {value}")
        
        return best_trial

    except Exception as e:
        logger.error(f"Optimization failed: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        # Hydra will parse command line arguments and pass them to run_optimization
        # The @hydra.main decorator handles this.
        # Example CLI overrides:
        # python optimize.py optuna.n_trials=50 training.include_synthetic_training_data=true
        # To delete study: python optimize.py --delete_study
        
        # Setup logging based on Hydra config if desired (e.g. cfg.logging.debug_mode)
        # For now, the global setup_logging and potential Hydra overrides will manage this.
        # If --debug is passed, Hydra can set logging.debug_mode=true which can be checked.

        best_trial = run_optimization()
        
        if best_trial: # Check if optimization ran successfully
            print(f"\nOptimization completed. Best trial: {best_trial.number}")
            print(f"Best parameters: {best_trial.params}")
        else:
            print("\nOptimization did not complete successfully or was interrupted.")

    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")
        sys.exit(0)
    except Exception as e:
        # Hydra might catch some exceptions itself, but this is a fallback.
        logger.error(f"Script failed: {str(e)}")
        logger.error("Stack trace:", exc_info=True)
        sys.exit(1)  
