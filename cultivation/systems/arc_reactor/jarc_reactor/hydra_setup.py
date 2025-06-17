# cultivation/systems/arc_reactor/jarc_reactor/hydra_setup.py

import sys 
from hydra.core.config_store import ConfigStore
import hydra
from .config_schema import (
    JARCReactorConfigSchema,
    ModelConfigSchema,
    TrainingConfigSchema,
    DataLoaderConfig,
    OptunaConfigSchema,
    LoggingConfigSchema,
    FineTuningConfigSchema,
    MetricsConfigSchema,
    SchedulerConfigSchema,
    EvaluationConfigSchema
)

def register_hydra_configs():
    print("DEBUG: hydra_setup.py inside register_hydra_configs()", file=sys.__stderr__) 
    """Registers all structured configs with Hydra's ConfigStore."""
    cs = ConfigStore.instance()

    # Register the main application config
    # This is the top-level config that will be loaded by @hydra.main()
    try:
        cs.store(group="schema", name="jarc_app", node=JARCReactorConfigSchema)  # Main application schema
    except hydra.errors.ConfigAlreadyRegistered:
        pass

    # Register individual group configs. 
    # This allows them to be selected or overridden from the command line
    # or from other config files.
    # The 'group' parameter corresponds to the subdirectories in the 'conf' folder.
    try:
        cs.store(group="model", name="base_model", node=ModelConfigSchema)
    except hydra.errors.ConfigAlreadyRegistered:
        pass
    try:
        cs.store(group="training", name="base_training", node=TrainingConfigSchema)
    except hydra.errors.ConfigAlreadyRegistered:
        pass
    try:
        cs.store(group="dataloader", name="base_dataloader", node=DataLoaderConfig)
    except hydra.errors.ConfigAlreadyRegistered:
        pass
    try:
        cs.store(group="optuna", name="base_optuna", node=OptunaConfigSchema)
    except hydra.errors.ConfigAlreadyRegistered:
        pass
    try:
        cs.store(group="logging", name="base_logging", node=LoggingConfigSchema)
    except hydra.errors.ConfigAlreadyRegistered:
        pass
    try:
        cs.store(group="finetuning", name="base_finetuning", node=FineTuningConfigSchema)
    except hydra.errors.ConfigAlreadyRegistered:
        pass
    try:
        cs.store(group="metrics", name="base_metrics", node=MetricsConfigSchema)
    except hydra.errors.ConfigAlreadyRegistered:
        pass
    try:
        cs.store(group="scheduler", name="base_scheduler", node=SchedulerConfigSchema)
    except hydra.errors.ConfigAlreadyRegistered:
        pass
    try:
        cs.store(group="evaluation", name="base_evaluation", node=EvaluationConfigSchema)
    except hydra.errors.ConfigAlreadyRegistered:
        pass

    # You might call this function early in your application's entry point,
    # e.g., in the main script before @hydra.main() is invoked, or within
    # a plugin system if Hydra is used that way.
    # For now, this function is defined and can be imported and called where needed.

# Example of how this might be called (though typically done in the main app script):
# if __name__ == "__main__":
# register_hydra_configs()
# print("Hydra configs registered.")
