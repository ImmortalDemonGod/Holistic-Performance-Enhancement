# cultivation/systems/arc_reactor/jarc_reactor/config_schema.py

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# It's good practice to import MISSING for optional fields without defaults
# from hydra.core.config_store import OmegaConf
# from omegaconf import MISSING # Not strictly needed if all fields have defaults or are Optional[type] = None

@dataclass
class LoraConfig:
    use_lora: bool = False
    rank: int = 128
    in_features: int = 128
    out_features: int = 128

@dataclass
class ContextEncoderConfig:
    d_model: int = 1024
    heads: int = 16
    dropout_rate: float = 0.12

@dataclass
class ModelConfigSchema:
    input_dim: int = 30
    seq_len: int = 30
    d_model: int = 256
    encoder_layers: int = 2
    decoder_layers: int = 2
    heads: int = 4
    d_ff: int = 128
    output_dim: int = 30
    dropout_rate: float = 0.15
    context_encoder: ContextEncoderConfig = field(default_factory=ContextEncoderConfig)
    encoder_dropout_rate: float = 0.61
    decoder_dropout_rate: float = 0.12
    lora: LoraConfig = field(default_factory=LoraConfig)
    checkpoint_path: Optional[str] = "/workspaces/JARC-Reactor/lightning_logs/checkpoints/model-step=step=40-val_loss=val_loss=1.7084.ckpt"

@dataclass
class TrainingConfigSchema:
    batch_size: int = 1
    learning_rate: float = 0.00002009
    max_epochs: int = 100
    device_choice: str = "auto"  # 'auto', 'cpu', 'gpu'
    precision: Any = 32 # int or str like 'bf16'
    gradient_clip_val: float = 1.0
    fast_dev_run: bool = False
    train_from_checkpoint: bool = False
    include_synthetic_training_data: bool = False
    training_data_dir: str = "jarc_reactor/data/training_data/training"

@dataclass
class OptunaPruningConfig:
    n_warmup_steps: int = 5
    n_startup_trials: int = 10
    patience: int = 20
    pruning_percentile: int = 25

@dataclass
class OptunaConfigSchema:
    n_trials: int = 1
    study_name: str = "jarc_optimization_v3"
    storage_url: str = "sqlite:///jarc_optuna.db"
    param_ranges: Dict[str, List[Any]] = field(default_factory=lambda: {
        "d_model": [32, 2048, 16],
        "heads": [2, 4, 8, 16, 32, 64],
        "encoder_layers": [1, 12],
        "decoder_layers": [1, 12],
        "d_ff": [64, 2048, 64],
        "dropout": [0.01, 0.7],
        "context_encoder_d_model": [32, 512, 32],
        "context_encoder_heads": [2, 4, 8],
        "batch_size": [8, 512],
        "learning_rate": [0.000001, 0.1],
        "max_epochs": [4, 5, 1],
        "gradient_clip_val": [0.0, 5.0]
    })
    pruning: OptunaPruningConfig = field(default_factory=OptunaPruningConfig)

@dataclass
class FileLoggingConfig:
    enable: bool = True
    log_file_name: str = "jarc_reactor_app.log"
    max_bytes: int = 10485760  # 10MB
    backup_count: int = 5

@dataclass
class ConsoleLoggingConfig:
    enable: bool = True

@dataclass
class LoggingConfigSchema:
    level: str = "INFO"
    debug_mode: bool = False
    log_dir: str = "jarc_reactor/logs"
    file_logging: FileLoggingConfig = field(default_factory=FileLoggingConfig)
    console_logging: ConsoleLoggingConfig = field(default_factory=ConsoleLoggingConfig)

@dataclass
class FineTuningConfigSchema:
    mode: str = "all"  # "all" or "random"
    num_random_tasks: int = 1
    save_dir: str = "finetuning_results"
    max_epochs: int = 100
    learning_rate: float = 0.00001
    patience: int = 5

@dataclass
class MetricsConfigSchema:
    confidence_threshold: float = 0.5

@dataclass
class SchedulerConfigSchema:
    use_cosine_annealing: bool = True
    T_0: int = 4
    T_mult: int = 2
    eta_min: float = 0.0000004 # 4e-7

@dataclass
class EvaluationConfigSchema:
    mode: str = 'all'
    output_dir: str = 'evaluation_results'
    debug_mode: bool = True
    save_predictions: bool = True
    create_submission: bool = True
    data_dir: str = "${oc.env:EVALUATION_DATA_DIR,cultivation/systems/arc_reactor/jarc_reactor/data/evaluation_data}"

@dataclass
class JARCReactorConfigSchema:
    # Defaults list for Hydra to know which sub-configs to load by default
    # This is usually in the main config.yaml, but can also be structured here if preferred for programmatic access
    # For now, we assume the defaults in config.yaml are primary.
    defaults: List[Any] = field(default_factory=lambda: [
        {"model": "default"},
        {"training": "default"},
        {"optuna": "default"},
        {"logging": "default"},
        {"finetuning": "default"},
        {"metrics": "default"},
        {"scheduler": "default"},
        {"evaluation": "default"},
        "_self_"
    ])

    model: ModelConfigSchema = field(default_factory=ModelConfigSchema)
    training: TrainingConfigSchema = field(default_factory=TrainingConfigSchema)
    optuna: OptunaConfigSchema = field(default_factory=OptunaConfigSchema)
    logging: LoggingConfigSchema = field(default_factory=LoggingConfigSchema)
    finetuning: FineTuningConfigSchema = field(default_factory=FineTuningConfigSchema)
    metrics: MetricsConfigSchema = field(default_factory=MetricsConfigSchema)
    scheduler: SchedulerConfigSchema = field(default_factory=SchedulerConfigSchema)
    evaluation: EvaluationConfigSchema = field(default_factory=EvaluationConfigSchema)

    # Global settings from the main config.yaml
    use_best_params: bool = False
    enable_cuda_optimizations: bool = True
