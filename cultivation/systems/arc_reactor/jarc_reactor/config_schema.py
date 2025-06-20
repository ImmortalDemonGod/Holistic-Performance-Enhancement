# cultivation/systems/arc_reactor/jarc_reactor/config_schema.py

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, ClassVar, Union
from enum import Enum

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
    checkpoint_path: Optional[str] = None

class DeviceChoice(Enum):
    AUTO = "auto"
    CPU = "cpu"
    GPU = "gpu"


class PrecisionOpt(Enum):
    BF16 = "bf16"
    MIXED_16 = "16-mixed"
    MIXED_32 = "32-mixed"


@dataclass
class TrainingConfigSchema:
    learning_rate: float = 0.00002009
    max_epochs: int = 100
    device_choice: DeviceChoice = DeviceChoice.AUTO
    precision: Union[int, PrecisionOpt] = 32
    gradient_clip_val: float = 1.0
    fast_dev_run: bool = False
    train_from_checkpoint: bool = False
    include_synthetic_training_data: bool = False
    current_data_is_synthetic: bool = False  # Indicates if the loaded dataset is synthetic
    training_data_dir: str = "${oc.env:TRAINING_DATA_DIR,cultivation/systems/arc_reactor/jarc_reactor/data/training_data/training}"
    synthetic_data_dir: Optional[str] = "cultivation/systems/arc_reactor/jarc_reactor/data/synthetic_data/training" # Path for synthetic training data
    training_log_dir: str = "cultivation/systems/arc_reactor/logs/training"  # Directory to save model checkpoints

@dataclass
class OptunaPruningConfig:
    n_warmup_steps: int = 5
    n_startup_trials: int = 10
    patience: int = 20
    pruning_percentile: int = 25

@dataclass
class IntRange:
    _target_: ClassVar[str] = "cultivation.systems.arc_reactor.jarc_reactor.config_schema.IntRange"
    """Defines an integer hyperparameter range for Optuna."""
    low: int
    high: int
    step: int = 1
    log: bool = False

    def __post_init__(self):
        if self.low >= self.high:
            raise ValueError(f"IntRange 'low' ({self.low}) must be less than 'high' ({self.high}).")
        if self.step <= 0:
            raise ValueError(f"IntRange 'step' ({self.step}) must be positive.")
        if self.log and self.low <= 0:
            raise ValueError(f"Log-scaled IntRange requires 'low' ({self.low}) to be positive.")

@dataclass
class FloatRange:
    _target_: ClassVar[str] = "cultivation.systems.arc_reactor.jarc_reactor.config_schema.FloatRange"
    """Defines a float hyperparameter range for Optuna."""
    low: float
    high: float
    step: Optional[float] = None
    log: bool = False

    def __post_init__(self):
        if self.low >= self.high:
            raise ValueError(f"FloatRange 'low' ({self.low}) must be less than 'high' ({self.high}).")
        if self.log and self.low <= 0:
            raise ValueError(f"Log-scaled FloatRange requires 'low' ({self.low}) to be positive.")
        if self.step is not None and self.step <= 0:
            raise ValueError(f"If provided, FloatRange 'step' ({self.step}) must be positive.")

@dataclass
class CategoricalChoice:
    _target_: ClassVar[str] = "cultivation.systems.arc_reactor.jarc_reactor.config_schema.CategoricalChoice"
    """Defines a categorical hyperparameter choice for Optuna."""
    choices: List[Any]

@dataclass
class OptunaConfigSchema:
    delete_study: bool = False
    n_trials: int = 1
    study_name: str = "jarc_optimization_v3"
    storage_url: str = "sqlite:///jarc_optuna.db"
    param_ranges: Dict[str, Any] = field(default_factory=lambda: {
        "d_model": IntRange(low=32, high=2048, step=16),
        "heads": CategoricalChoice(choices=[2, 4, 8, 16, 32, 64]),
        "encoder_layers": IntRange(low=1, high=12),
        "decoder_layers": IntRange(low=1, high=12),
        "d_ff": IntRange(low=64, high=2048, step=64),
        "dropout_rate": FloatRange(low=0.01, high=0.7),
        "context_encoder_d_model": IntRange(low=32, high=512, step=32),
        "context_encoder_heads": CategoricalChoice(choices=[2, 4, 8]),
        "batch_size": IntRange(low=8, high=512, log=True),
        "learning_rate": FloatRange(low=1e-6, high=1e-1, log=True),
        "max_epochs": IntRange(low=4, high=5, step=1),
        "gradient_clip_val": FloatRange(low=0.0, high=5.0)
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
    log_dir: str = "cultivation/systems/arc_reactor/logs/app"
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
    include_synthetic_data: bool = False
    synthetic_data_dir: Optional[str] = "cultivation/systems/arc_reactor/jarc_reactor/data/synthetic_data/evaluation" # Path for synthetic evaluation data

@dataclass
class DataLoaderConfig:
    batch_size: int = 1  # Add batch_size here
    num_workers: int = 0
    pin_memory: bool = False
    drop_last_train: bool = False  # For training dataloader
    drop_last_eval: bool = False   # For val/test dataloaders

@dataclass
class JARCReactorConfigSchema:
    # Defaults list for Hydra to know which sub-configs to load by default
    # This is usually in the main config.yaml, but can also be structured here if preferred for programmatic access
    # For now, we assume the defaults in config.yaml are primary.
    # defaults: List[Any] = field(default_factory=lambda: [
    #     {"model": "default"},
    #     {"training": "default"},
    #     {"optuna": "default"},
    #     {"logging": "default"},
    #     {"finetuning": "default"},
    #     {"metrics": "default"},
    #     {"scheduler": "default"},
    #     {"evaluation": "default"},
    #     "_self_"
    # ])

    model: ModelConfigSchema = field(default_factory=ModelConfigSchema)
    training: TrainingConfigSchema = field(default_factory=TrainingConfigSchema)
    optuna: OptunaConfigSchema = field(default_factory=OptunaConfigSchema)
    logging: LoggingConfigSchema = field(default_factory=LoggingConfigSchema)
    finetuning: FineTuningConfigSchema = field(default_factory=FineTuningConfigSchema)
    metrics: MetricsConfigSchema = field(default_factory=MetricsConfigSchema)
    scheduler: SchedulerConfigSchema = field(default_factory=SchedulerConfigSchema)
    evaluation: EvaluationConfigSchema = field(default_factory=EvaluationConfigSchema)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)

    # Global settings from config.yaml, merged via _self_
    use_best_params: bool = False
    enable_cuda_optimizations: bool = True
