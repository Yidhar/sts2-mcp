from dataclasses import dataclass, field

from sts2_rl.artifacts import resolve_artifact_path


@dataclass
class TrainConfig:
    # Environment
    session_file: str | None = None
    character: str | None = None
    defensive_buffs: bool = False
    reset_timeout_ms: int = 60000
    step_timeout_ms: int = 20000

    # Training
    total_timesteps: int = 100_000
    learning_rate: float = 3e-4
    n_steps: int = 64
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # Network
    embed_dim: int = 64
    text_proj_dim: int = 32
    n_heads: int = 2
    scorer_hidden: int = 64
    net_arch: list = field(default_factory=lambda: [128, 128])

    # Text
    use_text: bool = True
    text_model_name: str = "BAAI/bge-small-zh-v1.5"
    text_cache_path: str | None = None
    normalize_text_embeddings: bool = True
    use_decision_text: bool = True
    use_action_text: bool = True

    # Logging
    log_dir: str | None = None
    checkpoint_dir: str | None = None
    checkpoint_freq: int = 1000
    tensorboard: bool = True
    verbose: int = 1

    # Device
    device: str = "cpu"

    def __post_init__(self) -> None:
        log_dir = resolve_artifact_path(self.log_dir, default="runs")
        checkpoint_dir = resolve_artifact_path(self.checkpoint_dir, default="checkpoints")
        log_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = str(log_dir)
        self.checkpoint_dir = str(checkpoint_dir)
