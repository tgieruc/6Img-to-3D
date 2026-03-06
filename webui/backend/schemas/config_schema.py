from pydantic import BaseModel, ConfigDict, Field

# NOTE: Some field names differ from the raw config.py keys by design.
# The config_io module handles translation in both directions:
#   schema "towns" <-> config.py "town"
#   schema "num_warmup_steps" <-> config.py "num_training_steps"


class PIFConfig(BaseModel):
    enabled: bool = False
    factor: float = 0.125
    transforms_path: str = ""


class EncoderConfig(BaseModel):
    dim: int = 128
    num_heads: int = 8
    num_levels: int = 4
    max_cams: int = 6
    min_cams_train: int = 1
    tpv_h: int = 200
    tpv_w: int = 200
    tpv_z: int = 16
    num_encoder_layers: int = 5
    scene_contraction: bool = True
    scene_contraction_factor: list[float] = Field(default_factory=lambda: [0.5, 0.1, 0.1])
    offset: list[float] = Field(default_factory=lambda: [-4.0, 0.0, 0.0])
    scale: list[float] = Field(default_factory=lambda: [0.25, 0.25, 0.25])
    num_points_in_pillar: list[int] = Field(default_factory=lambda: [4, 32, 32])
    num_points: list[int] = Field(default_factory=lambda: [8, 64, 64])
    hybrid_attn_anchors: int = 16
    hybrid_attn_points: int = 32


class DecoderConfig(BaseModel):
    hidden_dim: int = 128
    hidden_layers: int = 5
    density_activation: str = "trunc_exp"
    nb_bins: int = 64
    nb_bins_sample: int = 64
    hn: float = 0.0
    hf: float = 60.0
    train_stratified: bool = True
    white_background: bool = False
    whiteout: bool = False
    testing_batch_size: int = 8192


class OptimizerConfig(BaseModel):
    lr: float = 5e-5
    num_epochs: int = 100
    num_warmup_steps: int = 1000
    lpips_loss_weight: float = 0.2
    tv_loss_weight: float = 0.0
    dist_loss_weight: float = 1e-3
    depth_loss_weight: float = 1.0
    clip_grad_norm: float = 1.5


class TrainLoaderConfig(BaseModel):
    pickled: bool = True
    batch_size: int = 1
    shuffle: bool = True
    num_workers: int = 12
    towns: list[str] = Field(
        default_factory=lambda: ["Town01", "Town03", "Town04", "Town05", "Town06", "Town07", "Town10HD"]
    )
    weather: list[str] = Field(default_factory=lambda: ["ClearNoon"])
    vehicle: list[str] = Field(default_factory=lambda: ["vehicle.tesla.invisible"])
    factor: float = 0.08
    num_imgs: int = 3
    depth: bool = True
    min_cams_train: int = 1
    max_cams_train: int = 6


class ValLoaderConfig(BaseModel):
    phase: str = "test"
    pickled: bool = False
    batch_size: int = 1
    num_workers: int = 12
    towns: list[str] = Field(default_factory=lambda: ["Town02"])
    weather: list[str] = Field(default_factory=lambda: ["ClearNoon"])
    vehicle: list[str] = Field(default_factory=lambda: ["vehicle.tesla.invisible"])
    spawn_point: list[int] = Field(default_factory=lambda: [3, 7, 12, 48, 98, 66])
    factor: float = 0.25
    depth: bool = True


class DatasetConfig(BaseModel):
    data_path: str = "/app/data/"
    train: TrainLoaderConfig = Field(default_factory=TrainLoaderConfig)
    val: ValLoaderConfig = Field(default_factory=ValLoaderConfig)


class FullConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    decoder: DecoderConfig = Field(default_factory=DecoderConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    pif: PIFConfig = Field(default_factory=PIFConfig)
