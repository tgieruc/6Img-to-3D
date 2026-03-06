"""Minimal local test config — uses seed4d data at ../seed4d/data."""

_base_ = [
    "./_base_/optimizer.py",
    "./_base_/triplane_decoder.py",
]

_dim_ = 128
num_heads = 8
_pos_dim_ = [40, 40, 48]
_ffn_dim_ = _dim_ * 2
_num_levels_ = 4
_max_cams_ = 6
_min_cams_train_ = 1

N_h_ = 50
N_w_ = 50
N_z_ = 8

offset_h = 0
offset_w = 0
offset_z = -4
offset = [offset_z, offset_h, offset_w]
scale_h = 0.25
scale_w = 0.25
scale_z = 0.25
scale = [scale_z, scale_h, scale_w]

scene_contraction = True
scene_contraction_factor = [0.5, 0.1, 0.1]

pif = False
pif_factor = 0.125

tpv_encoder_layers = 3
num_points_in_pillar = [4, 8, 8]
num_points = [4, 8, 8]
hybrid_attn_anchors = 4
hybrid_attn_points = 8
hybrid_attn_init = 0


self_cross_layer = dict(
    type="TPVFormerLayer",
    attn_cfgs=[
        dict(
            type="TPVImageCrossAttention",
            max_cams=_max_cams_,
            deformable_attention=dict(
                type="TPVMSDeformableAttention3D",
                embed_dims=_dim_,
                num_heads=num_heads,
                num_points=num_points,
                num_z_anchors=num_points_in_pillar,
                num_levels=_num_levels_,
                floor_sampling_offset=False,
                tpv_h=N_h_,
                tpv_w=N_w_,
                tpv_z=N_z_,
            ),
            embed_dims=_dim_,
            tpv_h=N_h_,
            tpv_w=N_w_,
            tpv_z=N_z_,
        ),
        dict(
            type="TPVCrossViewHybridAttention",
            tpv_h=N_h_,
            tpv_w=N_w_,
            tpv_z=N_z_,
            num_anchors=hybrid_attn_anchors,
            embed_dims=_dim_,
            num_heads=num_heads,
            num_points=hybrid_attn_points,
            init_mode=hybrid_attn_init,
        ),
    ],
    feedforward_channels=_ffn_dim_,
    ffn_dropout=0.1,
    operation_order=("cross_attn", "norm", "self_attn", "norm", "ffn", "norm"),
)

self_layer = dict(
    type="TPVFormerLayer",
    attn_cfgs=[
        dict(
            type="TPVCrossViewHybridAttention",
            tpv_h=N_h_,
            tpv_w=N_w_,
            tpv_z=N_z_,
            num_anchors=hybrid_attn_anchors,
            embed_dims=_dim_,
            num_heads=num_heads,
            num_points=hybrid_attn_points,
            init_mode=hybrid_attn_init,
        )
    ],
    feedforward_channels=_ffn_dim_,
    ffn_dropout=0.1,
    operation_order=("self_attn", "norm", "ffn", "norm"),
)


model = dict(
    type="TPVFormer",
    output_features=True,
    img_backbone=dict(
        type="ResNet",
        depth=101,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN2d", requires_grad=False),
        norm_eval=True,
        style="caffe",
        dcn=dict(type="DCNv2", deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True),
    ),
    img_neck=dict(
        type="FPN",
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs="on_output",
        num_outs=4,
        relu_before_extra_convs=True,
    ),
    tpv_head=dict(
        type="TPVFormerHead",
        tpv_h=N_h_,
        tpv_w=N_w_,
        tpv_z=N_z_,
        num_feature_levels=_num_levels_,
        max_cams=_max_cams_,
        embed_dims=_dim_,
        encoder=dict(
            type="TPVFormerEncoder",
            tpv_h=N_h_,
            tpv_w=N_w_,
            tpv_z=N_z_,
            offset=[offset_z, offset_h, offset_w],
            scale=[scale_z, scale_h, scale_w],
            intrin_factor=pif_factor,
            scene_contraction=scene_contraction,
            scene_contraction_factor=scene_contraction_factor,
            num_layers=tpv_encoder_layers,
            num_points_in_pillar=num_points_in_pillar,
            num_points_in_pillar_cross_view=[4, 4, 4],
            return_intermediate=False,
            transformerlayers=[
                self_cross_layer,
                self_cross_layer,
                self_layer,
            ],
        ),
        positional_encoding=dict(type="CustomPositionalEncoding", num_feats=_pos_dim_, h=N_h_, w=N_w_, z=N_z_),
    ),
)

dataset_params = dict(
    data_path="/home/bdw/Documents/seed4d/data/",
    version="triplane",
    train_data_loader=dict(
        pickled=False,
        phase="train",
        batch_size=1,
        shuffle=True,
        num_workers=4,
        town=["Town05"],
        weather=["ClearNoon"],
        vehicle=["vehicle.mini.cooper_s"],
        spawn_point=["all"],
        step=["all"],
        selection=["input_images", "sphere_dataset"],
        factor=0.08,
        img_factor=0.1,
        whole_image=True,
        num_imgs=1,
        depth=True,
        min_cams_train=1,
        max_cams_train=1,
    ),
    val_data_loader=dict(
        pickled=False,
        phase="test",
        batch_size=1,
        shuffle=False,
        num_workers=4,
        town=["Town02"],
        weather=["ClearNoon"],
        vehicle=["vehicle.mini.cooper_s"],
        spawn_point=["all"],
        step=["all"],
        selection=["input_images", "sphere_dataset"],
        factor=0.08,
        img_factor=0.1,
        depth=True,
    ),
)
