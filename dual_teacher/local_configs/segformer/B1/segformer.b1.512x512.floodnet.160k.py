_base_ = [
    '../../_base_/models/segformer.py',
    '../../_base_/datasets/floodnet.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
# norm_cfg = dict(type='BN', requires_grad=True)

find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/mit_b1.pth',
    backbone=dict(
        type='mit_b1',
        style='pytorch'),
    decode_head=dict(
        type='SegFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.2,
        num_classes=10,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=256),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0002, betas=(0.90, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

scheduler = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1250,  # Increased warmup iterations
                 warmup_ratio=5e-7,  # Increased warmup ratio
                 power=1,  # More aggressive decay
                 min_lr=0.0, by_epoch=False)

# checkpoint settings
checkpoint_config = dict(interval=1)  # Save checkpoints every epoch

# working directory
work_dir = './work_dirs/segformer.b1.512x512.floodnet.160k.py'

data = dict(samples_per_gpu=4)
evaluation = dict(interval=8000, metric='mIoU')

