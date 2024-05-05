_base_ = ['co_dino_r50.py']

# # to get pretrained resnet weights, which might be used when training co_dino from scratch: wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth
# swin_path = 'data/pretrained_weights/codino_coco/swin_large_patch4_window12_384_22k.pth'
load_from = 'data/pretrained_weights/codino_coco/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth'

default_hooks = dict(
    logger=dict(interval=304, type='LoggerHook')) # assuming batch size of 4x1 (4 gpus), this logs twice per epoch

# model settings
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        with_cp=True,
        convert_weights=True,
        # init_cfg=dict(type='Pretrained', checkpoint=swin_path)), # if training from scratch, you might use this pretrained swin checkpoint
        init_cfg=None),
    neck=dict(in_channels=[192, 384, 768, 1536]),
    query_head=dict(
        dn_cfg=dict(box_noise_scale=0.4, group_cfg=dict(num_dn_queries=500)),
        transformer=dict(encoder=dict(with_cp=6))))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 480), (512, 512), (544, 544), (576, 576),
                            (608, 608), (640, 640), (672, 672), (704, 704),
                            (736, 736), (768, 768), (800, 800), (832, 832),
                            (864, 864), (896, 896), (928, 928), (960, 960),
                            (992, 992), (1024, 1024), (1056, 1056),
                            (1088, 1088), (1120, 1120), (1152, 1152),
                            (1184, 1184), (1216, 1216), (1248, 1248),
                            (1280, 1280), (1312, 1312), (1344, 1344),
                            (1376, 1376), (1408, 1408), (1440, 1440)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 400), (500, 500), (600, 600)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 480), (512, 512), (544, 544), (576, 576),
                            (608, 608), (640, 640), (672, 672), (704, 704),
                            (736, 736), (768, 768), (800, 800), (832, 832),
                            (864, 864), (896, 896), (928, 928), (960, 960),
                            (992, 992), (1024, 1024), (1056, 1056),
                            (1088, 1088), (1120, 1120), (1152, 1152),
                            (1184, 1184), (1216, 1216), (1248, 1248),
                            (1280, 1280), (1312, 1312), (1344, 1344),
                            (1376, 1376), (1408, 1408), (1440, 1440)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='PackDetInputs')
]

train_dataloader = dict(
    batch_size=1, num_workers=1, dataset=dict(pipeline=train_pipeline))

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=_base_.image_size, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
optim_wrapper = dict(optimizer=dict(lr=1e-4))
