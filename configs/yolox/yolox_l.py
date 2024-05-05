_base_ = './yolox_s.py'

load_from = 'data/pretrained_weights/yolox_coco/yolox_l_fast_8xb8-300e_coco_20230213_160715-c731eb1c.pth'
# ========================modified parameters======================
deepen_factor = 1.0
widen_factor = 1.0

# =======================Unmodified in most cases==================
# model settings
model = dict(
    backbone=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    neck=dict(deepen_factor=deepen_factor, widen_factor=widen_factor),
    bbox_head=dict(head_module=dict(widen_factor=widen_factor)))
