_base_ = [
    '../_base_/models/resnet50.py', '../_base_/datasets/i_imagenet.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]

model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet',
        # type='ResNet_carafed',
        # type='ResNet_carafed_3_kernelexp',
        depth=50,
    )
)