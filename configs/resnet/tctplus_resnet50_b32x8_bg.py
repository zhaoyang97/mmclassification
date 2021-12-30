_base_ = [
    '../_base_/models/resnet50.py', '../_base_/datasets/TCT_17w.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]

epoch_times = 1       # max_epochs = 100 * epoch_times
samples_per_gpu = 32  # batch_size = samples_per_gpu * gpus
gpus = 8              # please change it to the number of gpus you use

eval_interval = 10
checkpoint_interval = 10

model = dict(
    head=dict(
        num_classes=11,
    )
)
data = dict(
    samples_per_gpu=samples_per_gpu,
)
# lr=0.1 when batch_size=256
optimizer = dict(type='SGD', lr=0.1/256 * gpus*samples_per_gpu, momentum=0.9, weight_decay=0.0001)
lr_config = dict(policy='step', step=[30*epoch_times, 60*epoch_times, 90*epoch_times])
runner = dict(type='EpochBasedRunner', max_epochs=100*epoch_times)
evaluation = dict(interval=eval_interval, metric='accuracy')
checkpoint_config = dict(interval=checkpoint_interval)
