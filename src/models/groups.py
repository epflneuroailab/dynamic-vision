DEBUG_MODEL_GROUP = [
    # dummy
    "pixels",

    # object recognition
    "alexnet",

    # action recognition
    "r3d_18",

    # masked autoencoder
    "VideoMAE-V1-B",

    # audio-video
    "AVID-CMA-Kinetics400",

    # forward prediction
    "PredRNN",
]

## Image Models

KADIR_MODELS = [
    "resnet18_imagenet_full",
    "resnet34_imagenet_full",
    "resnet50_imagenet_full",
    "resnet101_imagenet_full",
    "resnet152_imagenet_full",
    "resnet18_ecoset_full",
    "resnet34_ecoset_full",
    "resnet50_ecoset_full",
    "resnet101_ecoset_full",
    "resnet152_ecoset_full",
    "resnet50_imagenet_1_seed-0",
    "resnet50_imagenet_10_seed-0",
    "resnet50_imagenet_100_seed-0",
    "efficientnet_b0_imagenet_full",
    "efficientnet_b1_imagenet_full",
    "efficientnet_b2_imagenet_full",
    "deit_small_imagenet_full_seed-0",
    "deit_base_imagenet_full_seed-0",
    # "deit_large_imagenet_full_seed-0",
    "deit_small_imagenet_1_seed-0",
    "deit_small_imagenet_10_seed-0",
    "deit_small_imagenet_100_seed-0",
    "convnext_tiny_imagenet_full_seed-0",
    "convnext_small_imagenet_full_seed-0",
    "convnext_base_imagenet_full_seed-0",
    "convnext_large_imagenet_full_seed-0",
    "convnext_small_imagenet_1_seed-0",
    "convnext_small_imagenet_10_seed-0",
    "convnext_small_imagenet_100_seed-0",
]


COMMON_IMAGE_MODEL_GROUP = [
    "voneresnet-50-non_stochastic",
    "CORnet-S",
    "AlexNet_SIN",
    "alexnet"
]

BEST_IMAGE_MODEL_GROUP = [
    "resnext101_32x8d_wsl",
    # "resnext101_32x48d_wsl",
    # 'cvt_cvt-w24-384-in22k_finetuned-in1k_4',
]

IMAGE_MODELS = [
    # "IJEPA-ViT-H14-ImageNet1K",
    *COMMON_IMAGE_MODEL_GROUP,
    *BEST_IMAGE_MODEL_GROUP,
    *KADIR_MODELS,
]


## Video Models

ACTION_RECOGNITION_MODELS = [
    "r3d_18",
    "r2plus1d_18",
    "mc3_18",
    "s3d",
    "mvit_v1_b",
    "mvit_v2_s",
    "I3D",
    "I3D-nonlocal",
    "SlowFast",
    "X3D",
    "TimeSformer",
    "VideoSwin-B",
    "VideoSwin-L",
    "UniFormer-V1",
    "UniFormer-V2-B",
    # "UniFormer-V2-L",
    "MViT-V2-B-Kinetics400", 
    "MViT-V2-B-smthsmthv2",
    "I3D-R50-smthsmthv2",
    "I3D-R101-smthsmthv2",
    "ViViT",
]

MASKED_AUTOENCODER_MODELS = [
    "VideoMAE-V1-B",
    "VideoMAE-V1-B-smthsmthv2",
    "VideoMAE-V1-L",
    "VideoMAE-V2-B",
    "VideoMAE-V2-G",
    "MAE-ST-L",
    "MAE-ST-G",
    "VJEPA-Temporal",
]

AUDIO_VIDEO_MODELS = [
    "SeLaVi-Kinetics400",
    "SeLaVi-Kinetics-Sound",
    "SeLaVi-VGG-Sound",
    "SeLaVi-AVE",
    "AVID-CMA-Kinetics400",
    "AVID-CMA-Audioset",
    "AVID-Kinetics400",
    "AVID-Audioset",
    # "GDT-Kinetics400",
    # "GDT-IG65M",
]

FORWARD_PREDICTION_MODELS = [
    "ConvLSTM",
    "PredRNN",
    "TAU",
    "SimVP",
    "MIM",
]

TEXT_VIDEO_MODELS = [
    "s3d-HowTo100M",
    "VideoChat-7B",
    "VideoChat2-7B",
    "VideoChat-13B",
    "VideoLlava",
    # "GDT-HowTo100M",
]

RECURRENT_MODELS = [
    "ConvLSTM",
    "PredRNN",
    "MIM",
]

TEMPORAL_MODELS = [
    *ACTION_RECOGNITION_MODELS,
    *MASKED_AUTOENCODER_MODELS,
    *AUDIO_VIDEO_MODELS,
    *FORWARD_PREDICTION_MODELS,
    *TEXT_VIDEO_MODELS,
]


## Khaled's models
KHALED_MODELS = [
    "R3M-ResNet50-Temporal",
    "R3M-ResNet34-Temporal",
    "R3M-ResNet18-Temporal",

    "DFM-LSTM-SIM",
    "DFM-LSTM-SIM-OBSERVED",
    "DFM-LSTM-ENCODER",

    "DINO-LSTM-ENCODER",
    "DINO-LSTM-SIM",
    "DINO-LSTM-SIM-OBSERVED",

    "DINO-LARGE-Temporal",
    "DINO-BASE-Temporal",
    "DINO-GIANT-Temporal",

    "FEATSGT-Temporal",

    # "FITVID-EGO4D-OBSERVED",
    # "FITVID-PHYS-OBSERVED",

    # "FITVID-EGO4D-SIM",
    # "FITVID-PHYS-SIM",

    # "MAE-BASE-Temporal",
    # "MAE-LARGE-Temporal",

    # "MAE-LSTM-ENCODER",
    # "MAE-LSTM-SIM",
    # "MAE-LSTM-SIM-OBSERVED",

    "MCVD-EGO4D-OBSERVED",
    "MCVD-PHYS-OBSERVED",

    # "MCVD-EGO4D-SIM",
    # "MCVD-PHYS-SIM",

    "PixelNerf-Temporal",

    "PN-LSTM-ENCODER",
    "PN-LSTM-SIM",
    "PN-LSTM-SIM-OBSERVED",

    "R3M-LSTM-EGO4D-ENCODER",
    "R3M-LSTM-EGO4D-SIM",
    "R3M-LSTM-EGO4D-SIM-OBSERVED",
    "R3M-LSTM-PHYS-ENCODER",
    "R3M-LSTM-PHYS-SIM",
    "R3M-LSTM-PHYS-SIM-OBSERVED",

    "RAWGT-Temporal",

    "ResNet152-Temporal",
    "ResNet101-Temporal",
    "ResNet50-Temporal",
    "ResNet34-Temporal",
    "ResNet18-Temporal",

    "RESNET-LSTM-ENCODER",
    "RESNET-LSTM-SIM",
    "RESNET-LSTM-SIM-OBSERVED",

    "VJEPA-Temporal",
]

CONTROLS = [
    "hmax",
    "motion-energy",
    "pixels",
    "MotionNet",
]

RANDOM_MODELS = [
    "Random-VideoMAE-V1-L",
    "Random-UniFormer-V1",
    "Random-s3d-HowTo100M",
    "Random-resnext101_32x8d_wsl",
    "Random-VideoSwin-B",
    "Random-convnext_large_imagenet_full_seed-0",
    "Random-VideoChat2-7B",
]

EXTRA_MODELS = [
    *[f"S3D-afd101-0.001-32_{i}"
    for i in range(10_000, 100_000 + 1, 10_000)],
    *[f"S3D-afd101-0.001-32_{i}"
    for i in range(1_000, 10_000, 1_000)],
]

FINETUNE_MODELS = [
    "Finetune-Uniformer-afd101-1e-05-30_20000",
    "Finetune-Uniformer-imagenet-1e-05-30_95000",
    "Finetune-VideoMAE-afd101-1e-05-10_25000",
    "Finetune-VideoMAE-imagenet-1e-05-10_20000",
    "Finetune-VJEPA-afd101-1e-05-30_10000",
    "Finetune-VJEPA-imagenet-1e-05-30_30000",
    "Finetune-S3DHT-afd101-1e-05-30_70000",
    "Finetune-S3DHT-imagenet-1e-05-30_100000",
    "Finetune-S3DHT-imagenet@afd101-0@0001-30_5000",
    "Finetune-S3DHT-imagenet@afd101-0@0001-30_10000",
    "Finetune-S3DHT-imagenet@afd101-0@0001-30_15000",
    "Finetune-Uniformer-imagenet@afd101-0@0001-30_5000",
    "Finetune-Uniformer-imagenet@afd101-1e-05-36@_finetune_lora_r4_10000",
    "Finetune-Uniformer-imagenet@afd101-1e-05-36@_finetune_lora_r4_5000",
    "Finetune-Uniformer-imagenet@afd101-1e-05-36@_finetune_lora_r8_10000",
    "Finetune-Uniformer-imagenet@afd101-1e-05-36@_finetune_lora_r8_5000",

    "Finetune-Uniformer-imagenet@afd101-1e-05-36@_lora_r2_25000",
    "Finetune-Uniformer-imagenet@afd101-1e-05-36@_lora_r4_25000",
    "Finetune-Uniformer-imagenet@afd101-1e-05-36@_lora_r8_25000",
    "Finetune-Uniformer-imagenet@afd101-1e-06-36@_finetune_lora_r8_25000",
    "Finetune-Uniformer-imagenet@afd101-1e-06-36@_lora_r8_25000",
    "Finetune-Uniformer-imagenet@afd101-1e-05-36@_finetune_lora_r8_25000",
    "Finetune-Uniformer-imagenet@afd101-1e-05-36@_finetune_lora_r4_25000",
    "Finetune-VJEPA-imagenet@afd101-1e-05-30@_finetune_lora_r4_5000",
    "Finetune-VJEPA-imagenet@afd101-1e-05-30@_lora_r4_5000",

    "Finetune-VJEPA-imagenet.afd101-1e-05-30@_lora_r4_10000",
    "Finetune-VJEPA-imagenet.afd101-1e-05-30@_finetune_lora_r4_10000",
    "Finetune-VJEPA-imagenet.afd101-1e-05-30@_finetune_lora_r4_15000",
    "Finetune-VJEPA-afd101-1e-05-30_15000",

    "Finetune-VJEPA-imagenet.afd101-1e-05-30@_finetune_lora_r4_20000",
    "Finetune-VJEPA-imagenet.afd101-1e-05-30@_lora_r4_20000",
    "Finetune-Uniformer-imagenet.afd101-1e-06-36@_finetune_lora_r8_45000",
    "Finetune-Uniformer-imagenet.afd101-1e-05-36@_lora_r8_60000",
    "Finetune-Uniformer-imagenet.afd101-1e-05-36@_lora_r4_60000",
    "Finetune-Uniformer-imagenet.afd101-1e-05-36@_lora_r2_60000",
    "Finetune-Uniformer-imagenet.afd101-1e-05-36@_finetune_lora_r8_60000",
    "Finetune-Uniformer-imagenet.afd101-1e-05-36@_finetune_lora_r4_60000",

    "Finetune-VJEPA-imagenet.afd101-1e-05-30@_finetune_lora_r4_30000",
    "Finetune-VJEPA-imagenet.afd101-1e-05-30@_lora_r4_30000",
    "Finetune-Uniformer-imagenet.afd101-1e-05-36@_lora_r8_90000",
    "Finetune-Uniformer-imagenet.afd101-1e-05-36@_lora_r4_90000",
    "Finetune-Uniformer-imagenet.afd101-1e-05-36@_lora_r2_90000",
    "Finetune-Uniformer-imagenet.afd101-1e-05-36@_finetune_lora_r8_90000",
    "Finetune-Uniformer-imagenet.afd101-1e-05-36@_finetune_lora_r4_90000",

    "Finetune-VJEPA-afd101-0.001-32@_finetune_1",
    "Finetune-Uniformer-afd101-0.001-32@_finetune_1",

    "Finetune-VJEPA-imagenet.afd101-1e-05-30@_finetune_lora_r4_40000",
    "Finetune-VJEPA-imagenet.afd101-1e-05-30@_lora_r4_40000",

    "Finetune-VJEPA-imagenet.afd101-1e-05-30@_finetune_lora_r4_35000",
    "Finetune-VJEPA-imagenet.afd101-1e-05-30@_lora_r4_35000",

    "Finetune-VJEPA-imagenet.afd101-1e-05-30@_finetune_lora_r4_50000",
    "Finetune-VJEPA-imagenet.afd101-1e-05-30@_lora_r4_50000",

    "Finetune-VJEPA-imagenet.afd101-1e-05-30@_finetune_lora_r4_60000",
    "Finetune-VJEPA-imagenet.afd101-1e-05-30@_lora_r4_60000",

    "Finetune-VJEPA-imagenet.afd101-1e-05-30@_finetune_lora_r4_70000",
    "Finetune-VJEPA-imagenet.afd101-1e-05-30@_lora_r4_70000",
    "Finetune-VJEPA-afd101-1e-05-30@_lora_r4_10000",
    "Finetune-VJEPA-afd101-1e-05-30@_lora_r4_15000",

    "Finetune-VJEPA-imagenet.afd101-1e-05-30@_finetune_lora_r4_80000",
    "Finetune-VJEPA-imagenet.afd101-1e-05-30@_lora_r4_80000",
    "Finetune-VJEPA-afd101-1e-05-30@_lora_r4_20000",

    "Finetune-VJEPA-imagenet.afd101-1e-05-30@_finetune_lora_r4_100000",
    "Finetune-VJEPA-imagenet.afd101-1e-05-30@_lora_r4_100000",
    "Finetune-VJEPA-afd101-1e-05-30@_lora_r4_35000",

    "Finetune-VJEPA-afd101.smthsmthv2-1e-05-30@_lora_r4_20000",
    "Finetune-VJEPA-afd101.smthsmthv2-1e-05-30@_lora_r4_25000",
    "Finetune-VJEPA-afd101.smthsmthv2-1e-05-30@_lora_r4_35000",
    "Finetune-VJEPA-afd101.smthsmthv2-1e-05-30@_lora_r4_45000",

    "Finetune-VJEPA-smthsmthv2-1e-05-30@_lora_r4_10000",

    "Finetune-VJEPA-smthsmthv2-1e-05-30@_lora_r4_15000",
    "Finetune-VJEPA-afd101.smthsmthv2-1e-05-30@_lora_r4_55000",

    "Finetune-VJEPA-smthsmthv2-1e-05-30@_lora_r4_25000",

    "Finetune-VJEPA-smthsmthv2-1e-05-30@_lora_r4_50000",
    "Finetune-VJEPA-afd101.smthsmthv2-1e-05-30@_lora_r4_85000",
]

ALL_MODELS = [
    "pixels",
    "hmax",
    "motion-energy",
    "MotionNet",
    # "DorsalNet",
    # "blt_temporal",
    *IMAGE_MODELS,
    *TEMPORAL_MODELS,
    *RANDOM_MODELS,
    # *FINETUNE_MODELS,
    # *KHALED_MODELS,
    # *EXTRA_MODELS,
    # "VJEPA-Temporal",
]

STATIC_MODELS = [
    "pixels",
    "hmax",
    *IMAGE_MODELS,
]

# Temporal window size
FLEXIBLE_TEMPORAL_MODELS = [
    "r3d_18",
    "r2plus1d_18",
    "mc3_18",
    "mvit_v1_b",
    "mvit_v2_s",
    "X3D",
    "TimeSformer",
    "VideoSwin-B",
    "VideoSwin-L",
    "UniFormer-V1",
    "SeLaVi-Kinetics400",
    "SeLaVi-Kinetics-Sound",
    "SeLaVi-VGG-Sound",
    "SeLaVi-AVE",
    "AVID-CMA-Kinetics400",
    "AVID-CMA-Audioset",
    "AVID-Kinetics400",
    "AVID-Audioset",
    "s3d-HowTo100M",
    "ConvLSTM",
    "PredRNN",
]

AVOID = [
    "deit_large_imagenet_full_seed-0",
    "resnext101_32x48d_wsl",
    "UniFormer-V2-L",
    "cvt_cvt-w24-384-in22k_finetuned-in1k_4",
    "MAE-LSTM-SIM-OBSERVED",
    "DFM-LSTM-SIM",
    "DFM-LSTM-SIM-OBSERVED",
    "DFM-LSTM-ENCODER",
    "FEATSGT-Temporal",
    "MCVD-EGO4D-OBSERVED",
    "MCVD-PHYS-OBSERVED",
    "RAWGT-Temporal",
    "ResNet50-Temporal",
    "RESNET-LSTM-ENCODER",
    "RESNET-LSTM-SIM",
    "RESNET-LSTM-SIM-OBSERVED",

    "DINO-LSTM-ENCODER",
    "DINO-LSTM-SIM",
    "DINO-LSTM-SIM-OBSERVED",
    "DINO-LARGE-Temporal",
    "DINO-BASE-Temporal",
    "DINO-GIANT-Temporal",
    "PixelNerf-Temporal",
    "PN-LSTM-ENCODER",
    "PN-LSTM-SIM",
    "PN-LSTM-SIM-OBSERVED",
]
