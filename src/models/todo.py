from src.models.groups import *

MODELS = [
    # "hmax",
    # "motion-energy",
    # "MotionNet",

    # # "UniFormer-V2-B",
    # "MViT-V2-B-Kinetics400",
    # "MViT-V2-B-smthsmthv2",
    # "I3D-R50-smthsmthv2",
    # "I3D-R101-smthsmthv2",
    # "VideoMAE-V1-B-smthsmthv2",
    # # "PredRNN",
    # # "VideoChat-7B",
    # # "VideoChat-13B",
    # "VJEPA-Temporal",
    # "TAU",

    # "Random-VideoMAE-V1-L",
    # "Random-UniFormer-V1",
    # "Random-s3d-HowTo100M",
    # "Random-resnext101_32x8d_wsl",
    # "Random-VideoSwin-B",
    # "Random-convnext_large_imagenet_full_seed-0",
    # "Random-VideoChat2-7B",

    # "hmax",
    # "IJEPA-ViT-H14-ImageNet1K",
    # "AlexNet_SIN",
    # "alexnet",
    # "I3D-nonlocal",
    # "MViT-V2-B-Kinetics400",
    # "PredRNN",
    # "TAU",
    # "s3d-HowTo100M",
    # "VideoChat-13B",
    # "Random-VideoMAE-V1-L",
    # "Random-UniFormer-V1",
    # "Random-s3d-HowTo100M",
    # "Random-resnext101_32x8d_wsl",
    # "Random-VideoSwin-B",
    # "Random-convnext_large_imagenet_full_seed-0",
    # "Random-VideoChat2-7B",
    # "mvit_v1_b",
    # "mvit_v2_s"

    # "S3D-afd101-0.001-32_1000",
    # "S3D-afd101-0.001-32_2000",
    # "S3D-afd101-0.001-32_3000",
    # "S3D-afd101-0.001-32_4000",
    # "S3D-afd101-0.001-32_5000"
    # "S3D-afd101-0.001-32_6000",
    # "S3D-afd101-0.001-32_7000",
    # "S3D-afd101-0.001-32_8000",
    # "S3D-afd101-0.001-32_9000",
    # "S3D-afd101-0.001-32_10000",

    # *FINETUNE_MODELS,

    # "Finetune-S3DHT-imagenet@afd101-0@0001-30_5000",
    # "Finetune-S3DHT-imagenet@afd101-0@0001-30_10000",
    # "Finetune-S3DHT-imagenet@afd101-0@0001-30_15000",
    # "Finetune-Uniformer-imagenet@afd101-1e-05-30_5000",

    # "Finetune-Uniformer-imagenet@afd101-1e-05-36@_lora_r2_25000",
    # "Finetune-Uniformer-imagenet@afd101-1e-05-36@_lora_r4_25000",
    # "Finetune-Uniformer-imagenet@afd101-1e-05-36@_lora_r8_25000",
    # "Finetune-Uniformer-imagenet@afd101-1e-06-36@_finetune_lora_r8_25000",
    # "Finetune-Uniformer-imagenet@afd101-1e-06-36@_lora_r8_25000",
    # "Finetune-Uniformer-imagenet@afd101-1e-05-36@_finetune_lora_r8_25000",
    # "Finetune-Uniformer-imagenet@afd101-1e-05-36@_finetune_lora_r4_25000",
    # "Finetune-VJEPA-imagenet@afd101-1e-05-30@_finetune_lora_r4_5000",
    # "Finetune-VJEPA-imagenet@afd101-1e-05-30@_lora_r4_5000",

    # "Finetune-VJEPA-imagenet.afd101-1e-05-30@_lora_r4_10000",
    # "Finetune-VJEPA-imagenet.afd101-1e-05-30@_finetune_lora_r4_10000",
    # "Finetune-VJEPA-imagenet.afd101-1e-05-30@_finetune_lora_r4_15000",
    # "Finetune-VJEPA-afd101-1e-05-30_15000",

    # "Finetune-VJEPA-imagenet.afd101-1e-05-30@_finetune_lora_r4_20000",
    # "Finetune-VJEPA-imagenet.afd101-1e-05-30@_lora_r4_20000",
    # "Finetune-Uniformer-imagenet.afd101-1e-06-36@_finetune_lora_r8_45000",
    # "Finetune-Uniformer-imagenet.afd101-1e-05-36@_lora_r8_60000",
    # "Finetune-Uniformer-imagenet.afd101-1e-05-36@_lora_r4_60000",
    # "Finetune-Uniformer-imagenet.afd101-1e-05-36@_lora_r2_60000",
    # "Finetune-Uniformer-imagenet.afd101-1e-05-36@_finetune_lora_r8_60000",
    # "Finetune-Uniformer-imagenet.afd101-1e-05-36@_finetune_lora_r4_60000",

    # "Finetune-VJEPA-imagenet.afd101-1e-05-30@_finetune_lora_r4_30000",
    # "Finetune-VJEPA-imagenet.afd101-1e-05-30@_lora_r4_30000",
    # "Finetune-Uniformer-imagenet.afd101-1e-05-36@_lora_r8_90000",
    # "Finetune-Uniformer-imagenet.afd101-1e-05-36@_lora_r4_90000",
    # "Finetune-Uniformer-imagenet.afd101-1e-05-36@_lora_r2_90000",
    # "Finetune-Uniformer-imagenet.afd101-1e-05-36@_finetune_lora_r8_90000",
    # "Finetune-Uniformer-imagenet.afd101-1e-05-36@_finetune_lora_r4_90000",

    # "Finetune-VJEPA-afd101-0.001-32@_finetune_1",
    # "Finetune-Uniformer-afd101-0.001-32@_finetune_1",

    # "Finetune-VJEPA-imagenet.afd101-1e-05-30@_finetune_lora_r4_40000",
    # "Finetune-VJEPA-imagenet.afd101-1e-05-30@_lora_r4_40000",

    # "Finetune-VJEPA-imagenet.afd101-1e-05-30@_finetune_lora_r4_35000",
    # "Finetune-VJEPA-imagenet.afd101-1e-05-30@_lora_r4_35000",

    # "Finetune-VJEPA-imagenet.afd101-1e-05-30@_finetune_lora_r4_50000",
    # "Finetune-VJEPA-imagenet.afd101-1e-05-30@_lora_r4_50000",

    # "Finetune-VJEPA-imagenet.afd101-1e-05-30@_finetune_lora_r4_60000",
    # "Finetune-VJEPA-imagenet.afd101-1e-05-30@_lora_r4_60000",

    # "Finetune-VJEPA-imagenet.afd101-1e-05-30@_finetune_lora_r4_70000",
    # "Finetune-VJEPA-imagenet.afd101-1e-05-30@_lora_r4_70000",
    # "Finetune-VJEPA-afd101-1e-05-30@_lora_r4_10000",
    # "Finetune-VJEPA-afd101-1e-05-30@_lora_r4_15000",

    # "Finetune-VJEPA-imagenet.afd101-1e-05-30@_finetune_lora_r4_80000",
    # "Finetune-VJEPA-imagenet.afd101-1e-05-30@_lora_r4_80000",
    # "Finetune-VJEPA-afd101-1e-05-30@_lora_r4_20000",

    # "NoDownsample-pixels",

    # "Finetune-VJEPA-imagenet.afd101-1e-05-30@_finetune_lora_r4_100000",
    # "Finetune-VJEPA-imagenet.afd101-1e-05-30@_lora_r4_100000",
    # "Finetune-VJEPA-afd101-1e-05-30@_lora_r4_35000",

    # "Finetune-VJEPA-afd101.smthsmthv2-1e-05-30@_lora_r4_20000",
    # "Finetune-VJEPA-afd101.smthsmthv2-1e-05-30@_lora_r4_25000",
    # "Finetune-VJEPA-afd101.smthsmthv2-1e-05-30@_lora_r4_35000",
    # "Finetune-VJEPA-afd101.smthsmthv2-1e-05-30@_lora_r4_45000",

    # "Finetune-VJEPA-smthsmthv2-1e-05-30@_lora_r4_10000",

    # "Finetune-VJEPA-smthsmthv2-1e-05-30@_lora_r4_15000",
    # "Finetune-VJEPA-afd101.smthsmthv2-1e-05-30@_lora_r4_55000",

    # "Finetune-VJEPA-smthsmthv2-1e-05-30@_lora_r4_25000",

    # "Finetune-VJEPA-smthsmthv2-1e-05-30@_lora_r4_50000",
    # "Finetune-VJEPA-afd101.smthsmthv2-1e-05-30@_lora_r4_85000",

    # "ViViT",
    # "VideoLlava",
    # "blt_temporal",
    # "DorsalNet",

    "hmax",
    "TAU",
    "SimVP",
]
