# Causal Multimodal Learning for Glaucoma Detection

PyTorch implementation of Causal Representation Learning for multimodal (image + text) classification on the FairCLIP medical imaging dataset.

## Reference

This work is based on the following paper:

**Causal Representation Learning for Multimodal Medical Imaging**  
https://arxiv.org/pdf/2407.14058

## Dataset

The dataset used in this project is **Harvard-FairCLIP** dataset. For dataset access and information, please refer to:

**https://github.com/Harvard-Ophthalmology-AI-Lab/FairCLIP**


## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
Glaucoma_detection_Causal_Representation_learning/
├── backbone/                      # Encoder architectures
│   ├── image_encoder.py          # Image encoders (ViT)
│   ├── image_encoder_efficientnet.py
│   ├── image_encoder_resnet.py
│   ├── image_encoder_vgg.py
│   ├── text_encoder.py           # Text encoders (BERT)
│   ├── text_encoder_deberta.py
│   ├── text_encoder_distilbert.py
│   └── text_encoder_roberta.py
├── data/                          # Dataset loaders
│   └── fairclip_dataset.py       # FairCLIP dataset loader
├── models/                        # Model architectures
│   ├── causal_multimodal_vit_bert.py
│   ├── causal_multimodal_efficientnet_distilbert.py
│   ├── causal_multimodal_resnet_roberta.py
│   ├── causal_multimodal_vgg_deberta.py
│   └── causal_multimodal_base.py
├── train/                         # Training scripts
│   ├── train_vit_bert.py
│   ├── train_efficientnet_distilbert.py
│   ├── train_resnet_roberta.py
│   └── train_vgg_deberta.py
├── run/                           # Bash execution scripts
│   ├── run_vit_bert.sh
│   ├── run_efficientnet_distilbert.sh
│   ├── run_resnet_roberta.sh
│   ├── run_vgg_deberta.sh
│   └── evaluate_all_models.sh
├── evaluation/                    # Evaluation scripts
│   └── evaluate_checkpoint.py
├── utils/                         # Utility functions
│   └── utils.py
├── logs/                          # Logging utilities
│   └── logger.py
├── test/                          # Test scripts
│   └── test_setup.py
├── checkpoints/                   # Model checkpoints
├── evaluation_results/           # Evaluation results
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Training

### Using Bash Scripts

```bash
# ViT + BERT
bash run/run_vit_bert.sh [experiment_name] [gpu_id]

# EfficientNet + DistilBERT
bash run/run_efficientnet_distilbert.sh [experiment_name] [gpu_id]

# ResNet + RoBERTa
bash run/run_resnet_roberta.sh [experiment_name] [gpu_id]

# VGG + DeBERTa
bash run/run_vgg_deberta.sh [experiment_name] [gpu_id]
```

**Example:**
```bash
bash run/run_efficientnet_distilbert.sh efficientnet_distilbert 1
```

## Evaluation

### Evaluate All Models

```bash
bash run/evaluate_all_models.sh [gpu_id] [output_dir]
```

**Example:**
```bash
bash run/evaluate_all_models.sh 0
```

### Evaluate Single Checkpoint

```bash
python evaluation/evaluate_checkpoint.py \
    --checkpoint_path ./checkpoints/vit_bert/model_best.pt \
    --data_dir /medailab/medailab/shilab/FairCLIP \
    --csv_path /medailab/medailab/shilab/FairCLIP/data_summary.csv \
    --gpu_id 0
```

## Supported Models

| Model | Image Encoder | Text Encoder |
|-------|--------------|--------------|
| **ViT+BERT** | Vision Transformer | BERT-base-uncased |
| **EfficientNet+DistilBERT** | EfficientNet-B0 | DistilBERT-base |
| **ResNet+RoBERTa** | ResNet152 | RoBERTa-base |
| **VGG+DeBERTa** | VGG16 | DeBERTa-base |

## Outputs

- **Checkpoints**: `./checkpoints/{experiment_name}/`
  - `checkpoint.pt`: Latest checkpoint
  - `model_best.pt`: Best model (highest validation F1)
- **Evaluation Results**: `./evaluation_results/`
  - `summary_all_models.csv`: Summary of all model evaluations

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch_size` | 16 | Batch size for training |
| `--max_epochs` | 50 | Maximum training epochs |
| `--lr` | 2e-5 | Learning rate |
| `--lambda_v` | 1.0 | Weight for KL divergence loss |
| `--lambda_fe` | 1.0 | Weight for feature extraction loss |
| `--patience` | 15 | Early stopping patience |

## Citation

If you use this code in your research, please cite:

```bibtex
@article{causal2024,
  title={Causal Representation Learning for Multimodal Medical Imaging},
  author={...},
  journal={arXiv preprint arXiv:2407.14058},
  year={2024},
  url={https://arxiv.org/pdf/2407.14058}
}
```


