# Correcting Token Selection in Vision Transformers

This repository contains the implementation of our token pruning method applied across multiple vision tasks. Our approach improves token selection by combining CLS attention scores and column L4-norm scores.

## Overview

We integrate our token selection method into four baselines:

| Baseline | Task | Original Repo |
|----------|------|---------------|
| EViT | Image Classification | [youweiliang/evit](https://github.com/youweiliang/evit) |
| TCA | CLIP Zero-shot Classification | [Jo-wang/TCA](https://github.com/Jo-wang/TCA) |
| VisPruner | Multimodal LLM | [Theia-4869/VisPruner](https://github.com/Theia-4869/VisPruner) |
| LLaVA-PruMerge | Multimodal LLM | [42Shawn/LLaVA-PruMerge](https://github.com/42Shawn/LLaVA-PruMerge) |

For the image classification baseline, we also use [pytorch-image-models (timm)](https://github.com/huggingface/pytorch-image-models) for naive token pruning experiments.

## Installation

Clone each original repository and copy our modified files into the corresponding locations:

### 1. EViT (Image Classification)

```bash
git clone https://github.com/youweiliang/evit.git
cd evit

# Copy our modified files
cp /path/to/this/repo/evit/evit_correcting.py .
```

Then follow the original EViT setup instructions for dataset and checkpoint preparation.

**Fine-tuning:**
```bash
bash finetune.sh
```

**Evaluation:**
```bash
bash eval_vit.sh
```

---

### 2. pytorch-image-models / timm (Naive Pruning Baseline)

```bash
git clone https://github.com/huggingface/pytorch-image-models.git
cd pytorch-image-models

# Copy our modified ViT variants
cp /path/to/this/repo/pytorch-image-models/timm/models/vision_transformer_correcting.py timm/models/
cp /path/to/this/repo/pytorch-image-models/timm/models/vision_transformer_col_ln.py timm/models/
cp /path/to/this/repo/pytorch-image-models/timm/models/vision_transformer_cls.py timm/models/

pip install -e .
```

---

### 3. TCA (CLIP Zero-shot Classification)

```bash
git clone https://github.com/Jo-wang/TCA.git
cd TCA

# Copy our modified CLIP model
cp /path/to/this/repo/TCA/clip/model_col_ln.py clip/
```

Then follow TCA's original instructions for dataset preparation.

**Run:**
```bash
bash run.sh
```

---

### 4. VisPruner (Multimodal LLM)

```bash
git clone https://github.com/Theia-4869/VisPruner.git
cd VisPruner

# Copy our modified model directory
cp -r /path/to/this/repo/VisPruner/llava/model_correcting llava/

# Update llava/__init__.py to import from model_correcting
sed -i 's/from .model import/from .model_correcting import/' llava/__init__.py
```

Follow VisPruner's original setup for model checkpoints and evaluation data.

**MME Evaluation:**
```bash
bash scripts/v1_5/eval/mme.sh
```

---

### 5. LLaVA-PruMerge (Multimodal LLM)

```bash
git clone https://github.com/42Shawn/LLaVA-PruMerge.git
cd LLaVA-PruMerge

# Copy our modified CLIP encoder
cp /path/to/this/repo/LLaVA-PruMerge/llava/model/multimodal_encoder/clip_encoder_col_ln.py llava/model/multimodal_encoder/

# Switch builder to use our encoder
sed -i 's/from .clip_encoder import/from .clip_encoder_col_ln import/' llava/model/multimodal_encoder/builder.py
```

Follow LLaVA-PruMerge's original setup for evaluation.

---

## Key Files

| File | Description |
|------|-------------|
| `evit/evit_correcting.py` | Token selection: CLS-attention first, col-L4-norm second |
| `pytorch-image-models/timm/models/vision_transformer_correcting.py` | ViT integration for image classification |
| `pytorch-image-models/timm/models/vision_transformer_col_ln.py` | Col-L4-norm only variant |
| `pytorch-image-models/timm/models/vision_transformer_cls.py` | CLS-attention only variant |
| `TCA/clip/model_col_ln.py` | CLIP model with col-L4-norm token selection |
| `LLaVA-PruMerge/llava/model/multimodal_encoder/clip_encoder_col_ln.py` | CLIP encoder with col-L4-norm for LLaVA |
| `VisPruner/llava/model_correcting/llava_arch.py` | LLaVA architecture with corrected token selection |
