# Correcting Token Selection in Vision Transformers

This repository contains our modified files to be integrated into existing codebases. Each file corresponds to a specific file in the original repository and replaces it with our corrected token selection method.

## File Correspondence

| Our File | Replace This File | In Repository |
|----------|------------------|---------------|
| `evit/evit_correcting.py` | `evit.py` | [youweiliang/evit](https://github.com/youweiliang/evit) |
| `evit/evit_original.py` | `evit.py` (original baseline) | [youweiliang/evit](https://github.com/youweiliang/evit) |
| `evit/evit_random.py` | `evit.py` (random baseline) | [youweiliang/evit](https://github.com/youweiliang/evit) |
| `evit/eval.sh` | `eval.sh` | [youweiliang/evit](https://github.com/youweiliang/evit) |
| `pytorch-image-models/timm/models/vision_transformer_correcting.py` | add to `timm/models/` | [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) |
| `pytorch-image-models/timm/models/vision_transformer_col_ln.py` | add to `timm/models/` | [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) |
| `pytorch-image-models/timm/models/vision_transformer_cls.py` | add to `timm/models/` | [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) |
| `TCA/clip/model_col_ln.py` | add to `clip/` | [Jo-wang/TCA](https://github.com/Jo-wang/TCA) |
| `LLaVA-PruMerge/llava/model/multimodal_encoder/clip_encoder_col_ln.py` | add to `llava/model/multimodal_encoder/` | [42Shawn/LLaVA-PruMerge](https://github.com/42Shawn/LLaVA-PruMerge) |
| `VisPruner/llava/model_correcting/` | replace `llava/model/` | [Theia-4869/VisPruner](https://github.com/Theia-4869/VisPruner) |

## Setup

### 1. EViT (Image Classification)

```bash
git clone https://github.com/youweiliang/evit.git
cd evit
cp /path/to/this/repo/evit/evit_correcting.py .
cp /path/to/this/repo/evit/evit_original.py .
cp /path/to/this/repo/evit/evit_random.py .
cp /path/to/this/repo/evit/eval.sh .
```

Fine-tuning:
```bash
bash finetune.sh
```

Evaluation:
```bash
bash eval.sh
```

---

### 2. pytorch-image-models / timm

```bash
git clone https://github.com/huggingface/pytorch-image-models.git
cd pytorch-image-models
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
cp /path/to/this/repo/TCA/clip/model_col_ln.py clip/
bash run.sh
```

---

### 4. VisPruner (Multimodal LLM)

```bash
git clone https://github.com/Theia-4869/VisPruner.git
cd VisPruner
cp -r /path/to/this/repo/VisPruner/llava/model_correcting llava/
sed -i 's/from .model import/from .model_correcting import/' llava/__init__.py
bash scripts/v1_5/eval/mme.sh
```

---

### 5. LLaVA-PruMerge (Multimodal LLM)

```bash
git clone https://github.com/42Shawn/LLaVA-PruMerge.git
cd LLaVA-PruMerge
cp /path/to/this/repo/LLaVA-PruMerge/llava/model/multimodal_encoder/clip_encoder_col_ln.py llava/model/multimodal_encoder/
sed -i 's/from .clip_encoder import/from .clip_encoder_col_ln import/' llava/model/multimodal_encoder/builder.py
```
