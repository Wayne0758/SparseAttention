# Implementation

This repository contains our modified files to be integrated into existing codebases. Each file corresponds to a specific file in the original repository and replaces it with our corrected token selection method.

## File Correspondence

| Our File | Replace This File | In Repository |
|----------|------------------|---------------|
| `evit/evit_correcting.py` | `evit.py` | [youweiliang/evit](https://github.com/youweiliang/evit) |
| `evit/evit_original.py` | `evit.py` (original baseline) | [youweiliang/evit](https://github.com/youweiliang/evit) |
| `evit/evit_random.py` | `evit.py` (random baseline) | [youweiliang/evit](https://github.com/youweiliang/evit) |
| `evit/eval.sh` | new | [youweiliang/evit](https://github.com/youweiliang/evit) |
| `pytorch-image-models/timm/models/vision_transformer_correcting.py` | `timm/models/vision_transformer.py` | [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) |
| `pytorch-image-models/timm/models/vision_transformer_col_ln.py` | `timm/models/vision_transformer.py` | [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) |
| `pytorch-image-models/timm/models/vision_transformer_cls.py` | `timm/models/vision_transformer.py` | [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) |
| `TCA/clip/model_col_ln.py` | `clip/model.py` | [Jo-wang/TCA](https://github.com/Jo-wang/TCA) |
| `LLaVA-PruMerge/llava/model/multimodal_encoder/clip_encoder_col_ln.py` | `llava/model/multimodal_encoder/clip_encoder.py` | [42Shawn/LLaVA-PruMerge](https://github.com/42Shawn/LLaVA-PruMerge) |
| `VisPruner/llava/model_correcting/` | `llava/model/` | [Theia-4869/VisPruner](https://github.com/Theia-4869/VisPruner) |

**Note:** The original EViT repository only supports DeiT backbones. We re-implemented `evit_original.py` and `evit_random.py` to support ViT (augreg) backbones. When running inference with ViT, the `--inception_norm` flag must be passed to `eval.sh`.
