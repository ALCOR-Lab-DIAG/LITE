# LITE: Look at the Image and Trust the Expert

Official implementation of **LITE**, accepted at the IEEE International Workshop on Information Forensics and Security (**WIFS 2026**).

**Alessandro Di Chiara, Simone Teglia, Irene Amerini**
Sapienza University of Rome

[[Paper]([LINK])] [[Checkpoints]([LINK])] [[B-RRDataset splits]([LINK])]

---

## Overview

AI-generated images rarely reach a fact-checker in pristine form. They are compressed and resized by social platforms, shared repeatedly, or photographed from a screen. These transformations change or remove the artifacts that detectors rely on.

LITE is a condition-aware detector built on this observation. It works in two steps:

1. An **Image-Condition Classifier** (DINOv3 ViT-L/16) estimates the acquisition condition of the input image: *original*, *social-media transferred* or *re-digitized*.
2. The image is routed to a single **condition-specific VLM expert**, fine-tuned with LoRA only on images of that condition, which outputs `REAL` or `FAKE`.

| Condition | Expert |
|---|---|
| Original | Qwen2.5-VL-7B-Instruct |
| Social-media transfer | Pixtral-12B-2409 |
| Re-digitization | Pixtral-12B-2409 |

Each expert backbone was chosen as the best zero-shot VLM on its condition, among six open-source VLMs evaluated on a separate benchmark split.

<p align="center">
  <img src="assets/lite_architecture.png" width="80%" alt="LITE architecture">
</p>

---

## Results

### B-RRDataset (in-distribution)

All detectors are fine-tuned on the same B-RRDataset training split and evaluated on the same 6K test images.

| Model | Original Acc | Social Acc | Re-dig. Acc | Overall Acc | Overall F1 |
|---|---|---|---|---|---|
| CNNSpot | 0.957 | 0.950 | **0.956** | 0.954 | 0.954 |
| DRCT-ConvB | 0.960 | 0.945 | 0.928 | 0.946 | 0.947 |
| Single Qwen2.5-VL-7B (all conditions) | **0.983** | 0.975 | 0.922 | 0.960 | 0.959 |
| **LITE** | 0.982 | **0.976** | 0.942 | **0.967** | **0.966** |

A single VLM fine-tuned on all three conditions matches LITE on original and transferred images. The gain from routing comes from re-digitized images, where the single model misses many recaptured synthetic images (FNR 0.108 vs. 0.077 for LITE).

### Cross-dataset (no additional fine-tuning)

The re-digitization expert is disabled on these benchmarks, since they contain no screen-captured images.

- **SID-Set** (Real and Fully Synthetic): 98.6% accuracy on real images and 99.9% on fully synthetic images.
- **TrueFake** (images shared on Facebook, Telegram and X): LITE has the highest TPR on every generator and platform (0.953 to 0.998). Class-balanced accuracy, averaging the TNR over the two real sources and the TPR over the eight generators, is 0.720 / 0.727 / 0.727 on FB / TL / X.

> **Known limitation.** LITE produces many false positives on FFHQ face images (TNR 0.083 to 0.122 after sharing), while its TNR on FORLAB stays between 0.815 and 0.848. It is not suited, in its current form, to settings dominated by aligned facial portraits. See Section V-C of the paper.

---

## Installation

```bash
git clone https://github.com/[ORG]/LITE.git
cd LITE
conda create -n lite python=[3.X] -y
conda activate lite
pip install -r requirements.txt
```

FlashAttention 2 and Liger Kernel require a compatible CUDA toolkit. [Add install notes if needed.]

---

## Data

### B-RRDataset

B-RRDataset is a balanced subset of [RRDataset](https://github.com/[RRDATASET_REPO]) (Li et al., ICCV 2025), with equal numbers of real and fake images in each of the three acquisition conditions.

| Split | Images | Use |
|---|---|---|
| Zero-shot benchmark | 6,000 (2,000 per condition) | VLM selection only |
| Train / validation | 18,000 | Classifier and expert fine-tuning |
| Test | 6,000 (2,000 per condition) | All reported results |

The three splits are disjoint. The train/test split is made at the level of source images, so all versions of a source image fall in the same partition. The file lists for every split are in `[data/splits/]`.

1. Download RRDataset following the instructions of the original authors.
2. Set the dataset root in `[configs/paths.yaml]`.
3. [Command to build B-RRDataset from the split lists.]

### SID-Set and TrueFake

- [SID-Set](https://github.com/[SIDA_REPO]): we use the official test split, Real and Fully Synthetic categories only.
- [TrueFake](https://github.com/[TRUEFAKE_REPO]): we use all social-media-shared images (Facebook, Telegram, X).

---

## Usage

### Inference

```bash
python [infer.py] --image path/to/image.jpg
```

[Describe output format and batch evaluation command.]

### Training

```bash
# 1. Image-Condition Classifier
python [train_classifier.py] --config [configs/classifier.yaml]

# 2. Condition-specific experts
python [train_expert.py] --config [configs/expert_original.yaml]
python [train_expert.py] --config [configs/expert_transfer.yaml]
python [train_expert.py] --config [configs/expert_redigital.yaml]
```

### Evaluation

```bash
python [evaluate.py] --dataset brr      # B-RRDataset test split
python [evaluate.py] --dataset sidset   # SID-Set
python [evaluate.py] --dataset truefake # TrueFake
```

---

## Implementation details

### Image-Condition Classifier

- **Backbone:** `facebook/dinov3-vitl16-pretrain-lvd1689m`
- **Training protocol:** Linear Probing followed by Fine-Tuning (LP-FT). A linear head is first trained with the backbone frozen. Then the last four transformer blocks are fine-tuned together with the head.
- **Accuracy:** 90% on the B-RRDataset test split, 89% on SID-Set and 89% on TrueFake.
- **Training time:** about 1 hour.
- **Routing on SID-Set and TrueFake:** the re-digitization output is discarded, and each image is routed to whichever of the two remaining experts has the higher probability.

### VLM experts

Each expert is fine-tuned only on training images of its condition, using the same structured prompt adopted for the zero-shot evaluation (see below).

| Setting | Value |
|---|---|
| Adaptation | LoRA |
| LoRA rank / alpha / dropout | r = 16, α = 32, dropout 0.05 |
| LoRA target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Epochs | 3 |
| Learning rate | 2 × 10⁻⁴ |
| Schedule | linear warmup over the first 10% of steps, then cosine decay |
| Batch size | 2 per device, 8 gradient-accumulation steps (effective 16) |
| Optimizer | 8-bit AdamW, weight decay 0.01 |
| Gradient clipping | max norm [X] |
| Memory | gradient checkpointing |
| Precision | BF16 |
| Kernels | FlashAttention 2; Liger Kernel (RoPE, fused linear cross-entropy, RMSNorm, SwiGLU) |
| Random seed | 42 |
| Training time | about 2 hours per expert |

### Inference settings

| Setting | Value |
|---|---|
| Precision | BF16 |
| Attention | FlashAttention 2 |
| Batch size | 1 |
| Max generated tokens | 5 |

Only one expert runs per image, so the per-image cost is one DINOv3 forward pass plus one VLM generation.

### Hardware and software

- **GPU:** a single NVIDIA RTX 5000 with 32 GB of memory.
- **Software:** Python [X], PyTorch [X], CUDA [X], Transformers [X], PEFT [X], Liger Kernel [X], FlashAttention [X].

### Prompt

The same prompt is used for zero-shot evaluation, fine-tuning and inference. The paper (Fig. 2) shows a condensed version; this is the full text.

```text
You are an expert in visual forensics specialized in detecting AI-generated and manipulated images under real-world conditions (including compression and re-digitization artifacts).
Carefully analyze the image using the following criteria:

1. Texture and Fine Details
  - Look for unnatural smoothness, over-sharpening, or inconsistent noise.
  - Check for missing or distorted fine structures (edges, text, small objects).

2. Global Lighting and Physical Consistency
  - Verify consistent lighting direction across the entire scene.
  - Check shadows, reflections, and illumination coherence between objects.

3. Structural and Semantic Coherence
  - Analyze whether objects, people, and scene elements are physically and logically plausible.
  - Look for distorted shapes, impossible structures, or inconsistent geometry.

4. Background and Scene Integrity
  - Detect repeating patterns, warped regions, unnatural transitions, or blending artifacts.
  - Check text, signs, and detailed areas for corruption or inconsistency.

5. Robustness to Real-World Degradation
  - Consider whether artifacts could come from compression or re-capturing.
  - Distinguish between real degradation and synthetic generation artifacts.

6. Overall Realism
  - Evaluate whether the image looks like a natural photograph captured in the real world.

After analyzing the image internally, provide your judgment using only one word:
- Output REAL if the image is a natural photograph.
- Output FAKE if the image appears AI-generated.
```

The unstructured prompt used in the zero-shot benchmark is:

```text
[Full unstructured prompt]
```

### Zero-shot VLM benchmark

| Model | Checkpoint |
|---|---|
| Qwen2.5-VL-7B | `Qwen/Qwen2.5-VL-7B-Instruct` |
| InternVL2.5-8B | `OpenGVLab/InternVL2_5-8B` |
| Pixtral-12B | `mistralai/Pixtral-12B-2409` |
| LLaVA-OneVision-1.5-8B | `lmms-lab/LLaVA-OneVision-1.5-8B-Instruct` |
| PaliGemma2-10B | `google/paligemma2-10b-mix-224` |
| DeepSeek-VL-7B | `deepseek-ai/deepseek-vl-7b-base` |

---

## Checkpoints

| Component | Base model | Weights |
|---|---|---|
| Image-Condition Classifier | DINOv3 ViT-L/16 | [LINK] |
| Original expert (LoRA) | Qwen2.5-VL-7B-Instruct | [LINK] |
| Social-transfer expert (LoRA) | Pixtral-12B-2409 | [LINK] |
| Re-digitization expert (LoRA) | Pixtral-12B-2409 | [LINK] |

The experts are released as LoRA adapters. Base models must be downloaded from Hugging Face under their own licenses.

---

## Citation

```bibtex

```

## Contact

For questions, open an issue or write to Simone Teglia (teglia@diag.uniroma1.it).
