# HiCoD
We propose Hierarchical Consistency-Guided Prompt Distillation (HiCoD), a framework that learns a robust, class-anchored semantic space. 

## Project Overview

Incomplete multimodal learning deals with situations where one or more input modalities are missing during training or inference. Existing methods often use re-alignment or re-construction to handle missing data but struggle to build a stable shared semantic space due to the key challenge anchor drift—a shift in class representations (prototypes) between complete and incomplete inputs—caused by distribution gaps and modality-specific variation. This drift can distort the representation space and hurt generalization. We propose Hierarchical Consistency-Guided Prompt Distillation (HiCoD), a framework that learns a robust, class-anchored semantic space. It combines: (i) a modality-aware semantic graph to model cross-modal relations; (ii) dual-level anchoring with LLM-based global prototypes and local exemplars; and (iii) multi-level distillation to align all features in the shared space. HiCoD achieves state-of-the-art results on CMU-MOSI, CMU-MOSEI, and other benchmarks, improving Acc-2 by up to 6.4 points over MPLMM under both fixed- and random-missing settings.

## Dataset Support

- MOSI (CMU Multimodal Opinion Sentiment and Intensity)
- MOSEI (CMU Multimodal Opinion Sentiment, Emotions and Attributes)

## Experiment Scenarios

The framework supports two distinct missing modality scenarios:

### 1. Fixed Missing Modality Scenario
Systematically discard specific modalities throughout evaluation:
- **Single modality missing**: {L}, {A}, {V}
- **Two modalities missing**: {L, A}, {L, V}, {A, V}

### 2. Random Missing Modality Scenario
Randomly select missing modalities with defined missing rate:
- **Missing rate range**: [0%, 10%, 20%, 30%, 40%, 50%, 60%, 70%]
- **Formula**: r_miss = (1 - Σ(m_i)/(N×M)) × 100%
  - N: total number of samples
  - m_i: number of available modalities for i-th sample
  - M: total number of modalities (3 for text, audio, vision)

## Project Structure

```
HiCoD/
├── trains/singleTask/model/
│   ├── components/           # Core component modules
│   │   ├── structure_aware.py      # Structure-aware representation learning
│   │   ├── modality_encoder.py     # Modality encoder (with pre-trained models)
│   │   ├── fusion_module.py        # Multimodal fusion module
│   │   ├── semantic_graph.py       # Cross-modal semantic graph construction
│   │   ├── gcn_enhancement.py      # GCN structure enhancement
│   │   ├── llm_encoder.py          # LLM encoder
│   │   ├── llm_prototype.py        # LLM prototype generation
│   │   ├── semantic_anchor.py      # Semantic anchoring
│   │   ├── dual_level_anchoring.py # Dual-level anchoring module
│   │   └── prompt_distillation.py  # Multi-level prompt distillation
│   ├── main_model.py        # Main model (integrating all components)
│   └── trainer.py           # Trainer
├── config/                  # Configuration files
├── data_loader.py           # Data loader
├── run.py                   # Main running script
└── train.py                 # Training script
```

## Core Components

### 1. Structure-aware Representation Learning
- Modality Encoding and Feature Projection: Support for BERT, ViT, Wav2Vec2 pre-trained encoders
- Adaptive Multimodal Fusion: Dynamic weight allocation based on available modalities
- Cross-modal Semantic Graph Construction: Edges between samples and sample-class centers
- GCN-enhanced Representation: Structure-aware feature enhancement via Graph Convolutional Networks

### 2. LLM-guided Dual-Level Semantic Anchoring
- LLM Category Prototype Construction: Learnable prompt tokens + [MOD] + class descriptions
- Local and Global Semantic Anchoring: Top-K high-confidence sample mean computation
- Similarity-based Anchor Selection: Entropy-regularized similarity computation

### 3. Hierarchical Consistency-Guided Prompt Distillation
- Local-Level Distillation: Align unimodal features with local anchors
- Fusion-Level Distillation: Align fused features with global anchors
- Prompt-Based Compensation Distillation: LLM-generated missing modality compensation
- Missing-Aware Classification Strategy: Adaptive compensation weight for final classification

## Implementation Status

### Completed Features
- Complete implementation of all three core components
- Pre-trained encoder support (BERT, ViT, Wav2Vec2)
- Learnable LLM prompt design
- Cross-modal edge aggregation algorithm
- Missing modality awareness and compensation mechanism
- Complete loss function implementation (classification loss + three distillation losses)
- Main model integrating all components
- Trainer and data loader
- Configuration file system
- Test scripts and validation

### Core Features
- Paper Formula Correspondence: All implementations strictly follow the mathematical formulas in the paper
- Missing Modality Robustness: Complete missing modality handling mechanism
- LLM Knowledge Integration: Frozen LLM + learnable prompt design
- Multi-level Semantic Alignment: Semantic consistency at local, global, and language levels

## Usage

### Environment Setup
```bash
pip install torch transformers numpy
```

### Validate Components
```bash
# Validate individual components
python validate_structure_aware.py      # Validate structure-aware component
python validate_dual_level_anchoring.py # Validate dual-level anchoring component
python validate_prompt_distillation.py  # Validate multi-level distillation component

# Validate complete pipeline
python validate_complete_pipeline.py    # Validate entire training pipeline
```

## Loss Functions

The framework includes 4 loss functions:
1. Classification Loss (CrossEntropyLoss) - Main task loss
2. Local Distillation Loss - Align unimodal features with local anchors
3. Fusion Distillation Loss - Align fused features with global anchors
4. Prompt Distillation Loss - Align LLM-generated compensation with fused features

## Configuration Parameters

### Model Architecture
- Text features: BERT-base (768d)
- Visual features: Facet (35d) 
- Acoustic features: COVAREP (74d)
- Shared embedding dimension: 512d
- LLM encoder: Frozen T5-Base

### Training Parameters
- Optimizer: AdamW
- Learning rate: 5×10^-5 (BERT), 1×10^-3 (other components)
- Batch size: 32
- Training epochs: 20
- Validation metric: F1 score

### Key Hyperparameters
- Top-K selection: k = 15
- Graph threshold: δ = 1.5
- Temperature: τ = 0.7
- Graph gate: β = 0.5
- Loss weights: (λ_local, λ_global, λ_prompt) = (1.0, 1.0, 0.5)
- Completion weight: γ = 0.3

## Project Status

Project Completed - All core features have been implemented and validated, ready for multimodal emotion recognition tasks.
