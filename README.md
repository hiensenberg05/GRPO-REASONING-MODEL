# Gemma 3 1B Reasoning: Critic-Free GRPO on a Single TPU

A reinforcement learning pipeline that teaches Google's open-weight Gemma 3 (1B) model to reason, deliberate, and format its logic using Group Relative Policy Optimization (GRPO). The entire orchestration is engineered to run on a single Kaggle TPU v5e-8 without Out-Of-Memory (OOM) errors.

## 🚀 Overview

Out of the box, a 1B model acts as an eager text generator. This project forces the model to pause, formulate a step-by-step logic trace inside a `<reasoning>` block, and output a final `<answer>`. 

Instead of an expensive Actor-Critic architecture, this pipeline implements a critic-free GRPO loop using programmatic Python reward functions to grade the model's logic on the fly—eliminating the need for human-annotated preference data.

## 🛠 Tech Stack & Hardware Architecture

To survive the strict 16GB HBM limit per TPU core, the pipeline is built natively for Google's TPU mesh:

* **JAX (XLA):** Compiles GRPO mathematical operations directly to the TPU.
* **Flax (v0.12.0 `nnx`):** Object-oriented model state management and lightweight LoRA checkpointing.
* **Qwix:** Rigid memory distribution mapping the model across a `1x4` logical grid using Fully Sharded Data Parallelism (FSDP) and Tensor Parallelism (TP).
* **Tunix:** Google's JAX-native RLHF library, orchestrating the rollout and update loops.

**Memory Optimization:** The base Gemma 3 weights are frozen (Reference Model). A Rank-64 LoRA adapter is injected into the attention and MLP layers (Actor Model), reducing trainable parameters to **~0.3%**.

## 🧠 The 3-Phase Curriculum

To prevent training collapse, the model is trained over a continuous 1000-step loop. The AdamW optimizer maintains gradient momentum while the reward functions are dynamically swapped to escalate difficulty:

1. **Phase 0: Formatting (Steps 0–250)**
   * **Goal:** Teach structural compliance.
   * **Reward:** Strict Regex checkers enforce the `<reasoning>` and `<answer>` wrapper formatting.
2. **Phase 1: Correctness (Steps 250–750)**
   * **Goal:** Domain-aware accuracy across a 50/30/20 data mixture (Math, Science, Logic).
   * **Reward:** Exact value extraction and partial credit tolerances for math logic.
3. **Phase 2: Brevity (Steps 750–1000)**
   * **Goal:** Anti-rambling optimization.
   * **Reward:** Heavy penalties for exceeding token limits or generating text after the final answer tag.

## 📊 Evaluation & Results

The model was evaluated deterministically (greedy decoding, temperature = 0) against a holdout test dataset.

| Metric | Base Model (Pre-Training) | Final LoRA Model |
| :--- | :--- | :--- |
| **Format Accuracy** | 33% | **97%** |
| **Partial Accuracy** | 18% | **28%** |
| **Strict Accuracy** | 15% | **25%** |

*Takeaway: The jump to 97% structural compliance confirms the GRPO pipeline successfully rewired the model's fundamental behavior to think before speaking. The strict accuracy boost reflects the physical limitations of training only 0.3% of the parameters on a 1B model architecture.*

## 🔗 Notebook & Code

The full, executable training pipeline is available here:
**Kaggle Notebook:** [Advanced GRPO Reasoning Pipeline](https://www.kaggle.com/code/uttkarsh05/advancednotebook)
