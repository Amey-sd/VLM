# Architecture: VLM Temporal Operation Intelligence

## 1. Model Selection Defense

The **Qwen2-VL-2B-Instruct** model was selected for this challenge due to its exceptional balance between performance and compute efficiency, specifically tailored for resource-constrained environments like Kaggle T4 GPUs (16GB VRAM).

| Model Component | Memory Cost | Explanation |
| :--- | :--- | :--- |
| **Base Model Weights** | **~1.6 GB** | **4-bit Quantization (NF4)** reduces the 2.2B parameters from ~4.4GB (FP16) to ~1.2GB + metadata overhead. |
| **LoRA Adapters** | **~0.1 GB** | We train only <1% of parameters (Rank 16), keeping the gradient footprint minimal. |
| **Activations** | **~2.5 GB** | This is the cost of "context." By limiting resolution to **224x224** and clip length to **4 frames**, we keep the token count (~1024) manageable. |
| **PyTorch Overhead** | **~1.0 GB** | CUDA kernels and workspace buffers. |
| **Total Usage** | **~5.2 GB** | **Result:** Fits comfortably on 16GB T4 or 12GB RTX 3060. |

Compared to **LLaVA-NeXT-Video (7B)** or **VideoLLaMA2 (7B)**, Qwen2-VL-2B allows for higher batch sizes and longer sequence lengths within the same VRAM budget. A 7B model would require offloading to CPU or extreme quantization that degrades temporal reasoning capabilities.

## 2. Data Engineering: The Skeleton Proxy Strategy

Due to licensing restrictions and bandwidth constraints preventing access to raw RGB footage, this project implements a **visual proxy strategy**.
*   **Input:** OpenPack 2D Keypoint streams (JSON).
*   **Transformation:** We render high-contrast "Skeleton Videos" (Green stick figures on black backgrounds).
*   **Constraint Handling:** This approach reduces the dataset size by >99% (from 50GB to <500MB) while preserving the essential temporal dynamics of human motion required for action recognition.

## 3. Frame Sampling Rationale: Motion-Magnitude Adaptive Sampling

Uniform sampling was rejected as it often captures redundant "static" frames while missing critical high-velocity movements (e.g., the swift motion of applying tape). Instead, we implemented **Motion-Magnitude Adaptive Sampling**.

**Strategy:**
1. Calculate the L2 distance (Euclidean motion delta) between skeleton keypoints across all consecutive frames in a 5-second window.
2. Divide the window into **4 temporal segments** (chunks).
3. In each segment, select the frame that exhibits the **highest motion magnitude**.

**Sampling Visualization:**
```text
Time (s):  0.0 --- 1.25 --- 2.50 --- 3.75 --- 5.0
Ops:       [--- Action A ---]|[--- Action B ---]
Boundary:                    ^ (2.5s)
Sample Pattern:  *      *        *      *
(Asterisks represent frames selected due to peak velocity)
```
This ensures that the model sees the "moment of action" rather than a blurred or static state, maximizing the information density of the limited 4-frame input window.

## 4. Failure Mode Analysis

**Observed Confusion:** "Tape" vs. "Box Setup".

**Hypothesis:**
The model frequently confuses "Tape" with "Box Setup" because both operations involve the same physical area (the box flaps) and similar hand positions.
*   **Visual Ambiguity:** In the OpenPack dataset, the "Box Setup" phase often includes manual folding that visually mimics the "Tape" application phase when viewed via skeleton keypoints alone. The absence of RGB texture (e.g., the visibility of the tape dispenser object itself) makes this distinction purely motion-based.
*   **Temporal Ambiguity:** The transition between folding a flap and applying tape is often less than 200ms. Without higher-frequency sampling (e.g., 30fps), the model struggles to distinguish the exact boundary based on sparse 4-frame inputs.

**Mitigation Strategy:**
Future iterations should incorporate object detection features (e.g., bounding box coordinates of the tape gun) alongside the skeleton keypoints to resolve this ambiguity.