# Real Street-Scene Inference Validation

## Adaptive Edge AI Perception Engine for Autonomous Mobile Robots

**Validation type:** Single-image real-backend functional test  
**Execution backend:** CUDA-enabled real inference  
**Input:** Urban street scene  
**Output:** Multi-model perception fusion result  

---

## 1. Purpose

This experiment validates that the Adaptive Edge AI Perception Engine can execute its real perception backends on a complex urban street image.

The objective of this test is functional validation of the real inference path rather than formal object-detection accuracy or GPU-performance benchmarking.

The tested perception chain includes:

- object detection,
- multi-model object detection,
- semantic segmentation,
- monocular relative depth,
- human pose estimation,
- multi-model perception fusion,
- collision-risk estimation,
- decision-confidence estimation.

---

## 2. Test Configuration

The test was executed using:

```bash
python3 -m benchmark.understanding_demo \
  --image assets/street_test.png \
  --backend real \
  --device cuda \
  --out assets/street_test_real_cuda.json
