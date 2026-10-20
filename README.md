# Real-Time Object Detection with NVIDIA TensorRT

Accelerating YOLO11 object detection on an NVIDIA T4 GPU with TensorRT FP16 and INT8, verified with GPU-synchronized benchmarks, mAP accuracy checks, and a side-by-side video demo.

<!-- Replace with your GIF after running the notebook: -->
![Side-by-side demo: PyTorch (left) vs. TensorRT FP16 (right)](assets/demo.gif)

<!-- Replace with your chart: -->
![FPS by version](assets/fps_chart.png)

## What I did

- Built a baseline of the pretrained **YOLO11n** detector in **PyTorch (FP32)** on a real video at 640x640.
- Compiled the same model into **TensorRT engines** at **FP16** and **INT8** (INT8 calibrated on coco128).
- Benchmarked every version the same way: identical inputs, 10 warm-up frames skipped, **GPU-synchronized timing**.
- Checked accuracy (**mAP50-95 / mAP50** on coco128) to measure what the speedup costs.
- Produced a **side-by-side demo video** with live FPS counters.

## Results

NVIDIA T4 (Google Colab), YOLO11n, 640x640, batch size 1.

| Version | FPS | Speedup vs. PyTorch | mAP50-95 | mAP50 |
|---|---|---|---|---|
| PyTorch FP32 (baseline) | [FPS] | 1.00x | [mAP] | [mAP] |
| TensorRT FP16 | [FPS] | [X.XX]x | [mAP] | [mAP] |
| TensorRT INT8 | [FPS] | [X.XX]x | [mAP] | [mAP] |

FPS is end-to-end per frame (pre-processing, inference and post-processing; video decoding not included), averaged over the whole video after warm-up. mAP is measured on coco128. Those images come from the COCO training set, so the absolute scores run high; compare the versions with each other, not with published COCO numbers.

## What I learned

**What TensorRT does.** TensorRT is NVIDIA's inference compiler. It takes a trained network and builds an "engine" tuned for one specific GPU. It fuses layers (for example convolution + activation) so data makes fewer trips to memory, removes work that is only needed during training, and benchmarks several kernel implementations per layer, keeping the fastest one for that hardware.

**Why FP16 and INT8 are faster.** Lower-precision numbers are smaller, so more of them fit in cache and less memory bandwidth is needed to move them. They also run on **Tensor Cores**, hardware built for fast low-precision matrix math. On a T4, FP16 and INT8 Tensor Core throughput is far higher than FP32.

**Why accuracy can drop slightly.** Fewer bits means coarser numbers. FP16 is precise enough for inference, so accuracy is usually unchanged. INT8 has only 256 levels per value, so TensorRT uses a calibration dataset to choose a scale for each layer. Values that fall outside the calibrated range, or small differences that get rounded away, can shift box positions and confidence scores a little, which shows up as a small mAP drop.

**Why the end-to-end speedup is smaller than the model speedup.** TensorRT accelerates the neural network itself. Pre-processing and post-processing (NMS) still run in Python/PyTorch, and with a model as small as YOLO11n they take up a large share of each frame, so they set a ceiling on the overall FPS gain.

## How to run it

1. Open `yolo_tensorrt_benchmark.ipynb` in [Google Colab](https://colab.research.google.com/) (File > Upload notebook).
2. Select a GPU: **Runtime > Change runtime type > T4 GPU**.
3. Upload a video named **`video.mp4`** using the folder panel on the left.
4. **Runtime > Run all.** The run takes roughly 10-20 minutes, and `results.zip` downloads at the end.

## Environment

<!-- Copy these values from environment.txt (printed in the notebook's Setup section). -->

| Component | Version |
|---|---|
| GPU | [GPU name] |
| NVIDIA driver | [driver] |
| Python | [version] |
| PyTorch | [version] |
| CUDA (PyTorch build) | [version] |
| TensorRT | [version] |
| Ultralytics | [version] |
