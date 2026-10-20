| Version | FPS | Speedup vs. PyTorch | Model inference (ms) | mAP50-95 | mAP50 |
|---|---|---|---|---|---|
| PyTorch FP32 | 67.5 | 1.00x | 11.57 | 0.508 | 0.675 |
| TensorRT FP16 | 154.4 | 2.29x | 3.53 | 0.505 | 0.675 |
| TensorRT INT8 | 150.2 | 2.23x | 3.76 | 0.483 | 0.657 |

GPU: Tesla T4. Model: yolo11n. Input: 640x640, batch size 1. 
FPS is end-to-end per frame (pre-processing + inference + post-processing, excluding video decode), averaged over the video after 10 warm-up frames. mAP measured on coco128.
