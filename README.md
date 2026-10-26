# VN ALPR lab

Bộ công cụ chọn model phát hiện và đọc biển số Việt Nam. Có hai nơi chạy:

- Colab hoặc máy NVIDIA x86 là TRAIN_HOST: tải dữ liệu, fine-tune, đo accuracy, xuất ONNX, zip kết quả.
- Box Jetson ARM64 là EDGE_TARGET: build TensorRT và đo latency, bộ nhớ, công suất. Engine không được build trên GPU server rồi mang sang.

## Mở trên Colab

[Open in Colab](https://colab.research.google.com/github/nqMinh-GR/NCS/blob/main/notebooks/VN_ALPR_Colab.ipynb)

Chọn GPU runtime rồi bấm **Runtime → Run all**. Notebook tự clone `https://github.com/nqMinh-GR/NCS.git`, cài stack, tải dữ liệu, train, zip và gọi `files.download`.

Ô tham số nằm gần đầu notebook. Sửa trước khi Run all nếu cần:

- `DET_BATCH_SIZE`, `OCR_BATCH_SIZE`
- `GRAD_ACCUM_STEPS`, `NUM_WORKERS`
- `DET_EPOCHS`, `OCR_EPOCHS`
- `LEARNING_RATE` (`None` giữ learning rate upstream)
- `MAX_IMAGES` (`None` dùng hết ảnh)
- `DET_INPUT_SIZE` và `FORCE_DET_INPUT` (chỉ RF-DETR; RT-DETRv4 và D-FINE vẫn resize 640)

`ROBOFLOW_API_KEY` để trong secret của Colab. Notebook không in key. Không có key thì bộ Hugging Face public vẫn được tải. Bộ đó là dữ liệu nhiều nước; nếu không có bộ biển Việt Nam đi kèm, manifest ghi `mixed_auxiliary_not_vn_leaderboard`.

Latency ONNX đo trên GPU Colab (hoặc CPU nếu không có CUDA) ghi vào `runs/vn_alpr/host_latency.csv` với p50, p95, p99. Đó không phải latency Jetson. Bảng Jetson trong cùng notebook vẫn là `pending_on_device`. Colab không build engine TensorRT.

`LATENCY_WARMUP`, `LATENCY_ITERATIONS`, `LATENCY_SECONDS` và `LATENCY_REPEATS` để `None` thì dùng mức trong `configs/project.yaml`.

## Lệnh local

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -r environments/requirements-core.txt pytest nbformat
.venv/bin/pytest
.venv/bin/python scripts/build_notebooks.py
.venv/bin/python scripts/hardware_probe.py
.venv/bin/python scripts/prepare_datasets.py
.venv/bin/python scripts/train.py --profile=smoke
.venv/bin/python scripts/package_results.py
```

`smoke` tạo ảnh synthetic để kiểm tra convert, dedup và evaluator. Số đó nằm trong `self_test` và không phải điểm biển số thật.

## Model mặc định

Detector screen: `rtdetrv4_s`, `rfdetr_small`, `dfine_n`, `dfine_s`.
OCR screen đủ điều kiện thương mại trong config hiện tại: `ppocrv6_tiny`, `cct_s_v2`. `svtrv2_t` có adapter nhưng weights chưa được cấp quyền rõ, nên `commercial_only` bỏ qua.

DEIMv2 và RF-DETR XL/2XL không được train. Checkpoint Objects365 của D-FINE không được tải. RF-DETR không đi qua Ultralytics. Nano của RF-DETR trong bảng upstream là khoảng 30,5 triệu tham số, không phải model nhỏ hơn D-FINE-N.

## Trạng thái chạy

Máy dùng để viết lab này không có NVIDIA GPU và không phải Jetson. `train_executed` và `edge_executed` là false cho đến khi Colab và box chạy thật. Không điền latency 0 cho Jetson.

Chi tiết license nằm trong `THIRD_PARTY_NOTICES.md`. Registry nằm trong `configs/models.yaml` và `configs/datasets.yaml`, đối chiếu ngày 2026-10-26.
