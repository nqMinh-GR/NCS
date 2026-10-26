# Third-party notices

Checked 2026-10-26. A code license is not by itself a weights license. Dataset pages were not treated as cleared image rights beyond the statements cited here.

## Detection code and weights

- RT-DETRv4, commit `55fefaaed7efe2a5f72d0a18fd4e05965e35c292`: Apache-2.0. Student COCO checkpoints are the Google Drive files linked from that commit's README (S `1jDAVxblqRPEWed7Hxm6GwcEl7zn72U6z`, M `1O-YpP4X-quuOXbi96y2TKkztbjroP5mX`). The published training config uses a DINOv3 ViT-B/16 teacher. This lab's fine-tune removes `teacher_model` and sets `loss_distill` to 0 so the teacher file is not downloaded. DINOv3 itself is under the Meta DINO License, which allows commercial use and requires the license to follow redistributed derivatives: https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md
- RF-DETR 1.11.2, commit `7433d5919d0f7e72cc6cb78663e0a731afd499e0`: Apache-2.0 for the package and for Nano, Small, Medium, and Large detection weights. Evidence: https://docs.roboflow.com/models/model-licenses and `src/rfdetr/assets/model_weights.py` at that tag. XL and 2XL are PML 1.0 and are not used. Weights URLs and MD5 values are the ones in that file.
- D-FINE, commit `956d1709314c2c6a4df6f34de232054578a7449f`: Apache-2.0. Default checkpoints are the COCO release assets `dfine_n_coco.pth`, `dfine_s_coco.pth`, and `dfine_m_coco.pth`. The README says `*_obj365` and `*_obj2coco` are not commercially cleared. Those files are not downloaded.
- DEIM, commit `09d35d53d39ee3145a1e61e3a989b28b9468d1dd`: Apache-2.0. Optional recipe only. No checkpoint URL was pinned.
- DEIMv2, commit `f005f50ce565e63f6c298feccb51901a98c3d035`: custom license, non-commercial only. Not trained or exported. https://github.com/Intellindust-AI-Lab/DEIMv2/blob/f005f50ce565e63f6c298feccb51901a98c3d035/LICENSE.md
- RT-DETRv2 in `lyuwenyu/RT-DETR` commit `df2a96c8c3a098567a1149bf0248da6efe6d495d`: Apache-2.0 code. Optional and off. Checkpoint URL not pinned.

## OCR

- PaddleOCR tag `v3.7.0`, commit `b03f46425e8ff4442b268ce449e3eef758146cd4`, package `paddleocr==3.7.0`: Apache-2.0. `PP-OCRv6_tiny_rec` Hugging Face revision `0736086f72f666350ebcdc0c3a504eeac89cdfad` and `PP-OCRv6_small_rec` revision `bd619643acac4b9650c040234da8d944476ee3f1` declare `apache-2.0`. Inference entry is `paddleocr.TextRecognition`. Training YAML uses CTC plus NRTR; the inference postprocess in the tiny config is `CTCLabelDecode`.
- fast-plate-ocr tag `v1.1.0`, commit `9ce7a5b64a939aa421c243b331d42e6bc25ffd44`: MIT code. The author said on 2026-04-15 that code and models are MIT: https://github.com/blakeblackshear/frigate/issues/22891. CCT v2 weights and configs are the `arg-plates` release assets `cct_s_v2_global.*` and `cct_xs_v2_global.*`. The head is per-slot classification, not CTC. The plate config stretches to 128x64 (`keep_aspect_ratio: false`); that protocol is recorded and is not treated as identical to a pad-preserving model.
- OpenOCR commit `1ccfc6ee6161f7133b192e16af3a9265273997ba`: Apache-2.0 code. SVTRv2 T/S weights are Drive folders in `configs/rec/svtrv2/readme.md`, without a file id or a weights license. Status stays unresolved. Decoder is CTC.

## Data

- Roboflow projects listed in `configs/datasets.yaml` with CC BY 4.0 stay eligible for download after the API returns a real version. `rf_vn_duke` had no export version at review, so none is invented.
- winter2897 commit `8741a28a1f4d4ebefa00b4908b374f2c849cb8e7`: dataset file ids were read from `doc/dataset.md`. The repo has no SPDX license and the text thanks Mì Ai without a commercial grant. Status unresolved.
- fict-labs/VNLP commit `d0b38df15c961393a7a1189912476c3d8d1edbfc`: citation requested, no commercial grant. The Drive link is a folder id.
- `keremberke/license-plate-object-detection` revision `a51194c739991abb50ac8afe14704aa99a66cf51`: README says CC BY 4.0. Load the zip files. Do not enable `trust_remote_code`.
- `sonnetechnology/license-plate-text-recognition-full` revision `e9eeda1872046de715d36de41d326e7a4e7d4634`: card says CC BY 4.0 and is quarantined because targets are not clean plate text.
- CCPD commit `02aaea15137c4d2fe662e57d257c6822356e9304`: README says the dataset is MIT. Auxiliary, Chinese charset, capped, not in the Vietnam score.
- UFPR-ALPR, RodoSol-ALPR, LRLPR-26, ELP, LPLC, and paid UniData-style sets are excluded and are not downloaded.

## NVIDIA

TensorRT, CUDA, cuDNN, and JetPack are used under NVIDIA's own license terms. They are not public domain. This lab does not vendor those SDKs. On Jetson, use the build already installed with the board image.

## What this repository adds

The `vn_alpr` package, configs, tests, and notebooks were written for this benchmark. They do not relicense the upstream models or datasets.
