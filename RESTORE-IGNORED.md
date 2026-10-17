# RESTORE-IGNORED — Strata

Tài liệu khôi phục các file **không nằm trong Git** của project. Cập nhật 2026-10-17,
trước khi chuyển `C:\Projects\strata` sang `F:\Projects\strata`.

## 1. Dữ liệu ignored

| Đường dẫn | Dung lượng | Số file | Loại |
|---|---:|---:|---|
| `.rocm-win/` | 7.58 GB | 11.505 | ROCm SDK do setup.py cài (TheRock wheels) |
| `engine-*/` | ~6.9 GB | ~9.000 | engine build theo từng nhánh/thử nghiệm |
| `dist/` | 0.54 GB | 316 | release package |
| `build-hip-win*/` | 0.23 GB | 428 | CMake build tree |
| `third_party/` | 0.16 GB | 2.834 | llama.cpp source pinned |
| `serve/*.bak`, `*.log`, `*.dmp`, sweep json | < 20 MB | — | log/dump/kết quả thử |

Tổng ignored: **~14.58 GB / 24.855 file**. Không có file ignored nào là duy nhất không thể tái tạo:
tất cả đến từ setup.py, build hoặc log.

## 2. Khôi phục chuẩn

```powershell
python setup.py --setup      # hoặc START-HERE.bat --setup
python setup.py --check
```

`setup.py` (đã commit) tự làm các bước:

1. Tạo `.venv` và cài `numpy, jinja2, regex, pyyaml, tqdm, requests, cmake, ninja, pillow, psutil`.
2. Cài ROCm TheRock wheels vào `.venv` (`rocm[libraries,devel]==<version>`, ~10 GB);
   `rocm-sdk path --root` trỏ tới `.rocm-win`. Version pin theo family GPU trong `setup.py`
   (`ROCM_FAMILY_PINS`), có thể ép bằng `STRATA_ROCM_VERSION`.
3. Tải engine HIP dựng sẵn cho Windows, hoặc build từ source nếu không có bản phù hợp.
4. Tải model từ Hugging Face tại revision pin trong `HF_REVISIONS`:

| Repo | Revision |
|---|---|
| `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` | `ed59f92082b1e93c0e96d60a8b11aab089b52f09` |
| `ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF` | `b22d729eae29b5796f76fb70f91aef549b9fc52c` |
| `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF` | `5348543e0147355ac9cbcb031184a3546350988e` |
| `unsloth/Qwen3.8-Flash-Next-GGUF` | `38bb39ee97821de2c9009abb7e93950eec396e66` |

5. Tải source llama.cpp:

```
https://github.com/ggml-org/llama.cpp/archive/3cf03257f219afbe7334045ff7c6a06ac68c627d.zip
```

giải nén vào `third_party/llama.cpp` (hằng số `LLAMA_CPP_COMMIT` trong `setup.py`).

6. Fetch MTP draft layer (~5 GB) từ checkpoint gốc và chuẩn bị pack.

Model đầy đủ ~70 GB; có thể dùng `--gguf-dir` nếu đã có file, hoặc
`--source modelscope` / `HF_ENDPOINT=https://hf-mirror.com` khi Hugging Face không truy cập được.
Chi tiết: `docs/INSTALL.md`, `docs/AMD_HIP.md`, `docs/MODELS.md`.

## 3. Build engine từ source

```powershell
python setup.py --build
```

hoặc theo `docs/AMD_HIP.md`:

```powershell
tools\hip\build_windows.bat
```

Yêu cầu: Visual Studio 2022 Build Tools (C++ workload + Windows SDK), CMake, Ninja,
ROCm SDK (`.rocm-win`). Kết quả nằm trong `build-hip-win/` và `engine/`.

Các engine thử nghiệm (`engine-0142`, `engine-0142r`, `engine-opt2`, `engine-async`,
`engine-helper`, `engine-foresight`, `engine-gfx1032`, `engine.bak-0138`...) chỉ là bản build
của các nhánh khác nhau; **không cần khôi phục** trừ khi đang chạy đúng variant đó.
Engine production hiện tại do `Start-LLM.ps1` chọn; xem `Config/strata/*.json`.

Riêng engine opt2 có patch native save/reclaim, rebuild theo hướng dẫn trong
`..\kurai-llm\Docs\portable-install.md`:

```powershell
python ..\kurai-llm\Scripts\install-native-save-reclaim.py --strata-root F:\Build\strata-opt2
..\kurai-llm\Scripts\build-strata-save-reclaim.cmd F:\Build\strata-opt2 F:\Build\strata-opt2-build F:\Projects\strata
```

Revision opt2: `d3eb59835750035c1fddfbcf318d6f0fc2cd46b6`.

## 4. Release package — `dist/`

`dist/` là output đóng gói release, không phải nguồn. Dựng lại bằng
`tools/hip/build_windows.bat` (hoặc `python setup.py --build`) rồi đóng gói theo
`docs/INSTALL.md`. Không cần copy `dist/` sang máy mới.

## 5. Benchmark fixtures

`bench/tiny-*.gguf`, `bench/wide-*.gguf`, `bench/micro/*` bị ignore và tái tạo bằng:

```powershell
python tools\make_tiny_model.py
```

## 6. Không cần khôi phục

- `*.log`, `*.dmp`, `*.nsys-rep`, `strata-stall-*`, `strata-*.json` (kết quả sweep).
- `serve/*.bak-*` (bản backup source cũ).
- `.pytest_cache/`, `__pycache__/`.
- `.venv/` — tạo lại bằng `python -m venv .venv` + `pip install -r requirements.txt`.

## 7. Kiểm tra

```powershell
python setup.py --check
python setup.py --inspect <đường-dẫn-gguf>    # kiểm tra GGUF trước khi chạy
```

Sau khi chuyển máy, cập nhật đường dẫn tuyệt đối trong `Config/strata/*.json` và
`run-*.bat` nếu có, rồi chạy `Start-LLM` theo `AGENTS.md`.
