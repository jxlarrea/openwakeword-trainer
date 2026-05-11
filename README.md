# OpenWakeWord Trainer

A Dockerized, GPU-accelerated trainer for [openWakeWord](https://github.com/dscripka/openWakeWord) custom models with a Web UI. Generates synthetic positives with [Piper](https://github.com/OHF-Voice/piper1-gpl) (and optional ElevenLabs voices), pulls real-world augmentation corpora (MIT IR Survey, MUSAN, FSD50K, Common Voice), trains a small classifier head on top of Google's frozen speech-embedding model, and exports an ONNX model that drops into the openWakeWord runtime.

Built and tuned on **NVIDIA DGX Spark** (Grace + Blackwell GB10, aarch64) but the same image runs on any NVIDIA host with CUDA 12.8 drivers.

![by Xavier Larrea](https://img.shields.io/badge/by-Xavier%20Larrea-blue)

## Highlights

- **GPU end-to-end.** Feature extraction (Google speech-embedding ONNX) runs on the GPU via a locally-compiled `onnxruntime-gpu` wheel for aarch64+CUDA. PyTorch trains the classifier head on the same device.
- **Parallel TTS.** Piper synthesis runs across a process pool (10 workers x 2 ORT threads on DGX Spark by default) so generating ~100k clips takes ~20 min instead of ~4 hours.
- **Resumable pipeline.** Every long phase drops sentinels: WAV generation is cached per run name, corpora downloads are sentinel-marked, and re-running a failed run skips work that was already done.
- **Hard-negative iteration.** Paste false-triggers you saw in production into the "Negative phrases" field; they get synthesized with the same emphasis as positives so the next model strongly rejects them.
- **Live progress.** Web UI shows phase banner, multiple progress bars, validation metrics, and a streaming log fed by every module's `logger.info` calls.
- **Sane validation.** The form blocks Start when required fields are empty, when no voice is selected, or when no augmentation corpus is enabled. The same checks fire server-side as a defense.

## Architecture

```
+------------------+     +--------------------+     +-------------------+
| Web UI           | --> | Pipeline           | --> | openWakeWord      |
| (FastAPI + SSE)  |     | Orchestrator       |     | runtime           |
+------------------+     |                    |     | (HA / Wyoming /   |
                         | 1. Piper x N voices |    |  your app)        |
                         | 2. Hard negatives  |     +-------------------+
                         | 3. Adversarial     |
                         | 4. Download corpora|
                         | 5. Augment + feat. |
                         | 6. Train classifier|
                         | 7. Export ONNX     |
                         +--------------------+
```

The classifier ONNX accepts `(1, 16, 96)` float32 input - 16 consecutive 96-dim Google speech-embeddings, which is exactly what `openwakeword.Model(wakeword_models=["mymodel.onnx"])` expects.

## Quickstart

Prerequisites:

- Docker 24+ with Compose v2.
- NVIDIA GPU with `nvidia-container-toolkit` installed.
- Driver supporting CUDA 12.8 (R555+).
- ~100 GB of free disk for the cached corpora and built image.

```bash
git clone https://github.com/jxlarrea/openwakeword-trainer
cd openwakeword-trainer

cp .env.example .env
# Edit .env to add HF_TOKEN (needed for Common Voice).
# Optional: paste an ELEVENLABS_API_KEY.

docker compose up --build
```

> **First build is slow.** The Dockerfile compiles `onnxruntime-gpu` from source for sm_120 Blackwell kernels. Expect **25 to 60 minutes** on a fast CPU. The base layers, dependencies, and ORT compile are then cached for every subsequent build.

Open http://localhost:8000.

1. Type a wake word (e.g. `hey jarvis`). Run name auto-fills as `hey_jarvis`.
2. Paste positive phrases (5 to 10 spelling/pronunciation variants of the wake word).
3. Optionally paste negative phrases (false-triggers observed in production).
4. Pick voices. Click **Select all** or **High quality only**.
5. Confirm augmentation corpora. All four are recommended; ~45 GB total disk on first run.
6. Click **Start training**. Live progress streams into the panel below.
7. When done, click **Download .onnx** in section 3. The model also stays at `/opt/models/openwakeword-trainer/models/<run_name>.onnx` on the host (or inside the docker volume if you didn't set `OWW_DATA_DIR_HOST`).

## DGX Spark notes

DGX Spark uses a GB10 Grace + Blackwell SoC (aarch64 + sm_121). The image's two-stage Dockerfile is built specifically with this in mind:

- **Stage 1**: `nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04` + CMake 3.31 (via pip) compiles `onnxruntime-gpu` v1.20.1 from source with `CMAKE_CUDA_ARCHITECTURES=120`. The wheel is built with `--parallel 8 --nvcc_threads 1` to keep peak memory under ~50 GB during flash-attention kernel compilation (running with full parallelism OOM'd a 128 GB host).
- **Stage 2**: `nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04` is the runtime image. PyTorch 2.7.0 + cu128 wheels are installed for Blackwell support. The custom-built `onnxruntime-gpu` wheel is copied from stage 1 and installed last, so its files are the ones that survive when the CPU `onnxruntime` package (pulled in by `piper-tts` and `openwakeword`) is uninstalled.

Tested working: CUDA EP runs Google's speech-embedding ONNX directly on the Blackwell GPU, ~50x faster than CPU.

Verify GPU visibility from inside the container:

```bash
docker compose exec trainer python -c "import torch, onnxruntime as ort; print('torch:', torch.__version__, '| cuda?', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0)); print('ort:', ort.__version__, '|', ort.get_available_providers())"
```

Expected output:

```
torch: 2.7.0+cu128 | cuda? True | NVIDIA GB10
ort: 1.20.1 | ['CUDAExecutionProvider', 'CPUExecutionProvider']
```

## Storage layout

The container always uses `/data` internally. Pick where that lives on the host with `OWW_DATA_DIR_HOST` in `.env`:

| `.env` setting | Effect |
|---|---|
| `OWW_DATA_DIR_HOST=` (unset) | Use the portable named docker volume `oww_trainer_data` |
| `OWW_DATA_DIR_HOST=/opt/models/openwakeword-trainer` | Bind-mount this host directory at `/data` |

The directory must exist and be writable by the container user (root by default).

Inside `/data`:

```
/data
  piper_voices/        downloaded Piper voice .onnx + .json (lazy, on first use)
  augmentations/
    rirs/mit/          MIT IR Survey wavs (via HF mirror)
    musan/             musan/{noise,music,speech}/
    fsd50k/clips/      FSD50K individual wavs (via Fhrozen/FSD50k HF mirror)
    common_voice/en/   16 kHz wavs decoded from CV tar shards
  generated/           historical generated wavs (per run)
  runs/<run_id>/
    config.json        the exact TrainRunConfig that was submitted
    wavs/
      positive/        positive synthesized wavs + .generated_pos sentinel
      negative/        adversarial + hard-negative wavs + sentinels
    train_features.bin val_features.bin   memmap feature shards
    train_labels.npy   val_labels.npy
    best.pt            best classifier weights
    wakeword.onnx      exported model
    result.json        final metrics + history
  models/              <run_name>.onnx copies, served by the UI download button
  cache/               HF / torch caches (HF_HOME, TORCH_HOME)
```

### Disk budget

| Corpus | Disk on /data |
|---|---|
| MIT IR Survey (RIRs) | ~10 MB |
| MUSAN (noise + music + speech) | ~26 GB |
| FSD50K dev + eval (clips/) | ~34 GB |
| Common Voice subset (15k clips) | ~3-5 GB |
| Cached Piper voice models | ~120 MB per multi-speaker voice, ~60 MB per single-speaker |
| Per-run WAVs (113k clips, ok_nabu scale) | ~10-15 GB |
| Per-run feature shards (113k x 5 augs) | ~20 GB |

Plan for ~100 GB on a healthy `/data` mount.

## Web UI walkthrough

### Section 1: Configure training run

- **Wake word** is required. The run name auto-derives from it (`ok nabu` -> `ok_nabu`); you can override.
- **Positive phrases** (left column) are TTS'd as positive samples. 5 to 10 spelling variants is the sweet spot. Empty defaults to the wake word itself.
- **Negative phrases (hard negatives)** (right column) are TTS'd as **negative** samples with the same emphasis as positives. Paste any false-trigger phrases you observed in production. This is the most important knob for v2+ models.
- **Piper voices**: select all, none, or high-quality-only. Use the search box to narrow. Multi-speaker voices like `en_US-libritts-high` cover hundreds of speakers per voice.
- **ElevenLabs** (only if `ELEVENLABS_API_KEY` is set in `.env`): adds external voice diversity. Note: costs per character; recommended only for positive phrases.
- **Sample volume**: per-phrase reps, adversarial-pool size, augmentations-per-clip. Defaults are tuned for DGX Spark.
- **Augmentation corpora**: each toggle has a hint. At least one corpus must be enabled.
- **Classifier + training**: model architecture, optimizer, schedule. Defaults match the openWakeWord reference recipe.

The Start button is hidden while a run is in progress, and the Cancel button is hidden when nothing is running. Cancellation is honored at every long phase (download, generation, feature extraction, training).

### Section 2: Progress

- **Phase banner** shows the current step (e.g. `phase: generate:piper:adv -> 10 phrases, 111000 synths, 10 workers`).
- **Progress bars** stack as phases come online. Each bar reaches 100% explicitly on completion.
- **Metrics tiles** populate during training: step, train_loss, val_loss, val_recall_at_p95, val_fp_per_hour, threshold, etc.
- **Live log** is fed by `logger.info` from every module in `src/*`. Long phases emit a heartbeat every ~5-10 seconds so the UI never feels hung.

### Section 3: Test a trained model

- **Model dropdown** lists everything under `/data/models/`. The pill shows the file size.
- **Download .onnx** button triggers a browser download via `/api/models/<name>`.
- **Refresh list** re-queries the models endpoint.
- **Upload audio** accepts WAV / MP3 / FLAC.
- **Record from mic** is disabled unless the page is served over HTTPS or from `localhost` (browser API requirement). The pill shows "mic disabled - HTTPS required" when it cannot run.
- **Run test** uploads the audio, scores it through the selected model, and shows max / mean score, detection count, and a score curve chart.

## Hard-negative iteration workflow

The intended dev loop for a production-quality model:

```
1. Train v1 with the standard recipe.
2. Deploy.
3. Collect false-trigger phrases for a few days (TV, kids, conversations, etc.).
4. Paste them into "Negative phrases" with the SAME run name.
5. Re-train. Piper WAVs are cached so synthesis is skipped; only the new
   hard negatives are added. Corpora downloads also skipped.
6. Repeat.
```

After 2 to 3 iterations the model converges on essentially zero false-triggers in your specific deployment environment. This is how the official `ok_nabu` model from Nabu Casa was tuned.

To force a fresh regeneration (e.g. after changing voice selection), either change the run name or delete `/data/runs/<run_name>/wavs/*/.generated_*` sentinel files.

## HTTP endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Web UI |
| GET | `/healthz` | Liveness + run status |
| POST | `/api/train/start` | Start a run (JSON `TrainRunConfig`). Returns 422 with details on invalid config. |
| POST | `/api/train/cancel` | Set the cancel flag |
| GET | `/api/train/status` | Current run state |
| GET | `/api/events` | SSE stream (phase, progress, metric, log, run_started, complete, run_error, cancelled) |
| POST | `/api/audition/piper` | One-shot Piper synth (JSON: text, voice_key, speaker_id) |
| POST | `/api/audition/elevenlabs` | One-shot ElevenLabs synth |
| GET | `/api/voices/piper` | List English Piper voices from HF manifest |
| GET | `/api/voices/elevenlabs` | List your ElevenLabs voices |
| GET | `/api/models` | List trained ONNX models with sizes |
| GET | `/api/models/{name}` | Download a trained ONNX |
| POST | `/api/test/file` | Score an uploaded audio file (multipart: model_name, threshold, audio) |

## Environment variables

All settable via `.env` in the repo root. Defaults are in `src/settings.py` and `.env.example`.

| Variable | Default | Purpose |
|---|---|---|
| `OWW_DATA_DIR_HOST` | (unset; uses named volume) | Host path to bind-mount at `/data` |
| `OWW_PORT`, `WEB_PORT` | 8000 | Web UI port |
| `OWW_LOG_LEVEL` | INFO | Python logging level |
| `OWW_GENERATION_WORKERS` | 0 (auto: `min(10, cpu_count)`) | Piper parallel pool size |
| `OWW_PIPER_ORT_THREADS` | 2 | Onnxruntime intra-op threads per Piper worker |
| `OWW_DATALOADER_WORKERS` | 0 (auto: `min(8, cpu_count)`) | PyTorch DataLoader workers |
| `HF_TOKEN` | (none) | Hugging Face token; required for Common Voice |
| `ELEVENLABS_API_KEY` | (none) | Optional, unlocks ElevenLabs voices |

Defaults pair `workers * threads = 20` to match DGX Spark's 20 Grace cores exactly without thrashing.

## Tuning for quality

Defaults are tuned for "extremely high quality" on the DGX Spark. Further knobs:

- **More voices, more accents.** Toggle every English Piper voice. Enable ElevenLabs (`ELEVENLABS_API_KEY`) for additional accent variations.
- **More augmentations per clip.** Bump from 5 to 8. Each WAV becomes more variants, forcing the classifier to learn channel-invariant cues. Linearly grows feature-extraction time.
- **More adversarial phrases.** From 3,000 to 10,000+. The main failure mode of a wake-word classifier is firing on phonetically-similar phrases.
- **More Common Voice.** 15k is enough to learn "real human speech is not the wake word"; 50k+ helps with rare-accent generalization.
- **Higher batch size** on big GPUs (DGX Spark handles 4096+).

What NOT to tune unless you really know:

- Augmentation probabilities. Default `RIR p = 0.7`, `noise p = 0.7` is the openWakeWord recipe. Pushing higher trades recall for noise robustness; pushing lower trades robustness for clean-mic recall.
- Min SNR (currently `3 dB`). Going negative (noise louder than signal) makes the model brittle to noise-as-signal false-triggers. Use hard-negatives instead.

## Known limitations

- **First build compiles onnxruntime-gpu from source** (no pre-built aarch64+CUDA wheel exists on PyPI). 25 to 60 min cold; cached after that.
- **`mozilla-foundation/common_voice_17_0` on Hugging Face requires accepting Mozilla's terms** once on huggingface.co before your `HF_TOKEN` can download. The trainer uses `fsicoli/common_voice_17_0` (community parquet-less mirror) and bypasses the gating with direct tar-shard fetches, but you still need a valid HF token.
- **Eigen FetchContent** is patched at build time to use the GitHub mirror because gitlab.com Cloudflare-blocks anonymous downloads from build containers.
- **`openwakeword==0.6.0` declares `tflite-runtime`** which has no aarch64 wheel past Python 3.11. We `pip install --no-deps openwakeword` and use the ONNX feature-model path only.
- **Build is memory-hungry** during the flash-attention CUDA kernel compile phase. Defaults are conservative (`--parallel 8 --nvcc_threads 1`, peak ~50 GB). If you have a smaller host, lower `BUILD_PARALLEL` in the Dockerfile.

## Local development (no Docker)

For iterating on UI / pipeline code without a full image rebuild:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install PyTorch from the cu128 index first.
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchaudio

# Install the rest. Note: onnxruntime here is CPU-only on most platforms.
pip install -r requirements.txt
pip install --no-deps openwakeword==0.6.0
pip install fast-mp3-augment

python -c "import openwakeword; openwakeword.utils.download_models()"

export OWW_DATA_DIR=$(pwd)/data
python -m uvicorn src.main:app --reload --port 8000
```

`espeak-ng` is required by Piper for phoneme generation. Install via your OS package manager (`brew install espeak-ng`, `apt install espeak-ng`, or scoop on Windows).

`/data` and `/app/src` are bind-mounted in the production compose file. For local dev, the same hot-reload behavior comes from `uvicorn --reload`.

## Project layout

```
.
├── Dockerfile              # multi-stage: builds onnxruntime-gpu from source then runtime image
├── compose.yaml            # named volume by default; OWW_DATA_DIR_HOST switches to bind mount
├── requirements.txt
├── .env.example
└── src/
    ├── main.py             # FastAPI entry
    ├── settings.py         # pydantic-settings
    ├── config_schema.py    # TrainRunConfig + model_validator
    ├── tts/
    │   ├── piper_generator.py    # process-pool Piper synthesis
    │   ├── elevenlabs_generator.py
    │   └── voices.py             # HF voice manifest + downloads
    ├── augment/
    │   ├── augmenter.py          # audiomentations chain
    │   └── downloader.py         # MIT IR, MUSAN, FSD50K, Common Voice
    ├── data/
    │   ├── features.py           # mel + speech_embedding ONNX (CUDA EP)
    │   ├── adversarial.py        # auto-generated hard negatives
    │   └── dataset.py            # memmap dataset
    ├── train/
    │   ├── model.py              # WakeWordDNN / RNN
    │   ├── trainer.py            # BCE + recall@p95 + early stop
    │   ├── export.py             # ONNX export
    │   └── progress.py           # EventBus + BusLoggingHandler
    ├── inference/
    │   └── tester.py             # /api/test/file
    ├── pipeline/
    │   └── orchestrator.py       # full run state machine + sentinels
    └── webui/
        ├── app.py                # FastAPI routes
        ├── templates/{base,index}.html
        └── static/{style.css,app.js}
```

## License

Provided as-is. openWakeWord, Piper, audiomentations, ONNX Runtime, and the bundled datasets each carry their own licenses; review them before redistributing trained models.