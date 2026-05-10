# OpenWakeWord Trainer

A Dockerized, GPU-first trainer for [openWakeWord](https://github.com/dscripka/openWakeWord) custom models with a Web UI. Generates synthetic positives with [Piper](https://github.com/OHF-Voice/piper1-gpl) (and optional ElevenLabs voices), pulls real-world augmentation corpora (MIT IR Survey, MUSAN, FSD50K, Common Voice), trains a small classifier head on top of Google's frozen speech-embedding model, and exports an ONNX model that drops into the openWakeWord runtime.

## Features

- One-shot Docker setup. `docker compose up --build` and open http://localhost:8000.
- Web UI (FastAPI + HTMX + Server-Sent Events) for run config, live training progress, sample auditioning, and on-device model testing (file upload or browser microphone).
- Synthetic positives via every English Piper voice (US, GB, AU, IN dialects, multi-speaker libritts variants).
- Optional ElevenLabs voices for additional accent diversity, generated at 16 kHz PCM (no resampling).
- Hard-negative phrase generation including phonetic neighbors of the wake word plus a curated conversational pool.
- Multi-corpus augmentation pipeline: MIT IR Survey RIRs, MUSAN noise + music, FSD50K environmental sounds, Common Voice clips as negative speech.
- audiomentations chain: room reverb, background-noise SNR mixing, pitch shift, time stretch, parametric EQ, air absorption, gain, MP3 compression.
- Multi-architecture image. Runs on x86_64 NVIDIA hosts and on **NVIDIA DGX Spark** (Grace + Blackwell, aarch64).

## Architecture

```
+------------------+     +--------------------+     +-------------------+
| Web UI           | --> | Pipeline           | --> | openWakeWord      |
| (FastAPI + HTMX) |     | Orchestrator       |     | runtime           |
+------------------+     |                    |     | (your downstream  |
                         | 1. Generate (TTS)  |     |  app)             |
                         | 2. Download corpora|     +-------------------+
                         | 3. Augment + feat. |
                         | 4. Train classifier|
                         | 5. Export ONNX     |
                         +--------------------+
```

The classifier ONNX accepts `(1, 16, 96)` float32 input -- 16 consecutive 96-dim Google speech-embeddings, which is exactly what `openwakeword.Model(wakeword_models=["mymodel.onnx"])` expects.

## Quickstart

Prerequisites:

- Docker 24+ with Compose v2.
- NVIDIA GPU with `nvidia-container-toolkit` installed.
- Driver supporting CUDA 12.8 (R555+).

```bash
git clone <this repo>
cd openwakeword-trainer

cp .env.example .env
# Optional: paste an ELEVENLABS_API_KEY and / or HF_TOKEN.

docker compose up --build
```

Open http://localhost:8000.

1. Type a wake word (e.g. `hey jarvis`) -- required, no default.
2. Pick voices. Click "High quality only" for a sensible default set.
3. Choose augmentation corpora (all four are recommended; ~45 GB total disk on first run).
4. Click "Start training". Live progress streams into the right panel.
5. When done, the model lands in `/data/models/<run_id>.onnx` and the test panel auto-refreshes. Test it with a recording or upload.

To pull the trained ONNX off the volume:

```bash
docker cp oww-trainer:/data/models/<run_id>.onnx ./wakeword.onnx
```

## DGX Spark

DGX Spark uses a GB10 Grace + Blackwell SoC (aarch64 + sm_100). It runs unmodified containers, but you need:

1. CUDA 12.8 base image (this repo's Dockerfile uses `nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04`, which is multi-arch).
2. PyTorch wheels from the cu128 index (this repo pins `--index-url https://download.pytorch.org/whl/cu128`, which publishes both x86_64 and aarch64 builds).
3. `nvidia-container-toolkit` configured on the DGX OS host (default on shipped DGX Spark units).

Run:

```bash
docker compose up --build
```

If buildx defaults to a non-native platform you can force it:

```bash
docker buildx build --platform linux/arm64 -t oww-trainer:dgx .
```

Verify GPU visibility from inside the container:

```bash
docker compose exec trainer python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

You should see `True` and `NVIDIA GB10` (or similar).

## Volume layout

Everything mutable lives in the `oww_trainer_data` named volume mounted at `/data`:

```
/data
  piper_voices/        downloaded Piper voice .onnx + .json
  augmentations/
    rirs/mit/          MIT IR Survey wavs
    musan/             musan/{noise,music,speech}/
    fsd50k/            FSD50K.dev_audio/, FSD50K.eval_audio/, ground_truth/
    common_voice/en/   wav clips streamed from HF
  generated/           historical generated wavs (per run)
  runs/<run_id>/       per-run config, features (memmap), best.pt, wakeword.onnx
  models/              published trained ONNX models
  cache/               HF / torch caches
```

First run is slow because corpora are downloaded once. Subsequent runs reuse them.

### Disk budget

| Corpus              | Compressed | Extracted |
|---------------------|------------|-----------|
| MIT IR Survey       | ~400 MB    | ~700 MB   |
| MUSAN               | ~11 GB     | ~26 GB    |
| FSD50K dev + eval   | ~34 GB     | ~36 GB    |
| Common Voice subset | streamed   | ~3-5 GB (10k clips at 16 kHz) |

Disable any of them in the UI if you are short on disk.

## Tuning for quality

Defaults are tuned for "extremely high quality" per the project goal. If you want to push further:

- **More voices, more accents.** Toggle every English Piper voice. Add ElevenLabs voices via the UI (set `ELEVENLABS_API_KEY`) -- v3 multilingual covers ~40 accent variants.
- **More augmentations per clip.** Bump "augmentations per clip" from 3 to 5-8. This multiplies the dataset size and forces the classifier to learn channel-invariant cues.
- **More adversarial phrases.** Increase from 2,000 to 10,000+. The classifier's main failure mode is responding to phonetically-similar phrases.
- **Larger Common Voice subset.** 10k clips is enough to learn "speech that is not the wake word"; 50k+ helps with rare-accent generalization.
- **Higher batch size** on big GPUs (DGX Spark: 4096+).

## Endpoints

| Method | Path                          | Description                              |
|--------|-------------------------------|------------------------------------------|
| GET    | `/`                           | Web UI                                   |
| GET    | `/healthz`                    | Liveness + run status                    |
| POST   | `/api/train/start`            | Start a run (JSON `TrainRunConfig`)      |
| POST   | `/api/train/cancel`           | Cancel the in-flight run                 |
| GET    | `/api/train/status`           | Current run state                        |
| GET    | `/api/events`                 | SSE stream (phase, progress, metric, log)|
| POST   | `/api/audition/piper`         | Synthesize a single phrase via Piper     |
| POST   | `/api/audition/elevenlabs`    | Synthesize via ElevenLabs                |
| GET    | `/api/voices/piper`           | List English Piper voices                |
| GET    | `/api/voices/elevenlabs`      | List ElevenLabs voices                   |
| GET    | `/api/models`                 | List trained ONNX models                 |
| GET    | `/api/models/{name}`          | Download a trained ONNX                  |
| POST   | `/api/test/file`              | Score an uploaded audio file             |

## Local development (no Docker)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchaudio
pip install -r requirements.txt
python -c "import openwakeword; openwakeword.utils.download_models()"
export OWW_DATA_DIR=$(pwd)/data
python -m uvicorn src.main:app --reload --port 8000
```

`espeak-ng` is required by Piper for phoneme generation. Install via your OS package manager (`brew install espeak-ng`, `apt install espeak-ng`, or scoop on Windows).

## License

This project is provided as-is. openWakeWord, Piper, audiomentations, and the bundled datasets each carry their own licenses; review them before redistributing trained models.
