<h1 align="center" style="border-bottom: none">
   <img alt="OpenWakeWord Trainer" src="https://raw.githubusercontent.com/jxlarrea/openwakeword-trainer/refs/heads/main/src/assets/logo-wide.png" width="650" />
</h1>

<p align="center">
<img src="https://img.shields.io/github/stars/jxlarrea/openwakeword-trainer?style=for-the-badge&label=Stars&color=yellow" alt="Stars">
<a href="https://github.com/jxlarrea/openwakeword-trainer/releases"><img src="https://shields.io/github/v/release/jxlarrea/openwakeword-trainer?style=for-the-badge&color=purple" alt="version"></a>
<a href="https://buymeacoffee.com/jxlarrea"><img src="https://img.shields.io/badge/Buy%20Me%20A%20Coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me A Coffee"></a>
</p>

A Dockerized, GPU-accelerated trainer for [openWakeWord](https://github.com/dscripka/openWakeWord) custom models with a Web UI. It is built especially for people using [Voice Satellite for Home Assistant](https://github.com/jxlarrea/voice-satellite-card-integration), where wake words usually need to work from wall-mounted tablets instead of dedicated microphone arrays. The default recipe trains for that reality: off-axis speech, room echo, background noise, quieter voices, and the slightly muffled capture you get when someone talks to a tablet from across the room. Generates synthetic positives and hard negatives with [Piper](https://github.com/OHF-Voice/piper1-gpl), high-quality local positives with [Kokoro](https://github.com/hexgrad/kokoro), and optional ElevenLabs voices, pulls real-world augmentation corpora (MIT IR Survey, BUT ReverbDB, MUSAN, FSD50K, Common Voice), uses the official openWakeWord ACAV100M/validation negative feature banks, trains a small classifier head on top of Google's frozen speech-embedding model, and exports an ONNX model that drops into the openWakeWord runtime.

Built and tuned on **NVIDIA DGX Spark** (Grace + Blackwell GB10, aarch64) but the same image runs on any NVIDIA host with CUDA 12.8 drivers.

## Table of contents

- [Highlights](#highlights)
- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [DGX Spark notes](#dgx-spark-notes)
- [Desktop NVIDIA GPUs](#desktop-nvidia-gpus)
- [Storage layout](#storage-layout)
- [Web UI walkthrough](#web-ui-walkthrough)
- [Example trained models](#example-trained-models)
- [Hard-negative iteration workflow](#hard-negative-iteration-workflow)
- [Training/export behavior](#trainingexport-behavior)
- [HTTP endpoints](#http-endpoints)
- [Environment variables](#environment-variables)
- [Tuning for quality](#tuning-for-quality)
- [Known limitations](#known-limitations)
- [Local development](#local-development-no-docker)
- [License](#license)

## Highlights

- **GPU end-to-end.** Feature extraction (Google speech-embedding ONNX) runs on the GPU via `onnxruntime-gpu` (public wheel on amd64, source-built on aarch64). PyTorch trains the classifier head on the same device.
- **Parallel TTS.** Piper synthesis runs across a process pool (10 workers x 2 ORT threads on DGX Spark by default) so generating ~100k clips takes ~20 min instead of ~4 hours.
- **Kokoro positives.** Kokoro is enabled by default as an additive high-quality local TTS source for positive wake-word phrases.
- **Session-based workflow.** Create or select a wake-word session first. A session owns `/data/runs/<session_id>`, so cached WAVs, features, checkpoints, config, and model naming are reused on later visits.
- **Three-page UI.** Trainer, Tester, and System pages split training, model testing/downloads, and disk/session cleanup into separate workflows.
- **Browser-resilient runs.** Running sessions stay locked if you close/reopen the browser, live logs are persisted, and the progress card stays available until the run is cancelled or finishes.
- **Resumable pipeline.** Every long phase drops sentinels: WAV generation is cached per session, corpora downloads are sentinel-marked, feature extraction checkpoints completed clips, and re-running a failed run skips work that was already done.
- **Far-field tablet robustness.** BUT ReverbDB RIRs plus tablet far-field augmentation are enabled by default to simulate distant/off-axis single-mic capture, muffling, low level, early reflections, and light device compression.
- **Hard-negative iteration.** Paste false-triggers you saw in production into the "Negative phrases" field; they get synthesized with the same emphasis as positives so the next model strongly rejects them.
- **Official negative feature banks.** ACAV100M generic negatives are added to training, and the official openWakeWord validation negatives are used for FP/hr calibration.
- **Quality-first defaults.** The default UI settings are tuned from tablet testing to produce high-quality, low-false-positive models without extra knob turning.
- **False-positive guarded export.** The trainer calibrates the validated threshold into the exported ONNX and refuses to publish models that miss the configured recall/FP/hr gates.
- **Packaged exports.** Successful runs publish both a bare `.onnx` and a `.zip` with the ONNX, exact training config, checkpoint, and checkpoint metadata.
- **Live progress.** Web UI shows phase banner, multiple progress bars, validation metrics, and a streaming log fed by every module's `logger.info` calls.
- **Live system telemetry.** CPU, RAM, GPU utilization, GPU temperature, and VRAM (when reported by `nvidia-smi`) stream into the progress panel.
- **Sane validation.** The form blocks Start when required fields are empty, when no voice is selected, or when no augmentation corpus is enabled. The same checks fire server-side as a defense.

## Architecture

```
+-----------------------+     +----------------------+     +-------------------+
| Web UI                | --> | Pipeline             | --> | openWakeWord      |
| Trainer / Tester /    |     | Orchestrator         |     | runtime           |
| System (FastAPI+SSE)  |     |                      |     | (HA / Wyoming /   |
+-----------------------+     | 1. Piper/Kokoro TTS  |     |  your app)        |
                              | 2. Hard negatives    |     +-------------------+
                              | 3. Adversarial pool  |
                              | 4. Download corpora  |
                              | 5. Add OWW neg banks |
                              | 6. Augment + feature |
                              | 7. Train + evaluate  |
                              | 8. Export ONNX + zip |
                              +----------------------+
```

The classifier ONNX accepts `(1, 16, 96)` float32 input - 16 consecutive 96-dim Google speech-embeddings, which is exactly what `openwakeword.Model(wakeword_models=["mymodel.onnx"])` expects.

## Quickstart

Prerequisites:

- Docker 24+ with Compose v2.
- NVIDIA GPU with `nvidia-container-toolkit` installed.
- Driver supporting CUDA 12.8 (R555+).
- 120 GB+ of free disk for the cached corpora, openWakeWord feature banks, sessions, and built image.

```bash
git clone https://github.com/jxlarrea/openwakeword-trainer
cd openwakeword-trainer

cp .env.example .env
# Edit .env to add HF_TOKEN (needed for Common Voice).
# Optional: paste an ELEVENLABS_API_KEY.

docker compose up --build
```

> **First build on DGX Spark / arm64 is slow.** The Dockerfile compiles `onnxruntime-gpu` from source for sm_120 Blackwell kernels because there is no public aarch64 CUDA wheel. Expect **25 to 60 minutes** on a fast CPU. On normal linux/amd64 hosts, Compose defaults to the public `onnxruntime-gpu` wheel instead, so the build is much faster.

Open http://localhost:8000.

The top navigation has three pages:

- **Trainer**: create/select a wake-word session, configure a run, start/cancel training, and watch persistent progress/logs.
- **Tester**: upload or record audio, score it against exported models, and download either the bare `.onnx` or the full model package.
- **System**: inspect session disk usage, delete individual session caches, delete sessions, or clear all downloaded/generated cache.

Basic training flow:

1. Open **Trainer** and create or select a wake-word session. The session name is an arbitrary stable id such as `ok_nabu_v2`; the wake word is the actual phrase to detect, such as `ok nabu`.
2. Paste positive phrases (5 to 10 spelling/pronunciation variants of the wake word).
3. Optionally paste negative phrases (false-triggers observed in production).
4. Review voices. High/medium Piper voices and the best Kokoro voices are selected by default; click **Select all** if you want every available voice.
5. Confirm augmentation corpora. ACAV100M, the validation feature bank, BUT ReverbDB, and tablet far-field augmentation are enabled by default and strongly recommended for tablet deployments.
6. Click **Start training**. The page scrolls to the progress panel, and live progress/system telemetry/logs stream there.
7. You can close the browser and come back later. The session stays locked while training is running, fields remain disabled, and the persisted progress/logs rehydrate when the UI loads.
8. When done, open **Tester** to download `<session_id>.onnx` or `<session_id>.zip`. The model also stays at `/opt/models/openwakeword-trainer/models/<session_id>.onnx` on the host (or inside the docker volume if you didn't set `OWW_DATA_DIR_HOST`).

Sessions are saved to disk. Reopen the UI later, select the same session, and the trainer will reuse cached WAVs/features/checkpoints where possible. Use **System** to remove one session's cache, delete a session entirely, or clear shared downloaded corpora/feature banks.

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

## Desktop NVIDIA GPUs

On regular linux/x86_64 systems (for example RTX 4090 or RTX 5090 desktops), the default Compose build uses the public `onnxruntime-gpu` wheel instead of compiling ONNX Runtime from source:

```bash
docker compose build
docker compose up
```

That path ignores `OWW_CUDA_ARCHITECTURES`; the wheel already contains the CUDA provider. You still need an NVIDIA driver new enough for CUDA 12.8 and `nvidia-container-toolkit`.

If you explicitly force an ONNX Runtime source build on x86_64, set the GPU architecture in `.env`:

```bash
# RTX 4090 / Ada Lovelace
OWW_ORT_BUILD_FROM_SOURCE=true
OWW_CUDA_ARCHITECTURES=89

# DGX Spark GB10 / RTX 50-series Blackwell
OWW_ORT_BUILD_FROM_SOURCE=true
OWW_CUDA_ARCHITECTURES=120

# One image for both families, at the cost of much higher compile time/RAM
OWW_ORT_BUILD_FROM_SOURCE=true
OWW_CUDA_ARCHITECTURES=89;120
```

For most x86_64 users, leave `OWW_ORT_BUILD_FROM_SOURCE=auto`.

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
    rirs/but_reverbdb/ BUT ReverbDB measured far-field RIRs
    musan/             musan/{noise,music,speech}/
    fsd50k/clips/      FSD50K individual wavs (via Fhrozen/FSD50k HF mirror)
    common_voice/en/   16 kHz wavs decoded from CV tar shards
  openwakeword_features/
    openwakeword_features_ACAV100M_2000_hrs_16bit.npy
    validation_set_features.npy
  runs/<session_id>/
    session.json       durable UI session metadata + latest form config
    config.json        the exact TrainRunConfig that was submitted
    wavs/
      positive/        positive synthesized wavs + .generated_pos sentinel
      negative/        adversarial + hard-negative wavs + sentinels
    train_features.bin val_features.bin   memmap feature shards
    features_resume.json                  resumable feature-extraction checkpoint
    train_labels.npy   val_labels.npy
    best.pt            best classifier weights
    wakeword.onnx      exported model
    result.json        final metrics + history
  models/              <session_id>.onnx and <session_id>.zip, served by Tester downloads
  cache/               HF / torch caches (HF_HOME, TORCH_HOME)
```

### Disk budget

| Corpus | Disk on /data |
|---|---|
| MIT IR Survey (RIRs) | ~10 MB |
| BUT ReverbDB RIR-only release | ~9 GB download, larger after extraction |
| MUSAN (noise + music + speech) | ~26 GB |
| FSD50K dev + eval (clips/) | ~34 GB |
| Common Voice subset (100k clips) | ~20-30 GB |
| openWakeWord ACAV100M feature bank | ~17 GB |
| openWakeWord validation feature bank | ~177 MB |
| Cached Piper voice models | ~120 MB per multi-speaker voice, ~60 MB per single-speaker |
| Per-run WAVs (113k clips, ok_nabu scale) | ~10-15 GB |
| Per-run feature shards (113k x 5 augs) | ~20 GB |

Plan for 120 GB+ on a healthy `/data` mount if you enable all corpora and keep several sessions around. Use **System -> Delete cache** to remove generated data for one session while keeping its settings, **Delete session** to remove the session and its model, or **Delete all disk cache** to reclaim shared downloaded/generated data while preserving sessions.

## Web UI walkthrough

### Navigation

The header includes three primary pages:

- **Trainer** (`/`): session selection/creation, run configuration, start/cancel, persistent progress, metrics, logs, and live system telemetry.
- **Tester** (`/tester`): exported model downloads and audio scoring.
- **System** (`/system`): session inventory, disk usage, and cleanup actions.

### Trainer page

#### Session

- **Wake-word session** selects an existing durable run. Existing `/data/runs/<id>` directories with `session.json` or `config.json` appear here.
- **Create session** asks for two values: a session name and the actual wake word. The session name becomes the stable cache/model id (`ok_nabu_v2`), while the wake word remains the phrase the model should detect (`ok nabu`).
- Explicit session names cannot overwrite an existing session. Create a new name for each experiment so older models remain available for comparison.
- **Delete session** removes `/data/runs/<session_id>`, `/data/models/<session_id>.onnx`, and `/data/models/<session_id>.zip`.
- Selecting a session fills the form with the last saved config and shows its cache/model status.
- While any training run is active, session creation/deletion is locked so two sessions cannot train at the same time.

#### Configure training run

- **Wake word** is set when the session is created and kept read-only so cached data stays attached to the correct phrase.
- **Session / model name** is the session id and is also read-only. This keeps the output model, package, and run directory stable across retries.
- **Positive phrases** (left column) are TTS'd as positive samples. 5 to 10 spelling variants is the sweet spot. Empty defaults to the wake word itself.
- **Negative phrases (hard negatives)** (right column) are TTS'd as **negative** samples with the same emphasis as positives. Paste any false-trigger phrases you observed in production. This is the most important knob for v2+ models.
- **Piper voices**: high/medium voices are selected by default; select all, none, or high-quality-only as needed. Use the search box to narrow. Multi-speaker voices like `en_US-libritts-high` cover hundreds of speakers per voice.
- **Kokoro voices**: enabled by default for positive phrases. Use **Best quality** to keep only the strongest Kokoro voices if you want less synthesis volume.
- **ElevenLabs** (only if `ELEVENLABS_API_KEY` is set in `.env`): adds external voice diversity. Note: costs per character; recommended only for positive phrases.
- **Sample volume**: per-phrase reps, adversarial-pool size, augmentations-per-clip. Defaults are tuned for DGX Spark.
- **Augmentation corpora**: each toggle has a hint. At least one corpus must be enabled. BUT ReverbDB adds measured far-field room impulse responses. ACAV100M adds generic negative training windows; the openWakeWord validation bank is used for low-FP threshold calibration.
- **Tablet far-field augmentation**: default-on channel simulation for off-axis tablet microphones. It applies mic band-limiting, distance attenuation, a device/room noise floor, early reflections, and light capture compression.
- **Classifier + training**: model architecture, optimizer, schedule, and target FP/hr. Defaults are tuned for low false positives on DGX Spark.

The Start button is disabled/hidden while a run is in progress, and the Cancel button is shown only when cancellation is possible. All editable fields are disabled during training so the saved session config cannot drift under a running process. Cancellation is honored at every long phase (download, generation, feature extraction, training).

#### Progress

- The progress card appears at the top once a run starts and remains visible after cancellation or completion.
- The title includes the active run id, e.g. `Progress: ok_nabu`.
- **Phase banner** shows the current step (e.g. `phase: generate:piper:adv -> 10 phrases, 111000 synths, 10 workers`).
- **Progress bars** are cleared for each new run, stack as phases come online, and reach 100% explicitly on completion.
- **Metrics tiles** populate during training: step, train_loss, val_loss, recall@p95, recall@targetFP, FP/hr, threshold, etc.
- **System metrics** update continuously: CPU, RAM, GPU, GPU temperature, and VRAM on systems where `nvidia-smi` reports dedicated memory. DGX Spark uses unified memory, so VRAM may be absent while RAM still reflects total pressure.
- **Live log** is persisted in the progress snapshot. If you close and reopen the browser during a run, the current phase/progress/metrics/logs rehydrate from `/api/train/status`.
- **Cancel training** is available from the progress card, which matters after reload because the rest of the form remains locked.

### Tester page

- **Model dropdown** lists everything under `/data/models/`. The pill shows the ONNX file size and package size when a zip exists.
- **Download .onnx** button triggers a browser download via `/api/models/<name>`.
- **Download package** downloads `/data/models/<session_id>.zip`, which contains the ONNX, exact training config, checkpoint, and checkpoint metadata.
- **Refresh list** re-queries the models endpoint without leaving the page.
- **Upload audio** accepts common browser/server-decodable audio files. The server can decode WAV/MP3/FLAC through libsndfile and browser-recorded WebM/Opus through ffmpeg.
- **Record from mic** is disabled unless the page is served over HTTPS or from `localhost` (browser API requirement). The pill shows "mic disabled - HTTPS required" when it cannot run.
- **Run test** uploads the audio, scores it through the selected model, and shows max / mean score, detection count, and a score curve chart.

### System page

- **Summary tiles** show total session count, session data size, and shared cache/model size.
- **Sessions table** lists wake word, session id, disk usage, and model status for every saved session.
- **Delete cache** removes generated data for one session while keeping `session.json` and the saved settings. Use this when you want to reclaim WAV/features/checkpoints but keep the session reusable.
- **Delete session** removes the session, its generated data, and matching model/package artifacts.
- **Delete all disk cache** removes generated/downloaded cache and trained models while preserving sessions and their parameters.
- **Delete all sessions and cache** removes sessions, saved settings, generated data, downloaded cache, and trained models.
- Destructive System actions are disabled while training is running.

## Example trained models

The `trained_models/` folder contains models trained with this project, including the ONNX file plus the config/checkpoint metadata used to produce it. Use them as examples of the exported artifact format, or copy the ONNX into your openWakeWord runtime to test it directly.

## Hard-negative iteration workflow

The intended dev loop for a production-quality model:

```
1. Train v1 with the standard recipe.
2. Deploy.
3. Collect false-trigger phrases for a few days (TV, kids, conversations, etc.).
4. Reopen the same session and paste them into "Negative phrases".
5. Re-train. Piper WAVs are cached so synthesis is skipped; only the new
   hard negatives are added. Corpora downloads also skipped.
6. Repeat.
```

After 2 to 3 iterations the model converges on essentially zero false-triggers in your specific deployment environment. This is how the official `ok_nabu` model from Nabu Casa was tuned.

To force a fresh regeneration (e.g. after changing voice selection), create a new session or delete `/data/runs/<session_id>/wavs/*/.generated_*` sentinel files. To reclaim space, use the **System** page to delete a session cache or remove the session entirely.

## Training/export behavior

The trainer optimizes for low false positives, not just low validation loss:

- Negative pressure ramps up during training so generic speech/noise remains dominant without crushing early positive learning.
- Focal loss, label smoothing, mixup, AdamW, and a warmup/hold/cosine LR schedule are enabled by default.
- Optional positive-confidence, negative-confidence, and separation losses are available for experiments, but are off in the shipped v7 defaults.
- Validation reports target-threshold metrics and raw `0.5` metrics, including `recall@targetFP`, `FP/hr`, `raw recall@0.5`, `raw FP/hr@0.5`, positive median score, and positive p10 score.
- Export is refused unless the best checkpoint meets both calibrated and raw-score quality gates.
- The shipped default export gates are `0.5` FP/hr target recall >= `0.70`, calibration threshold <= `0.80`, raw recall@0.5 >= `0.80`, raw FP/hr@0.5 <= `10`, positive median >= `0.75`, and positive p10 >= `0.35`.
- Low-quality runs still produce logs and diagnostics, but they are not exported or published as usable models.
- The selected operating threshold is baked into the exported ONNX so a runtime threshold of `0.5` maps to the validated threshold.
- Successful runs publish both `/data/models/<session>.onnx` and `/data/models/<session>.zip`. The zip contains the ONNX model, `training_config.json`, `checkpoint.pt`, and `checkpoint_metadata.json`.

## HTTP endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Trainer page |
| GET | `/tester` | Tester page |
| GET | `/system` | System page |
| GET/HEAD | `/favicon.ico` | Browser favicon |
| GET | `/healthz` | Liveness + run status |
| GET | `/api/sessions` | List saved wake-word sessions |
| POST | `/api/sessions` | Create a session (JSON: `wake_word`) |
| GET | `/api/sessions/{session_id}` | Load one session, including saved config and cache size |
| DELETE | `/api/sessions/{session_id}` | Delete a session, run cache, and matching model/package |
| GET | `/api/system/disk` | Disk summary for sessions, shared cache, and models |
| DELETE | `/api/system/sessions/{session_id}/cache` | Delete generated data/model for one session while keeping settings |
| DELETE | `/api/system/cache` | Delete shared/generated disk cache while preserving sessions |
| DELETE | `/api/system/all` | Delete all sessions and cache |
| POST | `/api/train/start` | Start a run (JSON `TrainRunConfig`). Returns 422 with details on invalid config. |
| POST | `/api/train/cancel` | Set the cancel flag |
| GET | `/api/train/status` | Current run state plus current system telemetry |
| GET | `/api/events` | SSE stream (phase, progress, metric, system, log, run_started, complete, run_error, cancelled) |
| POST | `/api/audition/piper` | One-shot Piper synth (JSON: text, voice_key, speaker_id) |
| POST | `/api/audition/elevenlabs` | One-shot ElevenLabs synth |
| GET | `/api/voices/piper` | List English Piper voices from HF manifest |
| GET | `/api/voices/elevenlabs` | List your ElevenLabs voices |
| GET | `/api/models` | List trained ONNX models with ONNX/package sizes |
| GET | `/api/models/{name}` | Download a trained ONNX |
| GET | `/api/model-packages/{name}` | Download a trained model zip package |
| POST | `/api/test/file` | Score an uploaded audio file (multipart: model_name, threshold, audio) |

## Environment variables

All settable via `.env` in the repo root. Defaults are in `src/settings.py` and `.env.example`.

| Variable | Default | Purpose |
|---|---|---|
| `OWW_DATA_DIR_HOST` | (unset; uses named volume) | Host path to bind-mount at `/data` |
| `OWW_PORT`, `WEB_PORT` | 8000 | Web UI port |
| `OWW_LOG_LEVEL` | INFO | Python logging level |
| `OWW_ORT_BUILD_FROM_SOURCE` | auto | Build-time: `auto` uses PyPI ORT GPU wheel on amd64 and source-build on arm64 |
| `OWW_CUDA_ARCHITECTURES` | 120 | Build-time: CUDA arch list for source-built ONNX Runtime (`89` for RTX 4090, `120` for GB10/RTX 50-series) |
| `OWW_BUILD_PARALLEL` | 8 | Build-time: source-build parallelism for ONNX Runtime |
| `OWW_GENERATION_WORKERS` | 0 (auto: `min(10, cpu_count)`) | Piper parallel pool size |
| `OWW_PIPER_ORT_THREADS` | 2 | Onnxruntime intra-op threads per Piper worker |
| `OWW_PIPER_USE_CUDA` | true | Run Piper TTS on GPU |
| `OWW_PIPER_MAX_TASKS_PER_CHILD` | 250 | Recycle Piper workers periodically to release native/CUDA memory |
| `OWW_PIPER_RELEASE_AFTER_SYNTH` | false | Strict per-synth Piper cache release; much slower |
| `OWW_KOKORO_DEVICE` | cuda | Torch device for Kokoro TTS; CUDA uses Kokoro's custom STFT path to avoid complex-kernel JIT issues on newer GPUs |
| `OWW_DATALOADER_WORKERS` | 0 (auto: `min(8, cpu_count)`) | PyTorch DataLoader workers |
| `HF_TOKEN` | (none) | Hugging Face token; required for Common Voice |
| `ELEVENLABS_API_KEY` | (none) | Optional, unlocks ElevenLabs voices |

Defaults pair `workers * threads = 20` to match DGX Spark's 20 Grace cores exactly without thrashing.

## Tuning for quality

Defaults are tuned for high-quality, low-false-positive models on the DGX Spark, including noisy/far-field tablet use. Further knobs:

- **More voices, more accents.** High/medium Piper voices are the default quality set. Toggle every English Piper voice only if you want more variety and are willing to test whether lower-quality voices help your phrase. Enable ElevenLabs (`ELEVENLABS_API_KEY`) for additional accent variations.
- **Use Kokoro for natural positives and hard negatives.** Kokoro is on by default with 2 positive renders per phrase/voice and 1 hard-negative render per phrase/voice.
- **More augmentations per clip.** Bump from 6 to 8. Each WAV becomes more variants, forcing the classifier to learn channel-invariant cues. Linearly grows feature-extraction time.
- **Tablet / far-field deployments.** BUT ReverbDB, Tablet far-field augmentation, `RIR p = 0.9`, and 6 augmentations per clip are the defaults. Increase tablet far-field probability toward `0.8` for harsher off-axis environments.
- **More adversarial phrases.** The default is 8,000. Increase toward 10,000+ for especially confusable wake words.
- **More Common Voice.** 100k is the default speech-negative pool on DGX-class hardware; lower this on smaller machines if disk/time is tight.
- **Keep ACAV100M and validation negatives enabled.** They are the biggest generic false-positive guardrail and make FP/hr calibration meaningful.
- **Higher batch size** on big GPUs (DGX Spark handles 4096+).

What NOT to tune unless you really know:

- Augmentation probabilities. Default `RIR p = 0.9`, `noise p = 0.75` is tuned for tablet/far-field robustness. Pushing higher trades clean/direct recall for noise/room robustness; pushing lower trades robustness for clean-mic recall.
- Min SNR (currently `3 dB`). Going negative (noise louder than signal) makes the model brittle to noise-as-signal false-triggers. Use hard-negatives instead.

## Known limitations

- **DGX Spark / arm64 builds compile onnxruntime-gpu from source** (no pre-built aarch64+CUDA wheel exists on PyPI). 25 to 60 min cold; cached after that. linux/amd64 builds use the public wheel by default.
- **First full-data run downloads a lot of data.** ACAV100M alone is ~17 GB, FSD50K is ~34 GB, MUSAN is ~26 GB, and BUT ReverbDB is several GB more if enabled. Keep `/data` on a filesystem with enough headroom.
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

`espeak-ng` is required by Piper for phoneme generation. `ffmpeg` is required if you want the Tester page to decode browser-recorded WebM/Opus locally. Install via your OS package manager (`brew install espeak-ng ffmpeg`, `apt install espeak-ng ffmpeg`, or scoop on Windows).

`/data` and `/app/src` are bind-mounted in the production compose file. For local dev, the same hot-reload behavior comes from `uvicorn --reload`.

## License

Provided as-is. openWakeWord, Piper, audiomentations, ONNX Runtime, and the bundled datasets each carry their own licenses; review them before redistributing trained models.
