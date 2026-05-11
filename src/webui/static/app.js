// OpenWakeWord Trainer client-side app.

(function () {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  // ---------- Auto run-name from wake word ----------

  const wakeInput = $("input[name='wake_word']");
  const runNameInput = $("#run-name-input");
  let lastAutoRunName = "";

  function slugify(s) {
    return String(s || "")
      .toLowerCase()
      .trim()
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_+|_+$/g, "");
  }

  function applyAutoRunName() {
    if (!wakeInput || !runNameInput) return;
    const slug = slugify(wakeInput.value);
    // Only overwrite if the user hasn't typed their own value.
    if (!runNameInput.value || runNameInput.value === lastAutoRunName) {
      runNameInput.value = slug;
    }
    lastAutoRunName = slug;
  }
  wakeInput?.addEventListener("input", applyAutoRunName);
  // Init from server-rendered value
  applyAutoRunName();

  // ---------- Voice list filtering / select-all ----------

  $("#select-all-voices")?.addEventListener("click", () => {
    $$("#config .voice-row input[name='piper_voice']").forEach((cb) => (cb.checked = true));
  });
  $("#select-none-voices")?.addEventListener("click", () => {
    $$("#config .voice-row input[name='piper_voice']").forEach((cb) => (cb.checked = false));
  });
  $("#select-high-quality")?.addEventListener("click", () => {
    $$("#config .voice-row").forEach((row) => {
      const q = row.dataset.quality;
      const cb = row.querySelector("input[name='piper_voice']");
      cb.checked = q === "high" || q === "medium";
    });
  });
  $("#voice-filter")?.addEventListener("input", (e) => {
    const q = e.target.value.toLowerCase().trim();
    $$("#config .voice-row").forEach((row) => {
      row.classList.toggle("hidden", q && !row.dataset.key.toLowerCase().includes(q));
    });
  });

  // ---------- ElevenLabs voices ----------

  $("#load-elevenlabs-voices")?.addEventListener("click", async () => {
    const container = $("#elevenlabs-voices");
    container.innerHTML = "<p class='hint'>loading...</p>";
    const res = await fetch("/api/voices/elevenlabs");
    const voices = await res.json();
    if (!voices.length) {
      container.innerHTML = "<p class='hint'>No voices returned. Check API key.</p>";
      return;
    }
    container.innerHTML = voices
      .map(
        (v) => `
        <label class="voice-row" data-key="${v.voice_id}">
          <input type="checkbox" name="elevenlabs_voice_id" value="${v.voice_id}">
          <span class="voice-name">${v.name}</span>
          <span class="voice-meta">${v.category || ""}</span>
        </label>`
      )
      .join("");
  });

  // ---------- Submit training run ----------

  function clearValidationState(form) {
    form.querySelectorAll("fieldset.invalid").forEach((fs) => fs.classList.remove("invalid"));
    const box = $("#form-errors");
    if (box) {
      box.hidden = true;
      box.innerHTML = "";
    }
  }

  function showValidationErrors(form, errors, firstInvalidElement) {
    const box = $("#form-errors");
    if (!box) return;
    box.hidden = false;
    box.innerHTML =
      `<strong>Fix these before starting:</strong>` +
      `<ul>${errors.map((e) => `<li>${e}</li>`).join("")}</ul>`;
    if (firstInvalidElement) {
      firstInvalidElement.scrollIntoView({ behavior: "smooth", block: "center" });
    } else {
      box.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }

  function validateTrainForm(form) {
    clearValidationState(form);
    const errors = [];
    let firstInvalid = null;

    // Native HTML5 first - lets the browser highlight required fields,
    // out-of-range numbers, etc., with its built-in messages.
    if (!form.checkValidity()) {
      form.reportValidity();
      const firstNative = form.querySelector(":invalid");
      return { errors: ["Some required fields are missing or out of range."], firstInvalid: firstNative };
    }

    // Custom: at least one voice source.
    const piperCount = form.querySelectorAll('input[name="piper_voice"]:checked').length;
    const elUsed = form.querySelector('input[name="use_elevenlabs"]')?.checked;
    const elCount = form.querySelectorAll('input[name="elevenlabs_voice_id"]:checked').length;

    if (piperCount === 0 && !(elUsed && elCount > 0)) {
      errors.push("Select at least one Piper voice (or enable ElevenLabs with voices).");
      const voicesFieldset = form.querySelector(".voice-list")?.closest("fieldset");
      voicesFieldset?.classList.add("invalid");
      if (!firstInvalid) firstInvalid = voicesFieldset;
    }

    // Custom: at least one augmentation corpus must be on.
    const corpusBoxes = [
      "use_mit_rirs",
      "use_musan_noise",
      "use_musan_music",
      "use_fsd50k",
      "use_common_voice_negatives",
    ];
    const anyCorpus = corpusBoxes.some(
      (name) => form.querySelector(`input[name="${name}"]`)?.checked
    );
    if (!anyCorpus) {
      errors.push(
        "Enable at least one augmentation corpus (MIT IR, MUSAN, FSD50K, or Common Voice)."
      );
      const corpusFieldset = form.querySelector(
        'input[name="use_mit_rirs"]'
      )?.closest("fieldset");
      corpusFieldset?.classList.add("invalid");
      if (!firstInvalid) firstInvalid = corpusFieldset;
    }

    // Custom: wake word can't be just whitespace (`required` only rejects empty).
    const wake = form.querySelector('input[name="wake_word"]');
    if (wake && !wake.value.trim()) {
      errors.push("Wake word is required.");
      wake.focus();
      if (!firstInvalid) firstInvalid = wake;
    }

    return { errors, firstInvalid };
  }

  $("#train-form")?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.target;

    const { errors, firstInvalid } = validateTrainForm(form);
    if (errors.length) {
      showValidationErrors(form, errors, firstInvalid);
      return;
    }

    const fd = new FormData(form);
    const payload = formToPayload(fd);
    const res = await fetch("/api/train/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const body = await res.text();
      let parsed;
      try {
        parsed = JSON.parse(body);
      } catch {}
      const msg = parsed?.detail || body;
      showValidationErrors(form, [`Server rejected the run: ${msg}`], null);
      return;
    }
    const data = await res.json();
    setRunState(data.status);
  });

  // Clear validation styling when the user starts fixing things.
  $("#train-form")?.addEventListener("change", () => {
    const form = $("#train-form");
    if (form && $("#form-errors") && !$("#form-errors").hidden) {
      // Re-validate quietly to clear/refresh state without scrolling.
      clearValidationState(form);
    }
  });

  $("#cancel-btn")?.addEventListener("click", async () => {
    await fetch("/api/train/cancel", { method: "POST" });
  });

  function formToPayload(fd) {
    const v = (k) => fd.get(k) || "";
    const vNum = (k, dflt) => {
      const x = fd.get(k);
      if (x === null || x === "") return dflt;
      const n = Number(x);
      return Number.isFinite(n) ? n : dflt;
    };
    const vBool = (k) => fd.get(k) !== null;
    const piperVoices = fd.getAll("piper_voice").map((k) => ({ voice_key: k }));
    const elVoices = fd.getAll("elevenlabs_voice_id");
    return {
      wake_word: String(v("wake_word")).trim(),
      run_name: String(v("run_name")).trim(),
      generation: {
        positive_phrases: String(v("positive_phrases"))
          .split("\n")
          .map((s) => s.trim())
          .filter(Boolean),
        n_positive_per_phrase_per_voice: vNum("n_positive_per_phrase_per_voice", 4),
        negative_phrases: String(v("negative_phrases"))
          .split("\n")
          .map((s) => s.trim())
          .filter(Boolean),
        n_negative_per_phrase_per_voice: vNum("n_negative_per_phrase_per_voice", 4),
        n_adversarial_phrases: vNum("n_adversarial_phrases", 3000),
        n_adversarial_per_phrase_per_voice: vNum("n_adversarial_per_phrase_per_voice", 1),
        piper_voices: piperVoices,
        use_elevenlabs: vBool("use_elevenlabs"),
        elevenlabs_voice_ids: elVoices,
        elevenlabs_model: String(v("elevenlabs_model")) || "eleven_multilingual_v2",
      },
      augmentation: {
        rir_probability: vNum("rir_probability", 0.7),
        background_noise_probability: vNum("background_noise_probability", 0.7),
        augmentations_per_clip: vNum("augmentations_per_clip", 5),
      },
      datasets: {
        use_mit_rirs: vBool("use_mit_rirs"),
        use_musan_noise: vBool("use_musan_noise"),
        use_musan_music: vBool("use_musan_music"),
        use_fsd50k: vBool("use_fsd50k"),
        use_common_voice_negatives: vBool("use_common_voice_negatives"),
        common_voice_subset: vNum("common_voice_subset", 15000),
      },
      training: {
        model_type: v("model_type") || "dnn",
        layer_dim: vNum("layer_dim", 128),
        n_blocks: vNum("n_blocks", 1),
        learning_rate: vNum("learning_rate", 0.0001),
        batch_size: vNum("batch_size", 2048),
        max_steps: vNum("max_steps", 75000),
        val_every_n_steps: vNum("val_every_n_steps", 500),
        early_stop_patience: vNum("early_stop_patience", 8),
        target_false_positives_per_hour: vNum("target_false_positives_per_hour", 0.2),
        seed: vNum("seed", 42),
      },
    };
  }

  // ---------- Start/Cancel button state machine ----------

  function setRunState(state) {
    const pill = $("#status-pill");
    const startBtn = $("#start-btn");
    const cancelBtn = $("#cancel-btn");
    if (!pill) return;
    pill.textContent = state || "idle";
    pill.className = "pill " + (state || "");

    const isRunning = state === "running";
    if (startBtn) startBtn.style.display = isRunning ? "none" : "";
    if (cancelBtn) cancelBtn.style.display = isRunning ? "" : "none";
  }

  // ---------- SSE: live progress ----------

  const phaseBanner = $("#phase-banner");
  const progressBars = $("#progress-bars");
  const metricsBox = $("#metrics");
  const logEl = $("#log");
  const progressMap = new Map();
  const metricMap = new Map();

  function ensureProgressBar(name) {
    let bar = progressMap.get(name);
    if (bar) return bar;
    bar = document.createElement("div");
    bar.className = "progress-bar";
    bar.innerHTML = `
      <div class="label"><span>${name}</span><span class="pct">0%</span></div>
      <div class="track"><div class="fill" style="width:0%"></div></div>
    `;
    progressBars.appendChild(bar);
    progressMap.set(name, bar);
    return bar;
  }

  function setProgress(name, frac, detail) {
    const bar = ensureProgressBar(name);
    const pct = Math.round(frac * 100);
    bar.querySelector(".fill").style.width = pct + "%";
    bar.querySelector(".pct").textContent =
      pct + "%" + (detail ? " (" + detail + ")" : "");
  }

  function setMetric(name, value) {
    let tile = metricMap.get(name);
    if (!tile) {
      tile = document.createElement("div");
      tile.className = "metric-tile";
      tile.innerHTML = `<div class="name">${name}</div><div class="value"></div>`;
      metricsBox.appendChild(tile);
      metricMap.set(name, tile);
    }
    tile.querySelector(".value").textContent = value;
  }

  function appendLog(level, msg) {
    const span = document.createElement("span");
    span.className = "l-" + (level || "info");
    span.textContent = msg + "\n";
    logEl.appendChild(span);
    logEl.scrollTop = logEl.scrollHeight;
  }

  function fmt(v) {
    if (typeof v === "number") {
      if (Math.abs(v) >= 1e4 || (v !== 0 && Math.abs(v) < 1e-3)) {
        return v.toExponential(2);
      }
      return v.toFixed(4).replace(/\.?0+$/, "");
    }
    return String(v);
  }

  const es = new EventSource("/api/events");
  es.addEventListener("phase", (e) => {
    const d = JSON.parse(e.data);
    phaseBanner.textContent = `phase: ${d.name}` + (d.detail ? ` -> ${d.detail}` : "");
  });
  es.addEventListener("progress", (e) => {
    const d = JSON.parse(e.data);
    setProgress(d.name, d.fraction, d.detail || "");
  });
  es.addEventListener("metric", (e) => {
    const d = JSON.parse(e.data);
    Object.keys(d).forEach((k) => {
      if (k === "kind" || k === "ts") return;
      setMetric(k, fmt(d[k]));
    });
  });
  es.addEventListener("log", (e) => {
    const d = JSON.parse(e.data);
    appendLog(d.level, d.message);
  });
  es.addEventListener("run_started", (e) => {
    const d = JSON.parse(e.data);
    appendLog("info", `Run started: ${d.run_id} (${d.wake_word})`);
    setRunState("running");
  });
  es.addEventListener("complete", (e) => {
    const d = JSON.parse(e.data);
    appendLog("info", `Complete -> ${d.onnx_path}`);
    setRunState("succeeded");
    refreshModels();
  });
  es.addEventListener("run_error", (e) => {
    try {
      const d = JSON.parse(e.data);
      appendLog("error", d.message);
      setRunState("failed");
    } catch {}
  });
  es.addEventListener("cancelled", () => {
    appendLog("warning", "Run cancelled");
    setRunState("cancelled");
  });
  es.onerror = () => {
    appendLog("warning", "SSE connection lost; retrying...");
  };

  // ---------- Test panel ----------

  async function refreshModels() {
    const res = await fetch("/api/models");
    const models = await res.json();
    const sel = $("#test-model");
    if (!sel) return;
    const prev = sel.value;
    sel.innerHTML = models
      .map((m) => `<option value="${m.name}" data-size="${m.size}">${m.name}</option>`)
      .join("");
    if (prev && [...sel.options].some((o) => o.value === prev)) {
      sel.value = prev;
    }
    updateModelInfo();
  }

  function updateModelInfo() {
    const sel = $("#test-model");
    const info = $("#model-info");
    if (!info || !sel) return;
    const opt = sel.options[sel.selectedIndex];
    if (!opt) {
      info.textContent = "no models yet";
      return;
    }
    const size = Number(opt.getAttribute("data-size") || 0);
    const kb = (size / 1024).toFixed(1);
    info.textContent = `${opt.value} (${kb} KB)`;
  }
  $("#test-model")?.addEventListener("change", updateModelInfo);

  $("#download-model-btn")?.addEventListener("click", () => {
    const name = $("#test-model")?.value;
    if (!name) {
      alert("No model selected.");
      return;
    }
    const a = document.createElement("a");
    a.href = `/api/models/${encodeURIComponent(name)}`;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
  });

  $("#refresh-models-btn")?.addEventListener("click", () => {
    refreshModels();
  });
  updateModelInfo();

  // ---------- Mic recording with HTTPS gate ----------

  let mediaRecorder = null;
  let recordedChunks = [];

  const isSecure = window.isSecureContext || location.hostname === "localhost" || location.hostname === "127.0.0.1";
  const recordBtn = $("#record-btn");
  const stopBtn = $("#stop-record-btn");
  const micHint = $("#mic-hint");

  if (!isSecure) {
    if (recordBtn) {
      recordBtn.disabled = true;
      recordBtn.title = "Microphone recording requires HTTPS (or localhost).";
    }
    if (micHint) {
      micHint.textContent = "mic disabled - HTTPS required";
      micHint.classList.add("failed");
    }
  } else if (micHint) {
    micHint.textContent = "mic ready";
  }

  recordBtn?.addEventListener("click", async () => {
    if (!isSecure) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
      recordedChunks = [];
      mediaRecorder.ondataavailable = (ev) => recordedChunks.push(ev.data);
      mediaRecorder.onstop = () => {
        const blob = new Blob(recordedChunks, { type: "audio/webm" });
        const file = new File([blob], "recording.webm", { type: "audio/webm" });
        const dt = new DataTransfer();
        dt.items.add(file);
        $("#test-audio-file").files = dt.files;
        stream.getTracks().forEach((t) => t.stop());
      };
      mediaRecorder.start();
      recordBtn.disabled = true;
      stopBtn.disabled = false;
      if (micHint) micHint.textContent = "recording...";
    } catch (err) {
      alert("Mic access failed: " + err.message);
    }
  });

  stopBtn?.addEventListener("click", () => {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
    }
    recordBtn.disabled = false;
    stopBtn.disabled = true;
    if (micHint) micHint.textContent = "mic ready";
  });

  $("#test-form")?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const fd = new FormData(e.target);
    const f = fd.get("audio");
    if (!f || (f instanceof File && f.size === 0)) {
      alert("Pick a file or record from mic first.");
      return;
    }
    const res = await fetch("/api/test/file", { method: "POST", body: fd });
    if (!res.ok) {
      alert("Test failed: " + (await res.text()));
      return;
    }
    const result = await res.json();
    renderTestResult(result);
  });

  function renderTestResult(r) {
    const out = $("#test-result");
    const triggered = r.triggered;
    out.innerHTML = `
      <div class="metric-tile">
        <div class="name">Triggered</div>
        <div class="value" style="color: ${triggered ? "var(--good)" : "var(--bad)"}">${triggered ? "YES" : "no"}</div>
      </div>
      <div class="metric-tile">
        <div class="name">Max score</div>
        <div class="value">${r.max_score.toFixed(3)}</div>
      </div>
      <div class="metric-tile">
        <div class="name">Mean score</div>
        <div class="value">${r.mean_score.toFixed(3)}</div>
      </div>
      <div class="metric-tile">
        <div class="name">Detections</div>
        <div class="value">${r.detections.length}</div>
      </div>
    `;
    if (r.score_curve.length) {
      const cv = document.createElement("canvas");
      cv.width = 800; cv.height = 120; cv.style.maxWidth = "100%";
      cv.style.gridColumn = "1 / -1";
      out.appendChild(cv);
      const ctx = cv.getContext("2d");
      ctx.fillStyle = "#06090e"; ctx.fillRect(0, 0, cv.width, cv.height);
      const thY = cv.height - r.threshold * cv.height;
      ctx.strokeStyle = "#d29922"; ctx.beginPath();
      ctx.moveTo(0, thY); ctx.lineTo(cv.width, thY); ctx.stroke();
      ctx.strokeStyle = "#58a6ff"; ctx.beginPath();
      const n = r.score_curve.length;
      r.score_curve.forEach((p, i) => {
        const x = (i / Math.max(1, n - 1)) * cv.width;
        const y = cv.height - p.s * cv.height;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }
  }

  // ---------- Initial state sync ----------

  fetch("/api/train/status")
    .then((r) => r.json())
    .then((s) => setRunState(s.status));
})();
