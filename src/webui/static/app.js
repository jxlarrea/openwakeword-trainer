// OpenWakeWord Trainer client-side app.
// Handles: form submission, SSE event stream, audition, mic recording, model test.

(function () {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

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

  // ---------- Audition (per voice) ----------

  $$(".audition-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const voiceKey = btn.dataset.voice;
      const text = $("#audition-text")?.value?.trim() || $("input[name='wake_word']")?.value?.trim() || "hey jarvis";
      const speakerId = $("#audition-speaker")?.value;
      btn.disabled = true;
      btn.textContent = "...";
      try {
        const res = await fetch("/api/audition/piper", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text, voice_key: voiceKey, speaker_id: speakerId || null }),
        });
        if (!res.ok) throw new Error(await res.text());
        const blob = await res.blob();
        const player = $("#audition-player");
        player.src = URL.createObjectURL(blob);
        player.play();
      } catch (err) {
        alert("Audition failed: " + err.message);
      } finally {
        btn.disabled = false;
        btn.textContent = "audition";
      }
    });
  });

  // ---------- ElevenLabs voices ----------

  $("#load-elevenlabs-voices")?.addEventListener("click", async () => {
    const container = $("#elevenlabs-voices");
    container.innerHTML = "loading...";
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
          <button type="button" class="audition-el" data-voice="${v.voice_id}">audition</button>
        </label>`
      )
      .join("");
    container.querySelectorAll(".audition-el").forEach((btn) => {
      btn.addEventListener("click", async () => {
        const text = $("#audition-text")?.value?.trim() || "hey jarvis";
        btn.disabled = true;
        btn.textContent = "...";
        try {
          const res = await fetch("/api/audition/elevenlabs", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text, voice_id: btn.dataset.voice }),
          });
          if (!res.ok) throw new Error(await res.text());
          const blob = await res.blob();
          const player = $("#audition-player");
          player.src = URL.createObjectURL(blob);
          player.play();
        } catch (err) {
          alert("Audition failed: " + err.message);
        } finally {
          btn.disabled = false;
          btn.textContent = "audition";
        }
      });
    });
  });

  // ---------- Submit training run ----------

  $("#train-form")?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.target;
    const fd = new FormData(form);
    const payload = formToPayload(fd);
    const res = await fetch("/api/train/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      alert("Could not start: " + (await res.text()));
      return;
    }
    const data = await res.json();
    setStatus(data.status);
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
        augmentations_per_clip: vNum("augmentations_per_clip", 3),
      },
      datasets: {
        use_mit_rirs: vBool("use_mit_rirs"),
        use_musan_noise: vBool("use_musan_noise"),
        use_musan_music: vBool("use_musan_music"),
        use_fsd50k: vBool("use_fsd50k"),
        use_common_voice_negatives: vBool("use_common_voice_negatives"),
        common_voice_subset: vNum("common_voice_subset", 10000),
      },
      training: {
        model_type: v("model_type") || "dnn",
        layer_dim: vNum("layer_dim", 128),
        n_blocks: vNum("n_blocks", 1),
        learning_rate: vNum("learning_rate", 0.0001),
        batch_size: vNum("batch_size", 1024),
        max_steps: vNum("max_steps", 50000),
        val_every_n_steps: vNum("val_every_n_steps", 500),
        early_stop_patience: vNum("early_stop_patience", 5),
        target_false_positives_per_hour: vNum("target_false_positives_per_hour", 0.2),
        seed: vNum("seed", 42),
      },
    };
  }

  // ---------- SSE: live progress ----------

  function setStatus(s) {
    const pill = $("#status-pill");
    if (!pill) return;
    pill.textContent = s;
    pill.className = "pill " + s;
  }

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
    bar.querySelector(".pct").textContent = pct + "%" + (detail ? " (" + detail + ")" : "");
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
    setStatus("running");
  });
  es.addEventListener("complete", (e) => {
    const d = JSON.parse(e.data);
    appendLog("info", `Complete -> ${d.onnx_path}`);
    setStatus("succeeded");
    refreshModels();
  });
  es.addEventListener("run_error", (e) => {
    try {
      const d = JSON.parse(e.data);
      appendLog("error", d.message);
      setStatus("failed");
    } catch {}
  });
  es.onerror = () => {
    // EventSource connection-level error. Browser will auto-reconnect.
    appendLog("warning", "SSE connection lost; retrying...");
  };
  es.addEventListener("cancelled", () => {
    appendLog("warning", "Run cancelled");
    setStatus("cancelled");
  });

  // ---------- Test panel ----------

  async function refreshModels() {
    const res = await fetch("/api/models");
    const models = await res.json();
    const sel = $("#test-model");
    sel.innerHTML = models.map((m) => `<option value="${m.name}">${m.name}</option>`).join("");
  }

  let mediaRecorder = null;
  let recordedChunks = [];

  $("#record-btn")?.addEventListener("click", async () => {
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
      $("#record-btn").disabled = true;
      $("#stop-record-btn").disabled = false;
    } catch (err) {
      alert("Mic access failed: " + err.message);
    }
  });

  $("#stop-record-btn")?.addEventListener("click", () => {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
    }
    $("#record-btn").disabled = false;
    $("#stop-record-btn").disabled = true;
  });

  $("#test-form")?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const fd = new FormData(e.target);
    if (!fd.get("audio") || (fd.get("audio") instanceof File && fd.get("audio").size === 0)) {
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
      out.appendChild(cv);
      const ctx = cv.getContext("2d");
      ctx.fillStyle = "#06090e"; ctx.fillRect(0, 0, cv.width, cv.height);
      // threshold line
      const thY = cv.height - r.threshold * cv.height;
      ctx.strokeStyle = "#d29922"; ctx.beginPath();
      ctx.moveTo(0, thY); ctx.lineTo(cv.width, thY); ctx.stroke();
      // curve
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

  // initial poll for status
  fetch("/api/train/status").then((r) => r.json()).then((s) => setStatus(s.status));
})();
