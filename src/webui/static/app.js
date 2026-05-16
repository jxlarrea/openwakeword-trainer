// OpenWakeWord Trainer client-side app.

(function () {
  "use strict";

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  // ---------- Sessions ----------

  const wakeInput = $("input[name='wake_word']");
  const runNameInput = $("#run-name-input");
  const sessionIdInput = $("#session-id-input");
  const sessionSelect = $("#session-select");
  const createSessionBtn = $("#create-session-btn");
  const deleteSessionBtn = $("#delete-session-btn");
  const newSessionName = $("#new-session-name");
  const newSessionWakeWord = $("#new-session-wake-word");
  const sessionSummary = $("#session-summary");
  const configCard = $("#config");
  const testForm = $("#test-form");
  const progressCancelBtn = $("#progress-cancel-btn");
  const progressStatusPill = $("#progress-status-pill");
  const progressCard = $("#progress");
  const progressTitle = $("#progress-title");
  let currentSession = null;
  let runStatus = "idle";
  let hasRunProgress = false;
  let progressRunId = null;

  function slugify(s) {
    return String(s || "")
      .toLowerCase()
      .trim()
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_+|_+$/g, "");
  }

  function fmtBytes(n) {
    if (!Number.isFinite(n) || n <= 0) return "0 B";
    const units = ["B", "KB", "MB", "GB", "TB"];
    let i = 0;
    while (n >= 1024 && i < units.length - 1) {
      n /= 1024;
      i += 1;
    }
    return `${n.toFixed(i ? 1 : 0)} ${units[i]}`;
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function setFormEnabled(enabled) {
    const canEdit = enabled && runStatus !== "running";
    if (configCard) configCard.hidden = !enabled;
    configCard?.classList.toggle("disabled-card", !canEdit);
    $$("#train-form input, #train-form select, #train-form textarea, #train-form button").forEach((el) => {
      if (el.id === "cancel-btn") return;
      el.disabled = !canEdit;
    });
    if (wakeInput) wakeInput.readOnly = true;
    if (runNameInput) runNameInput.readOnly = true;
  }

  function setSessionControlsLocked(locked) {
    if (sessionSelect) sessionSelect.disabled = locked;
    if (newSessionName) newSessionName.disabled = locked;
    if (newSessionWakeWord) newSessionWakeWord.disabled = locked;
    if (createSessionBtn) createSessionBtn.disabled = locked;
    if (deleteSessionBtn) deleteSessionBtn.disabled = locked || !currentSession;
  }

  function setTestFormEnabled(enabled) {
    [testForm].forEach((form) => {
      if (!form) return;
      form.querySelectorAll("input, select, textarea, button").forEach((el) => {
        if (!enabled) {
          el.disabled = true;
          return;
        }
        if (el.id === "stop-record-btn") {
          el.disabled = true;
          return;
        }
        if (el.id === "record-btn" && !(window.isSecureContext || location.hostname === "localhost" || location.hostname === "127.0.0.1")) {
          el.disabled = true;
          return;
        }
        el.disabled = false;
      });
    });
  }

  function setSystemControlsEnabled(enabled) {
    $$("#system-page button, #system-session-rows button").forEach((el) => {
      el.disabled = !enabled;
    });
  }

  function setChecked(form, name, value) {
    const el = form.querySelector(`input[name="${name}"]`);
    if (el) el.checked = Boolean(value);
  }

  function setValue(form, name, value) {
    const el = form.querySelector(`[name="${name}"]`);
    if (el && value !== undefined && value !== null) el.value = value;
  }

  const TRAINING_PROFILES = {
    standard: {
      summary:
        "Standard uses an official/openWakeWord-style 64x3 recipe with moderate false-positive pressure and 3 augmentation rounds.",
      positiveBudget: 200000,
      negativeBudget: 200000,
      augmentationRounds: 3,
      trainingSteps: 115000,
      fpPenalty: 250,
      layerSize: 64,
      nnDepth: 3,
      values: {
        positive_sample_budget: 200000,
        n_positive_per_phrase_per_voice: 8,
        n_kokoro_positive_per_phrase_per_voice: 2,
        augmentations_per_clip: 3,
        use_tablet_far_field_augmentation: false,
        tablet_far_field_probability: 0,
        rir_probability: 0.9,
        background_noise_probability: 0.75,
        max_steps: 115000,
        early_stop_min_steps: 30000,
        max_negative_loss_weight: 250,
        layer_dim: 64,
        n_blocks: 3,
      },
    },
    tablet: {
      summary:
        "Wall-mounted tablet defaults: the v24 recipe tuned for off-axis speech, room echo, TV/media playback, and low false positives.",
      positiveBudget: 200000,
      negativeBudget: 240000,
      augmentationRounds: 6,
      trainingSteps: 115000,
      fpPenalty: 1000,
      layerSize: 64,
      nnDepth: 3,
      values: {
        positive_sample_budget: 200000,
        n_positive_per_phrase_per_voice: 10,
        n_negative_per_phrase_per_voice: 5,
        n_adversarial_phrases: 8000,
        n_adversarial_per_phrase_per_voice: 1,
        use_kokoro: true,
        n_kokoro_positive_per_phrase_per_voice: 3,
        use_kokoro_for_negatives: true,
        n_kokoro_negative_per_phrase_per_voice: 1,
        kokoro_speed_min: 0.9,
        kokoro_speed_max: 1.1,
        augmentations_per_clip: 6,
        use_tablet_far_field_augmentation: true,
        tablet_far_field_probability: 0.75,
        rir_probability: 0.9,
        background_noise_probability: 0.75,
        use_mit_rirs: true,
        use_but_reverbdb: true,
        use_musan_noise: true,
        use_musan_music: true,
        use_fsd50k: true,
        use_common_voice_negatives: true,
        use_openwakeword_negative_features: true,
        use_openwakeword_validation_features: true,
        common_voice_subset: 100000,
        model_type: "dnn",
        max_steps: 115000,
        early_stop_min_steps: 30000,
        early_stop_patience: 40,
        val_every_n_steps: 500,
        learning_rate: 0.0001,
        weight_decay: 0.01,
        use_focal_loss: true,
        focal_gamma: 2,
        label_smoothing: 0.05,
        mixup_alpha: 0.2,
        batch_size: 2048,
        positive_sample_fraction: 0.08,
        negative_loss_weight: 1,
        max_negative_loss_weight: 1000,
        hard_negative_loss_weight: 1,
        hard_negative_threshold: 0.9,
        hard_negative_mining_top_k: 50000,
        hard_negative_finetune_steps: 0,
        hard_negative_finetune_positive_fraction: 0.5,
        lr_warmup_fraction: 0.2,
        lr_hold_fraction: 0.33,
        lr_reduce_on_plateau: false,
        layer_dim: 64,
        n_blocks: 3,
        target_false_positives_per_hour: 0.5,
        min_recall_at_target_fp_for_export: 0.7,
        max_calibration_threshold_for_export: 0.8,
        min_recall_at_0_5_for_export: 0.8,
        max_fp_per_hour_at_0_5_for_export: 10,
        min_positive_median_score_for_export: 0.75,
        min_positive_p10_score_for_export: 0.35,
        use_positive_curve_validation: true,
        curve_validation_max_positive_clips: 400,
        min_curve_recall_for_export: 0.65,
        min_curve_median_peak_for_export: 0.78,
        min_curve_p10_peak_for_export: 0.02,
        min_curve_median_frames_for_export: 2,
        min_curve_median_span_ms_for_export: 160,
        min_curve_confirmation_rate_for_export: 0.3,
        use_tablet_curve_validation: true,
        tablet_curve_validation_variants_per_clip: 1,
        min_tablet_curve_recall_for_export: 0.24,
        min_tablet_curve_median_peak_for_export: 0,
        min_tablet_curve_p10_peak_for_export: 0,
        min_tablet_curve_median_frames_for_export: 0,
        min_tablet_curve_median_span_ms_for_export: 0,
        min_tablet_curve_confirmation_rate_for_export: 0.08,
        curve_confirmation_min_gap_ms: 320,
        seed: 4044,
      },
    },
  };

  function setStandardLayer(size) {
    const radio = document.querySelector(`input[name="std_layer_size"][value="${size}"]`);
    if (radio) radio.checked = true;
  }

  function getStandardLayer() {
    return Number(document.querySelector('input[name="std_layer_size"]:checked')?.value || 64);
  }

  function selectedPiperVoiceCount(form) {
    const count = form.querySelectorAll('input[name="piper_voice"]:checked').length;
    return count || 30;
  }

  function setCheckbox(form, name, value) {
    const el = form.querySelector(`input[name="${name}"]`);
    if (el) el.checked = Boolean(value);
  }

  function sampleBudgetToAdversarialPhrases(form, sampleBudget) {
    const voices = selectedPiperVoiceCount(form);
    return Math.max(100, Math.round(Number(sampleBudget || 0) / voices));
  }

  function setTrainingMode(mode) {
    const next = mode === "advanced" ? "advanced" : "standard";
    document.body.classList.toggle("training-mode-standard", next === "standard");
    document.body.classList.toggle("training-mode-advanced", next === "advanced");
    const select = $("#training-mode-select");
    if (select && select.value !== next) select.value = next;
  }

  function applyTrainingProfile(profileName) {
    const form = $("#train-form");
    const profile = TRAINING_PROFILES[profileName] || TRAINING_PROFILES.tablet;
    if (!form) return;

    $("#profile-summary") && ($("#profile-summary").textContent = profile.summary);
    const profileSelect = $("#profile-select");
    if (profileSelect && profileSelect.value !== profileName) profileSelect.value = profileName;

    $("#std-positive-budget") && ($("#std-positive-budget").value = profile.positiveBudget);
    $("#std-negative-budget") && ($("#std-negative-budget").value = profile.negativeBudget);
    $("#std-augmentation-rounds") && ($("#std-augmentation-rounds").value = profile.augmentationRounds);
    $("#std-training-steps") && ($("#std-training-steps").value = profile.trainingSteps);
    $("#std-fp-penalty") && ($("#std-fp-penalty").value = profile.fpPenalty);
    $("#std-nn-depth") && ($("#std-nn-depth").value = profile.nnDepth);
    setStandardLayer(profile.layerSize);

    Object.entries(profile.values).forEach(([name, value]) => {
      if (typeof value === "boolean") setCheckbox(form, name, value);
      else setValue(form, name, value);
    });
    setValue(
      form,
      "n_adversarial_phrases",
      sampleBudgetToAdversarialPhrases(form, profile.negativeBudget)
    );
    setValue(form, "n_adversarial_per_phrase_per_voice", 1);
  }

  function applyStandardControlsToForm() {
    const form = $("#train-form");
    if (!form || $("#training-mode-select")?.value === "advanced") return;
    const positiveBudget = Number($("#std-positive-budget")?.value || 200000);
    const negativeBudget = Number($("#std-negative-budget")?.value || 240000);
    const augmentationRounds = Number($("#std-augmentation-rounds")?.value || 6);
    const trainingSteps = Number($("#std-training-steps")?.value || 115000);
    const fpPenalty = Number($("#std-fp-penalty")?.value || 1000);
    const layerSize = getStandardLayer();
    const nnDepth = Number($("#std-nn-depth")?.value || 3);

    setValue(form, "positive_sample_budget", positiveBudget);
    setValue(form, "n_adversarial_phrases", sampleBudgetToAdversarialPhrases(form, negativeBudget));
    setValue(form, "n_adversarial_per_phrase_per_voice", 1);
    setValue(form, "augmentations_per_clip", augmentationRounds);
    setValue(form, "max_steps", trainingSteps);
    setValue(form, "max_negative_loss_weight", fpPenalty);
    setValue(form, "layer_dim", layerSize);
    setValue(form, "n_blocks", nnDepth);
  }

  function syncStandardControlsFromForm(form) {
    if (!form) return;
    const profile = form.querySelector('[name="use_tablet_far_field_augmentation"]')?.checked
      ? "tablet"
      : "standard";
    const profileSelect = $("#profile-select");
    if (profileSelect) profileSelect.value = profile;
    $("#profile-summary") && ($("#profile-summary").textContent = TRAINING_PROFILES[profile].summary);
    const adv = Number(form.querySelector('[name="n_adversarial_phrases"]')?.value || 0);
    const positiveBudget = Number(form.querySelector('[name="positive_sample_budget"]')?.value || 0);
    $("#std-positive-budget") && ($("#std-positive-budget").value = positiveBudget || TRAINING_PROFILES[profile].positiveBudget);
    $("#std-negative-budget") && ($("#std-negative-budget").value = adv * selectedPiperVoiceCount(form));
    $("#std-augmentation-rounds") && ($("#std-augmentation-rounds").value = form.querySelector('[name="augmentations_per_clip"]')?.value || 6);
    $("#std-training-steps") && ($("#std-training-steps").value = form.querySelector('[name="max_steps"]')?.value || 115000);
    $("#std-fp-penalty") && ($("#std-fp-penalty").value = form.querySelector('[name="max_negative_loss_weight"]')?.value || 1000);
    $("#std-nn-depth") && ($("#std-nn-depth").value = form.querySelector('[name="n_blocks"]')?.value || 3);
    setStandardLayer(form.querySelector('[name="layer_dim"]')?.value || 64);
  }

  function fillSessionForm(session) {
    const form = $("#train-form");
    if (!form || !session?.config) return;
    const cfg = session.config;
    currentSession = session;
    setFormEnabled(true);
    setRunState(runStatus);
    if (sessionIdInput) sessionIdInput.value = session.id;
    if (wakeInput) wakeInput.value = cfg.wake_word || session.wake_word || "";
    if (runNameInput) runNameInput.value = session.id;

    const gen = cfg.generation || {};
    const aug = cfg.augmentation || {};
    const ds = cfg.datasets || {};
    const tr = cfg.training || {};
    setValue(form, "positive_phrases", (gen.positive_phrases || []).join("\n"));
    setValue(form, "negative_phrases", (gen.negative_phrases || []).join("\n"));
    setValue(form, "positive_sample_budget", gen.positive_sample_budget);
    setValue(form, "n_positive_per_phrase_per_voice", gen.n_positive_per_phrase_per_voice);
    setValue(form, "n_negative_per_phrase_per_voice", gen.n_negative_per_phrase_per_voice);
    setValue(form, "n_adversarial_phrases", gen.n_adversarial_phrases);
    setValue(form, "n_adversarial_per_phrase_per_voice", gen.n_adversarial_per_phrase_per_voice);

    const savedPiperVoices = gen.piper_voices || [];
    $$("#config input[name='piper_voice']").forEach((cb) => {
      cb.checked =
        savedPiperVoices.length === 0 ||
        savedPiperVoices.some((v) => (v.voice_key || v) === cb.value);
    });
    setChecked(form, "use_kokoro", gen.use_kokoro !== false);
    const savedKokoroVoices = gen.kokoro_voices || [];
    $$("#config input[name='kokoro_voice']").forEach((cb) => {
      cb.checked =
        savedKokoroVoices.length === 0 ||
        savedKokoroVoices.includes(cb.value);
    });
    setValue(form, "n_kokoro_positive_per_phrase_per_voice", gen.n_kokoro_positive_per_phrase_per_voice);
    setChecked(form, "use_kokoro_for_negatives", gen.use_kokoro_for_negatives);
    setValue(form, "n_kokoro_negative_per_phrase_per_voice", gen.n_kokoro_negative_per_phrase_per_voice);
    setValue(form, "kokoro_speed_min", gen.kokoro_speed_min);
    setValue(form, "kokoro_speed_max", gen.kokoro_speed_max);

    setValue(form, "augmentations_per_clip", aug.augmentations_per_clip);
    setValue(form, "rir_probability", aug.rir_probability);
    setValue(form, "background_noise_probability", aug.background_noise_probability);
    setChecked(form, "use_tablet_far_field_augmentation", aug.use_tablet_far_field_augmentation !== false);
    setValue(form, "tablet_far_field_probability", aug.tablet_far_field_probability);

    setChecked(form, "use_mit_rirs", ds.use_mit_rirs !== false);
    setChecked(form, "use_but_reverbdb", ds.use_but_reverbdb !== false);
    setChecked(form, "use_musan_noise", ds.use_musan_noise !== false);
    setChecked(form, "use_musan_music", ds.use_musan_music !== false);
    setChecked(form, "use_fsd50k", ds.use_fsd50k !== false);
    setChecked(form, "use_common_voice_negatives", ds.use_common_voice_negatives !== false);
    setChecked(form, "use_openwakeword_negative_features", ds.use_openwakeword_negative_features !== false);
    setChecked(form, "use_openwakeword_validation_features", ds.use_openwakeword_validation_features !== false);
    setValue(form, "common_voice_subset", ds.common_voice_subset);

    setValue(form, "model_type", tr.model_type);
    setValue(form, "layer_dim", tr.layer_dim);
    setValue(form, "n_blocks", tr.n_blocks);
    setValue(form, "learning_rate", tr.learning_rate);
    setValue(form, "weight_decay", tr.weight_decay);
    setChecked(form, "use_focal_loss", tr.use_focal_loss !== false);
    setValue(form, "focal_gamma", tr.focal_gamma);
    setValue(form, "label_smoothing", tr.label_smoothing);
    setValue(form, "mixup_alpha", tr.mixup_alpha);
    setValue(form, "max_negative_loss_weight", tr.max_negative_loss_weight);
    setValue(form, "lr_warmup_fraction", tr.lr_warmup_fraction);
    setValue(form, "lr_hold_fraction", tr.lr_hold_fraction);
    setChecked(form, "lr_reduce_on_plateau", tr.lr_reduce_on_plateau === true);
    setValue(form, "batch_size", tr.batch_size);
    setValue(form, "positive_sample_fraction", tr.positive_sample_fraction);
    setValue(form, "negative_loss_weight", tr.negative_loss_weight);
    setValue(form, "hard_negative_loss_weight", tr.hard_negative_loss_weight);
    setValue(form, "hard_negative_threshold", tr.hard_negative_threshold);
    setValue(form, "hard_negative_mining_top_k", tr.hard_negative_mining_top_k);
    setValue(form, "hard_negative_finetune_steps", tr.hard_negative_finetune_steps);
    setValue(form, "hard_negative_finetune_positive_fraction", tr.hard_negative_finetune_positive_fraction);
    setValue(form, "max_steps", tr.max_steps);
    setValue(form, "val_every_n_steps", tr.val_every_n_steps);
    setValue(form, "early_stop_patience", tr.early_stop_patience);
    setValue(form, "early_stop_min_steps", tr.early_stop_min_steps);
    setValue(form, "target_false_positives_per_hour", tr.target_false_positives_per_hour);
    setValue(form, "min_recall_at_target_fp_for_export", tr.min_recall_at_target_fp_for_export);
    setValue(form, "max_calibration_threshold_for_export", tr.max_calibration_threshold_for_export);
    setValue(form, "min_recall_at_0_5_for_export", tr.min_recall_at_0_5_for_export);
    setValue(form, "max_fp_per_hour_at_0_5_for_export", tr.max_fp_per_hour_at_0_5_for_export);
    setValue(form, "min_positive_median_score_for_export", tr.min_positive_median_score_for_export);
    setValue(form, "min_positive_p10_score_for_export", tr.min_positive_p10_score_for_export);
    setChecked(form, "use_positive_curve_validation", tr.use_positive_curve_validation !== false);
    setValue(form, "curve_validation_max_positive_clips", tr.curve_validation_max_positive_clips);
    setValue(form, "min_curve_recall_for_export", tr.min_curve_recall_for_export);
    setValue(form, "min_curve_median_peak_for_export", tr.min_curve_median_peak_for_export);
    setValue(form, "min_curve_p10_peak_for_export", tr.min_curve_p10_peak_for_export);
    setValue(form, "min_curve_median_frames_for_export", tr.min_curve_median_frames_for_export);
    setValue(form, "min_curve_median_span_ms_for_export", tr.min_curve_median_span_ms_for_export);
    setValue(form, "min_curve_confirmation_rate_for_export", tr.min_curve_confirmation_rate_for_export);
    setChecked(form, "use_tablet_curve_validation", tr.use_tablet_curve_validation !== false);
    setValue(form, "tablet_curve_validation_variants_per_clip", tr.tablet_curve_validation_variants_per_clip);
    setValue(form, "min_tablet_curve_recall_for_export", tr.min_tablet_curve_recall_for_export);
    setValue(form, "min_tablet_curve_median_peak_for_export", tr.min_tablet_curve_median_peak_for_export);
    setValue(form, "min_tablet_curve_p10_peak_for_export", tr.min_tablet_curve_p10_peak_for_export);
    setValue(form, "min_tablet_curve_median_frames_for_export", tr.min_tablet_curve_median_frames_for_export);
    setValue(form, "min_tablet_curve_median_span_ms_for_export", tr.min_tablet_curve_median_span_ms_for_export);
    setValue(form, "min_tablet_curve_confirmation_rate_for_export", tr.min_tablet_curve_confirmation_rate_for_export);
    setValue(form, "curve_confirmation_min_gap_ms", tr.curve_confirmation_min_gap_ms);
    setValue(form, "seed", tr.seed);
    syncStandardControlsFromForm(form);
    setTrainingMode($("#training-mode-select")?.value || "standard");

    if (deleteSessionBtn) deleteSessionBtn.disabled = runStatus === "running";
    if (sessionSummary) {
      sessionSummary.textContent =
        `${session.id} - wake word "${session.wake_word}" - cache ${fmtBytes(session.size_bytes || 0)}` +
        (session.has_model ? " - model ready" : "");
    }
  }

  async function refreshSessions(selectId) {
    const res = await fetch("/api/sessions");
    const sessions = await res.json();
    if (!sessionSelect) return sessions;
    sessionSelect.innerHTML = `<option value="">Create or select a session...</option>`;
    sessions.forEach((s) => {
      const opt = document.createElement("option");
      opt.value = s.id;
      opt.textContent = `${s.id} - ${s.wake_word}${s.has_model ? " - model ready" : ""}`;
      sessionSelect.appendChild(opt);
    });
    if (selectId) {
      sessionSelect.value = selectId;
      await loadSession(selectId);
    }
    return sessions;
  }

  async function loadSession(id) {
    if (!id) {
      currentSession = null;
      setFormEnabled(false);
      setRunState(runStatus);
      setSessionControlsLocked(runStatus === "running");
      if (sessionSummary) sessionSummary.textContent = "Select or create a session to configure training.";
      return;
    }
    const res = await fetch(`/api/sessions/${encodeURIComponent(id)}`);
    if (!res.ok) return;
    fillSessionForm(await res.json());
  }

  sessionSelect?.addEventListener("change", (e) => loadSession(e.target.value));
  createSessionBtn?.addEventListener("click", async () => {
    const sessionName = newSessionName?.value?.trim();
    const wake = newSessionWakeWord?.value?.trim();
    if (!sessionName) {
      newSessionName?.focus();
      return;
    }
    if (!wake) {
      newSessionWakeWord?.focus();
      return;
    }
    const res = await fetch("/api/sessions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionName, wake_word: wake }),
    });
    if (!res.ok) {
      alert(await res.text());
      return;
    }
    const session = await res.json();
    if (newSessionName) newSessionName.value = "";
    if (newSessionWakeWord) newSessionWakeWord.value = "";
    await refreshSessions(session.id);
  });
  deleteSessionBtn?.addEventListener("click", async () => {
    if (!currentSession) return;
    const ok = confirm(
      `Delete session "${currentSession.id}" for wake word "${currentSession.wake_word}" and all cached WAVs/features/checkpoints?`
    );
    if (!ok) return;
    const res = await fetch(`/api/sessions/${encodeURIComponent(currentSession.id)}`, {
      method: "DELETE",
    });
    if (!res.ok) {
      alert(await res.text());
      return;
    }
    currentSession = null;
    await refreshSessions();
    await loadSession("");
    refreshModels();
  });
  setFormEnabled(false);
  const trainForm = $("#train-form");
  if (trainForm) trainForm.noValidate = true;

  $("#training-mode-select")?.addEventListener("change", (e) => {
    setTrainingMode(e.target.value);
    applyStandardControlsToForm();
  });
  $("#profile-select")?.addEventListener("change", (e) => {
    applyTrainingProfile(e.target.value);
  });
  [
    "#std-positive-budget",
    "#std-negative-budget",
    "#std-augmentation-rounds",
    "#std-training-steps",
    "#std-fp-penalty",
    "#std-nn-depth",
  ].forEach((sel) => {
    $(sel)?.addEventListener("input", applyStandardControlsToForm);
  });
  $$('input[name="std_layer_size"]').forEach((el) => {
    el.addEventListener("change", applyStandardControlsToForm);
  });
  setTrainingMode($("#training-mode-select")?.value || "standard");
  applyTrainingProfile($("#profile-select")?.value || "tablet");

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
  $("#select-all-kokoro")?.addEventListener("click", () => {
    $$("#config input[name='kokoro_voice']").forEach((cb) => (cb.checked = true));
  });
  $("#select-none-kokoro")?.addEventListener("click", () => {
    $$("#config input[name='kokoro_voice']").forEach((cb) => (cb.checked = false));
  });
  $("#select-best-kokoro")?.addEventListener("click", () => {
    const best = new Set(["A", "A-", "B-", "C+"]);
    $$("#config .voice-row input[name='kokoro_voice']").forEach((cb) => {
      const q = cb.closest(".voice-row")?.dataset.quality || "";
      cb.checked = best.has(q);
    });
  });
  $("#voice-filter")?.addEventListener("input", (e) => {
    const q = e.target.value.toLowerCase().trim();
    $$("#config .voice-row").forEach((row) => {
      row.classList.toggle("hidden", q && !row.dataset.key.toLowerCase().includes(q));
    });
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

    // Native HTML5 checks are only used in Advanced mode. Standard mode uses
    // explicit range checks below so hidden exact-knob fields and browser
    // step/base arithmetic cannot block production-scale budgets.
    const standardMode = $("#training-mode-select")?.value !== "advanced";
    if (!standardMode) {
      const invalidControls = Array.from(form.elements).filter((el) => {
        if (!(el instanceof HTMLInputElement || el instanceof HTMLSelectElement || el instanceof HTMLTextAreaElement)) {
          return false;
        }
        if (el.disabled || el.type === "hidden") return false;
        return !el.checkValidity();
      });
      if (invalidControls.length) {
        firstInvalid = invalidControls[0];
        firstInvalid.reportValidity();
        return { errors: ["Some required fields are missing or out of range."], firstInvalid };
      }
    } else {
      const numberRules = [
        ["#std-positive-budget", "Positive sample budget", 1000, 300000],
        ["#std-negative-budget", "Negative sample budget", 1000, 300000],
        ["#std-augmentation-rounds", "Augmentation rounds", 0, 10],
        ["#std-training-steps", "Training steps", 20000, 300000],
        ["#std-fp-penalty", "False-positive penalty", 50, 2000],
        ["#std-nn-depth", "NN depth", 1, 5],
      ];
      numberRules.forEach(([selector, label, min, max]) => {
        const el = $(selector);
        const value = Number(el?.value);
        if (!el || !Number.isFinite(value) || value < min || value > max) {
          errors.push(`${label} must be between ${min} and ${max}.`);
          if (!firstInvalid) firstInvalid = el;
        }
      });
    }

    // Custom: at least one voice source.
    const piperCount = form.querySelectorAll('input[name="piper_voice"]:checked').length;
    const kokoroUsed = form.querySelector('input[name="use_kokoro"]')?.checked;
    const kokoroCount = form.querySelectorAll('input[name="kokoro_voice"]:checked').length;

    if (piperCount === 0 && !(kokoroUsed && kokoroCount > 0)) {
      errors.push("Select at least one Piper or Kokoro voice.");
      const voicesFieldset = form.querySelector(".voice-list")?.closest("fieldset");
      voicesFieldset?.classList.add("invalid");
      if (!firstInvalid) firstInvalid = voicesFieldset;
    }

    // Custom: at least one augmentation corpus must be on.
    const corpusBoxes = [
      "use_mit_rirs",
      "use_but_reverbdb",
      "use_musan_noise",
      "use_musan_music",
      "use_fsd50k",
      "use_common_voice_negatives",
      "use_openwakeword_negative_features",
      "use_openwakeword_validation_features",
    ];
    const anyCorpus = corpusBoxes.some(
      (name) => form.querySelector(`input[name="${name}"]`)?.checked
    );
    if (!anyCorpus) {
      errors.push(
        "Enable at least one augmentation/negative dataset."
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
    applyStandardControlsToForm();

    const { errors, firstInvalid } = validateTrainForm(form);
    if (errors.length) {
      showValidationErrors(form, errors, firstInvalid);
      return;
    }

    const fd = new FormData(form);
    const payload = formToPayload(fd);
    if (!payload.session_id) {
      showValidationErrors(form, ["Select or create a session first."], sessionSelect);
      return;
    }
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
    resetProgressUi(data.run_id || payload.session_id || null);
    hasRunProgress = true;
    setRunState(data.status);
    if (data.status === "running") scrollToTop();
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
    await requestCancel();
  });
  progressCancelBtn?.addEventListener("click", async () => {
    await requestCancel();
  });

  async function requestCancel() {
    await fetch("/api/train/cancel", { method: "POST" });
  }

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
    const kokoroVoices = fd.getAll("kokoro_voice");
    return {
      wake_word: String(v("wake_word")).trim(),
      session_id: String(v("session_id")).trim(),
      run_name: String(v("run_name")).trim(),
      generation: {
        positive_phrases: String(v("positive_phrases"))
          .split("\n")
          .map((s) => s.trim())
          .filter(Boolean),
        positive_sample_budget: vNum("positive_sample_budget", 200000),
        n_positive_per_phrase_per_voice: vNum("n_positive_per_phrase_per_voice", 10),
        negative_phrases: String(v("negative_phrases"))
          .split("\n")
          .map((s) => s.trim())
          .filter(Boolean),
        n_negative_per_phrase_per_voice: vNum("n_negative_per_phrase_per_voice", 5),
        n_adversarial_phrases: vNum("n_adversarial_phrases", 8000),
        n_adversarial_per_phrase_per_voice: vNum("n_adversarial_per_phrase_per_voice", 1),
        piper_voices: piperVoices,
        use_kokoro: vBool("use_kokoro"),
        kokoro_voices: kokoroVoices,
        n_kokoro_positive_per_phrase_per_voice: vNum("n_kokoro_positive_per_phrase_per_voice", 3),
        use_kokoro_for_negatives: vBool("use_kokoro_for_negatives"),
        n_kokoro_negative_per_phrase_per_voice: vNum("n_kokoro_negative_per_phrase_per_voice", 1),
        kokoro_speed_min: vNum("kokoro_speed_min", 0.9),
        kokoro_speed_max: vNum("kokoro_speed_max", 1.1),
      },
      augmentation: {
        rir_probability: vNum("rir_probability", 0.9),
        background_noise_probability: vNum("background_noise_probability", 0.75),
        use_tablet_far_field_augmentation: vBool("use_tablet_far_field_augmentation"),
        tablet_far_field_probability: vNum("tablet_far_field_probability", 0.75),
        augmentations_per_clip: vNum("augmentations_per_clip", 6),
      },
      datasets: {
        use_mit_rirs: vBool("use_mit_rirs"),
        use_but_reverbdb: vBool("use_but_reverbdb"),
        use_musan_noise: vBool("use_musan_noise"),
        use_musan_music: vBool("use_musan_music"),
        use_fsd50k: vBool("use_fsd50k"),
        use_common_voice_negatives: vBool("use_common_voice_negatives"),
        use_openwakeword_negative_features: vBool("use_openwakeword_negative_features"),
        use_openwakeword_validation_features: vBool("use_openwakeword_validation_features"),
        common_voice_subset: vNum("common_voice_subset", 100000),
      },
      training: {
        model_type: v("model_type") || "dnn",
        layer_dim: vNum("layer_dim", 64),
        n_blocks: vNum("n_blocks", 3),
        learning_rate: vNum("learning_rate", 0.0001),
        weight_decay: vNum("weight_decay", 0.01),
        use_focal_loss: vBool("use_focal_loss"),
        focal_gamma: vNum("focal_gamma", 2),
        label_smoothing: vNum("label_smoothing", 0.05),
        mixup_alpha: vNum("mixup_alpha", 0.2),
        max_negative_loss_weight: vNum("max_negative_loss_weight", 1000),
        lr_warmup_fraction: vNum("lr_warmup_fraction", 0.2),
        lr_hold_fraction: vNum("lr_hold_fraction", 0.33),
        lr_reduce_on_plateau: vBool("lr_reduce_on_plateau"),
        batch_size: vNum("batch_size", 2048),
        positive_sample_fraction: vNum("positive_sample_fraction", 0.08),
        negative_loss_weight: vNum("negative_loss_weight", 1),
        hard_negative_loss_weight: vNum("hard_negative_loss_weight", 1),
        hard_negative_threshold: vNum("hard_negative_threshold", 0.9),
        hard_negative_mining_top_k: vNum("hard_negative_mining_top_k", 50000),
        hard_negative_finetune_steps: vNum("hard_negative_finetune_steps", 0),
        hard_negative_finetune_positive_fraction: vNum("hard_negative_finetune_positive_fraction", 0.5),
        max_steps: vNum("max_steps", 115000),
        val_every_n_steps: vNum("val_every_n_steps", 500),
        early_stop_patience: vNum("early_stop_patience", 40),
        early_stop_min_steps: vNum("early_stop_min_steps", 30000),
        target_false_positives_per_hour: vNum("target_false_positives_per_hour", 0.5),
        min_recall_at_target_fp_for_export: vNum("min_recall_at_target_fp_for_export", 0.7),
        max_calibration_threshold_for_export: vNum("max_calibration_threshold_for_export", 0.8),
        min_recall_at_0_5_for_export: vNum("min_recall_at_0_5_for_export", 0.8),
        max_fp_per_hour_at_0_5_for_export: vNum("max_fp_per_hour_at_0_5_for_export", 10),
        min_positive_median_score_for_export: vNum("min_positive_median_score_for_export", 0.75),
        min_positive_p10_score_for_export: vNum("min_positive_p10_score_for_export", 0.35),
        use_positive_curve_validation: vBool("use_positive_curve_validation"),
        curve_validation_max_positive_clips: vNum("curve_validation_max_positive_clips", 400),
        min_curve_recall_for_export: vNum("min_curve_recall_for_export", 0.65),
        min_curve_median_peak_for_export: vNum("min_curve_median_peak_for_export", 0.78),
        min_curve_p10_peak_for_export: vNum("min_curve_p10_peak_for_export", 0.02),
        min_curve_median_frames_for_export: vNum("min_curve_median_frames_for_export", 2),
        min_curve_median_span_ms_for_export: vNum("min_curve_median_span_ms_for_export", 160),
        min_curve_confirmation_rate_for_export: vNum("min_curve_confirmation_rate_for_export", 0.3),
        use_tablet_curve_validation: vBool("use_tablet_curve_validation"),
        tablet_curve_validation_variants_per_clip: vNum("tablet_curve_validation_variants_per_clip", 1),
        min_tablet_curve_recall_for_export: vNum("min_tablet_curve_recall_for_export", 0.24),
        min_tablet_curve_median_peak_for_export: vNum("min_tablet_curve_median_peak_for_export", 0),
        min_tablet_curve_p10_peak_for_export: vNum("min_tablet_curve_p10_peak_for_export", 0),
        min_tablet_curve_median_frames_for_export: vNum("min_tablet_curve_median_frames_for_export", 0),
        min_tablet_curve_median_span_ms_for_export: vNum("min_tablet_curve_median_span_ms_for_export", 0),
        min_tablet_curve_confirmation_rate_for_export: vNum("min_tablet_curve_confirmation_rate_for_export", 0.08),
        curve_confirmation_min_gap_ms: vNum("curve_confirmation_min_gap_ms", 320),
        seed: vNum("seed", 4044),
      },
    };
  }

  // ---------- Start/Cancel button state machine ----------

  function setRunState(state) {
    const pill = $("#status-pill");
    const startBtn = $("#start-btn");
    const cancelBtn = $("#cancel-btn");
    runStatus = state || "idle";
    [pill, progressStatusPill].forEach((el) => {
      if (!el) return;
      el.textContent = runStatus;
      el.className = "pill " + runStatus;
    });

    const isRunning = runStatus === "running";
    const progressBelongsToCurrentSession =
      !progressRunId || (currentSession && currentSession.id === progressRunId);
    const showProgress =
      isRunning ||
      ((hasRunProgress || !["idle", ""].includes(runStatus)) && progressBelongsToCurrentSession);
    if (progressCard) progressCard.hidden = !showProgress;
    if (progressTitle) progressTitle.textContent = progressRunId ? `Progress: ${progressRunId}` : "Progress";
    document.body.classList.toggle("run-active", showProgress);
    if (startBtn) startBtn.style.display = isRunning ? "none" : "";
    if (cancelBtn) cancelBtn.style.display = isRunning ? "" : "none";
    if (progressCancelBtn) progressCancelBtn.style.display = isRunning ? "" : "none";
    setSessionControlsLocked(isRunning);
    setFormEnabled(Boolean(currentSession));
    setTestFormEnabled(!isRunning);
    setSystemControlsEnabled(!isRunning);
    if (isRunning && phaseBanner && phaseBanner.textContent === "Idle") {
      phaseBanner.textContent = "training is running";
    }
  }

  function scrollToTop() {
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  // ---------- SSE: live progress ----------

  const phaseBanner = $("#phase-banner");
  const progressBars = $("#progress-bars");
  const metricsBox = $("#metrics");
  const logEl = $("#log");
  const progressMap = new Map();
  const metricMap = new Map();
  const seenLogs = new Set();

  function resetProgressUi(runId = null) {
    progressRunId = runId;
    hasRunProgress = Boolean(runId);
    progressMap.clear();
    metricMap.clear();
    seenLogs.clear();
    if (progressBars) progressBars.innerHTML = "";
    if (metricsBox) metricsBox.innerHTML = "";
    if (logEl) logEl.innerHTML = "";
    if (phaseBanner) phaseBanner.textContent = runId ? "starting..." : "Idle";
    if (progressTitle) progressTitle.textContent = runId ? `Progress: ${runId}` : "Progress";
  }

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
    if (!metricsBox) return;
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

  function appendLog(level, msg, ts) {
    const key = `${ts || ""}|${level || "info"}|${msg}`;
    if (seenLogs.has(key)) return;
    seenLogs.add(key);
    const span = document.createElement("span");
    span.className = "l-" + (level || "info");
    span.textContent = msg + "\n";
    logEl.appendChild(span);
    logEl.scrollTop = logEl.scrollHeight;
  }

  function hydrateProgress(progress) {
    if (!progress) return;
    if (progress.run_id && progressRunId && progress.run_id !== progressRunId) {
      resetProgressUi(progress.run_id);
    }
    progressRunId = progress.run_id || progressRunId;
    hasRunProgress = Boolean(
      progress.run_id ||
      progress.phase ||
      (progress.progress || []).length ||
      Object.keys(progress.metrics || {}).length
    );
    if (progress.phase?.name && phaseBanner) {
      phaseBanner.textContent =
        `phase: ${progress.phase.name}` +
        (progress.phase.detail ? ` -> ${progress.phase.detail}` : "");
    }
    (progress.progress || []).forEach((p) => {
      setProgress(p.name, Number(p.fraction) || 0, p.detail || "");
    });
    Object.entries(progress.metrics || {}).forEach(([k, v]) => {
      setMetric(k, fmt(v));
    });
    (progress.logs || []).forEach((l) => {
      appendLog(l.level, l.message, l.ts);
    });
    setRunState(runStatus);
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

  function fmtPct(v) {
    return typeof v === "number" ? v.toFixed(0) + "%" : "n/a";
  }

  function fmtGb(v) {
    return typeof v === "number" ? v.toFixed(1) + " GB" : "n/a";
  }

  function updateSystemMetrics(d) {
    if ("gpu_name" in d) setMetric("Accelerator", d.gpu_name);
    if ("cpu_percent" in d) setMetric("CPU", fmtPct(d.cpu_percent));
    if ("ram_percent" in d) {
      setMetric(
        "RAM",
        `${fmtPct(d.ram_percent)} (${fmtGb(d.ram_used_gb)} / ${fmtGb(d.ram_total_gb)})`
      );
    }
    if ("gpu_percent" in d) setMetric("GPU", fmtPct(d.gpu_percent));
    if ("gpu_mem_percent" in d) {
      setMetric(
        "GPU VRAM",
        `${fmtPct(d.gpu_mem_percent)} (${fmtGb(d.gpu_mem_used_gb)} / ${fmtGb(d.gpu_mem_total_gb)})`
      );
    } else if ("gpu_mem_note" in d) {
      setMetric("GPU Memory", `${d.gpu_mem_note} (see RAM)`);
    }
    if ("gpu_temp_c" in d) setMetric("GPU Temp", Math.round(d.gpu_temp_c) + " C");
  }

  const es = new EventSource("/api/events");
  es.addEventListener("phase", (e) => {
    const d = JSON.parse(e.data);
    hasRunProgress = true;
    progressRunId = d.run_id || progressRunId;
    if (progressCard) progressCard.hidden = false;
    document.body.classList.add("run-active");
    phaseBanner.textContent = `phase: ${d.name}` + (d.detail ? ` -> ${d.detail}` : "");
  });
  es.addEventListener("progress", (e) => {
    const d = JSON.parse(e.data);
    hasRunProgress = true;
    if (progressCard) progressCard.hidden = false;
    setProgress(d.name, d.fraction, d.detail || "");
  });
  es.addEventListener("metric", (e) => {
    const d = JSON.parse(e.data);
    hasRunProgress = true;
    if (progressCard) progressCard.hidden = false;
    Object.keys(d).forEach((k) => {
      if (k === "kind" || k === "ts") return;
      setMetric(k, fmt(d[k]));
    });
  });
  es.addEventListener("system", (e) => {
    const d = JSON.parse(e.data);
    updateSystemMetrics(d);
  });
  es.addEventListener("log", (e) => {
    const d = JSON.parse(e.data);
    if (progressRunId || hasRunProgress || runStatus === "running") {
      hasRunProgress = true;
      if (progressCard) progressCard.hidden = false;
    }
    appendLog(d.level, d.message, d.ts);
  });
  es.addEventListener("run_started", (e) => {
    const d = JSON.parse(e.data);
    resetProgressUi(d.run_id || null);
    hasRunProgress = true;
    if (progressCard) progressCard.hidden = false;
    appendLog("info", `Run started: ${d.run_id} (${d.wake_word})`, d.ts);
    setRunState("running");
    scrollToTop();
  });
  es.addEventListener("complete", (e) => {
    const d = JSON.parse(e.data);
    appendLog("info", `Complete -> ${d.onnx_path}`, d.ts);
    setRunState("succeeded");
    refreshModels();
  });
  es.addEventListener("run_error", (e) => {
    try {
      const d = JSON.parse(e.data);
      appendLog("error", d.message, d.ts);
      setRunState("failed");
    } catch {}
  });
  es.addEventListener("cancelled", (e) => {
    let ts;
    try { ts = JSON.parse(e.data).ts; } catch {}
    appendLog("warning", "Run cancelled", ts);
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
      .map(
        (m) => `<option value="${m.name}"
          data-size="${m.size || 0}"
          data-package-name="${m.package_name || ""}"
          data-package-size="${m.package_size || ""}">${m.name}</option>`
      )
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
    const packageBtn = $("#download-package-btn");
    if (!opt) {
      info.textContent = "no models yet";
      if (packageBtn) packageBtn.disabled = true;
      return;
    }
    const size = Number(opt.getAttribute("data-size") || 0);
    const kb = (size / 1024).toFixed(1);
    const packageName = opt.getAttribute("data-package-name") || "";
    const packageSize = Number(opt.getAttribute("data-package-size") || 0);
    const packageKb = packageSize ? `, package ${(packageSize / 1024).toFixed(1)} KB` : "";
    info.textContent = `${opt.value} (${kb} KB${packageKb})`;
    if (packageBtn) packageBtn.disabled = !packageName;
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

  $("#download-package-btn")?.addEventListener("click", () => {
    const sel = $("#test-model");
    const opt = sel?.options[sel.selectedIndex];
    const packageName = opt?.getAttribute("data-package-name");
    if (!packageName) {
      alert("No package is available for this model yet.");
      return;
    }
    const a = document.createElement("a");
    a.href = `/api/model-packages/${encodeURIComponent(packageName)}`;
    a.download = packageName;
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
  let liveSocket = null;
  let liveAudioContext = null;
  let liveStream = null;
  let liveSource = null;
  let liveProcessor = null;
  let liveScores = [];

  const isSecure = window.isSecureContext || location.hostname === "localhost" || location.hostname === "127.0.0.1";
  const recordBtn = $("#record-btn");
  const stopBtn = $("#stop-record-btn");
  const liveStartBtn = $("#live-start-btn");
  const liveStopBtn = $("#live-stop-btn");
  const micHint = $("#mic-hint");
  const livePanel = $("#live-panel");
  const liveLogEl = $("#live-log");
  const liveCanvas = $("#live-score-canvas");
  const liveSensitivity = $("#live-sensitivity");
  const testThresholdInput = $("#test-threshold");

  if (!isSecure) {
    if (recordBtn) {
      recordBtn.disabled = true;
      recordBtn.title = "Microphone recording requires HTTPS (or localhost).";
    }
    if (liveStartBtn) {
      liveStartBtn.disabled = true;
      liveStartBtn.title = "Live microphone testing requires HTTPS (or localhost).";
    }
    if (micHint) {
      micHint.textContent = "mic disabled - HTTPS required";
      micHint.classList.add("failed");
    }
  } else if (micHint) {
    micHint.textContent = "mic ready";
  }

  function liveWsUrl() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${location.host}/api/test/live`;
  }

  function appendLiveLog(level, message) {
    if (!liveLogEl) return;
    const ts = new Date().toLocaleTimeString();
    liveLogEl.textContent += `[${ts}] ${level || "info"}: ${message}\n`;
    liveLogEl.scrollTop = liveLogEl.scrollHeight;
  }

  function setLiveMetric(id, value) {
    const el = $(id);
    if (el) el.textContent = value;
  }

  function drawLiveGraph(cutoff) {
    if (!liveCanvas) return;
    const ctx = liveCanvas.getContext("2d");
    const w = liveCanvas.width;
    const h = liveCanvas.height;
    ctx.fillStyle = "#06090e";
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = "#30363d";
    ctx.lineWidth = 1;
    for (let i = 1; i < 4; i += 1) {
      const y = (i / 4) * h;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }
    const thY = h - Math.max(0, Math.min(1, cutoff || 0.5)) * h;
    ctx.strokeStyle = "#d29922";
    ctx.beginPath();
    ctx.moveTo(0, thY);
    ctx.lineTo(w, thY);
    ctx.stroke();
    if (!liveScores.length) return;
    ctx.strokeStyle = "#58a6ff";
    ctx.lineWidth = 2;
    ctx.beginPath();
    liveScores.forEach((p, i) => {
      const x = (i / Math.max(1, liveScores.length - 1)) * w;
      const y = h - Math.max(0, Math.min(1, p.score)) * h;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    const last = liveScores[liveScores.length - 1];
    if (last?.triggered) {
      ctx.fillStyle = "#3fb950";
      ctx.beginPath();
      ctx.arc(w - 12, h - Math.max(0, Math.min(1, last.score)) * h, 5, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function downsampleFloat32(input, inputRate, outputRate) {
    if (inputRate === outputRate) return new Float32Array(input);
    const ratio = inputRate / outputRate;
    const outLen = Math.floor(input.length / ratio);
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i += 1) {
      const start = Math.floor(i * ratio);
      const end = Math.min(input.length, Math.floor((i + 1) * ratio));
      let sum = 0;
      let count = 0;
      for (let j = start; j < end; j += 1) {
        sum += input[j];
        count += 1;
      }
      out[i] = count ? sum / count : input[start] || 0;
    }
    return out;
  }

  function currentLiveConfig() {
    return {
      model_name: $("#test-model")?.value || "",
      threshold: Number(testThresholdInput?.value || 0.5),
      sensitivity: liveSensitivity?.value || "Moderately sensitive",
    };
  }

  function sendLiveConfig() {
    if (liveSocket?.readyState !== WebSocket.OPEN) return;
    liveSocket.send(JSON.stringify({ type: "config", ...currentLiveConfig() }));
  }

  async function stopLiveMic() {
    if (liveProcessor) {
      liveProcessor.disconnect();
      liveProcessor.onaudioprocess = null;
      liveProcessor = null;
    }
    if (liveSource) {
      liveSource.disconnect();
      liveSource = null;
    }
    if (liveStream) {
      liveStream.getTracks().forEach((t) => t.stop());
      liveStream = null;
    }
    if (liveAudioContext) {
      await liveAudioContext.close().catch(() => {});
      liveAudioContext = null;
    }
    if (liveSocket && liveSocket.readyState === WebSocket.OPEN) {
      liveSocket.send(JSON.stringify({ type: "stop" }));
      liveSocket.close();
    }
    liveSocket = null;
    if (liveStartBtn) liveStartBtn.disabled = !isSecure;
    if (liveStopBtn) liveStopBtn.disabled = true;
    if (recordBtn) recordBtn.disabled = !isSecure;
    if (micHint && isSecure) micHint.textContent = "mic ready";
  }

  async function startLiveMic() {
    const cfg = currentLiveConfig();
    if (!cfg.model_name) {
      alert("No model selected.");
      return;
    }
    if (!isSecure || liveSocket) return;
    liveScores = [];
    if (liveLogEl) liveLogEl.textContent = "";
    if (livePanel) livePanel.hidden = false;
    setLiveMetric("#live-score", "0.000");
    setLiveMetric("#live-peak", "0.000");
    setLiveMetric("#live-gate", "starting");
    drawLiveGraph(cfg.threshold);

    liveSocket = new WebSocket(liveWsUrl());
    liveSocket.binaryType = "arraybuffer";
    liveSocket.onopen = async () => {
      liveSocket.send(JSON.stringify(cfg));
      try {
        liveStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            echoCancellation: false,
            noiseSuppression: false,
            autoGainControl: false,
          },
        });
        liveAudioContext = new AudioContext();
        liveSource = liveAudioContext.createMediaStreamSource(liveStream);
        liveProcessor = liveAudioContext.createScriptProcessor(4096, 1, 1);
        liveProcessor.onaudioprocess = (ev) => {
          const input = ev.inputBuffer.getChannelData(0);
          ev.outputBuffer.getChannelData(0).fill(0);
          if (liveSocket?.readyState !== WebSocket.OPEN) return;
          const pcm16k = downsampleFloat32(input, liveAudioContext.sampleRate, 16000);
          if (pcm16k.byteLength) liveSocket.send(pcm16k.buffer);
        };
        liveSource.connect(liveProcessor);
        liveProcessor.connect(liveAudioContext.destination);
        if (liveStartBtn) liveStartBtn.disabled = true;
        if (liveStopBtn) liveStopBtn.disabled = false;
        if (recordBtn) recordBtn.disabled = true;
        if (micHint) micHint.textContent = "live mic running";
        appendLiveLog("info", "live microphone stream started");
      } catch (err) {
        appendLiveLog("error", err?.message || err);
        alert("Live mic failed: " + (err?.message || err));
        stopLiveMic();
      }
    };
    liveSocket.onmessage = (ev) => {
      let msg;
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return;
      }
      if (msg.type === "ready" || msg.type === "config") {
        setLiveMetric("#live-cutoff", Number(msg.cutoff || 0).toFixed(3));
        setLiveMetric("#live-gate", msg.type === "ready" ? "ready" : "updated");
        appendLiveLog("info", msg.message || `cutoff=${Number(msg.cutoff || 0).toFixed(3)}`);
        drawLiveGraph(Number(msg.cutoff || 0.5));
        return;
      }
      if (msg.type === "score") {
        liveScores.push(msg);
        if (liveScores.length > 240) liveScores = liveScores.slice(-240);
        setLiveMetric("#live-score", Number(msg.score || 0).toFixed(3));
        setLiveMetric("#live-peak", Number(msg.peak || 0).toFixed(3));
        setLiveMetric("#live-cutoff", Number(msg.cutoff || 0).toFixed(3));
        setLiveMetric("#live-gate", msg.gate || "-");
        drawLiveGraph(Number(msg.cutoff || 0.5));
        if (msg.triggered) {
          appendLiveLog(
            "detect",
            `triggered score=${Number(msg.score).toFixed(3)} cutoff=${Number(msg.cutoff).toFixed(3)} gate=${msg.gate}`
          );
        } else if (msg.message && msg.gate !== "warming") {
          appendLiveLog("info", msg.message);
        }
        return;
      }
      if (msg.type === "log" || msg.type === "error") {
        appendLiveLog(msg.level || msg.type, msg.message || "");
      }
    };
    liveSocket.onclose = () => {
      appendLiveLog("info", "live microphone stream closed");
      stopLiveMic();
    };
    liveSocket.onerror = () => {
      appendLiveLog("error", "live WebSocket error");
    };
  }

  liveStartBtn?.addEventListener("click", startLiveMic);
  liveStopBtn?.addEventListener("click", stopLiveMic);
  liveSensitivity?.addEventListener("change", sendLiveConfig);
  testThresholdInput?.addEventListener("change", sendLiveConfig);
  $("#test-model")?.addEventListener("change", () => {
    if (liveSocket) stopLiveMic();
  });

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
    if (!out) return;
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

  // ---------- System page ----------

  const systemRows = $("#system-session-rows");
  const systemStatus = $("#system-action-status");
  let systemDiskRequestSeq = 0;

  function setSystemStatus(text, kind) {
    if (!systemStatus) return;
    systemStatus.textContent = text || "";
    systemStatus.className = "pill";
    if (kind) systemStatus.classList.add(kind);
  }

  function renderSystemDisk(disk) {
    if (!disk || !systemRows) return;
    const count = $("#system-session-count");
    const sessionBytes = $("#system-session-bytes");
    const cacheBytes = $("#system-cache-bytes");
    if (count) count.textContent = String((disk.sessions || []).length);
    const hasSizes = disk.sizes_included !== false;
    if (sessionBytes) sessionBytes.textContent = hasSizes ? fmtBytes(disk.session_bytes || 0) : "calculating...";
    if (cacheBytes) cacheBytes.textContent = hasSizes ? fmtBytes(disk.cache_bytes || 0) : "calculating...";
    if (!disk.sessions?.length) {
      systemRows.innerHTML = `<tr><td colspan="5" class="empty-row">No sessions yet.</td></tr>`;
      return;
    }
    systemRows.innerHTML = disk.sessions
      .map(
        (s) => `
          <tr data-session-id="${escapeHtml(s.id)}">
            <td>${escapeHtml(s.wake_word || s.id)}</td>
            <td><code>${escapeHtml(s.id)}</code></td>
            <td>${s.size_bytes == null ? "calculating..." : fmtBytes(s.size_bytes || 0)}</td>
            <td>${s.has_model ? "ready" : "-"}</td>
            <td class="row-actions">
              <button type="button" class="danger delete-session-cache-btn" data-session-id="${escapeHtml(s.id)}">Delete cache</button>
              <button type="button" class="danger delete-system-session-btn" data-session-id="${escapeHtml(s.id)}">Delete session</button>
            </td>
          </tr>
        `
      )
      .join("");
  }

  async function refreshSystemDisk(force = false) {
    if (!systemRows) return;
    const seq = ++systemDiskRequestSeq;
    const quickRes = await fetch("/api/system/disk?sizes=false", { cache: "no-store" });
    if (seq !== systemDiskRequestSeq) return;
    if (quickRes.ok) renderSystemDisk(await quickRes.json());

    const fullUrl = force ? "/api/system/disk?refresh=true&sizes=true" : "/api/system/disk?sizes=true";
    const fullRes = await fetch(fullUrl, { cache: "no-store" });
    if (seq !== systemDiskRequestSeq) return;
    if (fullRes.ok) {
      renderSystemDisk(await fullRes.json());
    } else {
      setSystemStatus("disk metrics unavailable", "failed");
    }
  }

  async function systemDelete(url, confirmText) {
    if (!confirm(confirmText)) return;
    setSystemStatus("working...", "running");
    const res = await fetch(url, { method: "DELETE" });
    if (!res.ok) {
      setSystemStatus("failed", "failed");
      alert(await res.text());
      return;
    }
    const data = await res.json();
    if (data.disk) renderSystemDisk(data.disk);
    setSystemStatus(`reclaimed ${fmtBytes(data.reclaimed_bytes || 0)}`, "succeeded");
    return data;
  }

  $("#delete-all-cache-btn")?.addEventListener("click", () => {
    systemDelete(
      "/api/system/cache",
      "Delete all generated/downloaded cache and trained models while keeping session settings?"
    );
  });

  $("#delete-all-data-btn")?.addEventListener("click", () => {
    systemDelete(
      "/api/system/all",
      "Delete all sessions, saved settings, generated data, downloaded cache, and trained models?"
    );
  });

  systemRows?.addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-session-id]");
    if (!btn) return;
    const id = btn.getAttribute("data-session-id");
    if (btn.classList.contains("delete-session-cache-btn")) {
      systemDelete(
        `/api/system/sessions/${encodeURIComponent(id)}/cache`,
        `Delete generated data and model for session "${id}" but keep its settings?`
      );
    } else if (btn.classList.contains("delete-system-session-btn")) {
      systemDelete(
        `/api/sessions/${encodeURIComponent(id)}`,
        `Delete session "${id}" and all of its cached data?`
      ).then(() => refreshSystemDisk(true));
    }
  });

  if (systemRows) refreshSystemDisk();

  // ---------- Initial state sync ----------

  fetch("/api/train/status")
    .then((r) => r.json())
    .then(async (s) => {
      setRunState(s.status);
      if (s.status === "running" && s.run_id) {
        if (sessionSelect) sessionSelect.value = s.run_id;
        await loadSession(s.run_id);
        if (sessionSummary) {
          sessionSummary.textContent = `${s.run_id} - wake word "${s.wake_word || s.run_id}" - training in progress`;
        }
      }
      progressRunId = s.progress?.run_id || s.run_id || progressRunId;
      hydrateProgress(s.progress);
      if (s.run_id && s.status !== "idle") {
        hasRunProgress = true;
        setRunState(s.status);
      }
      if (s.status === "running" && !s.progress?.phase && phaseBanner) {
        phaseBanner.textContent = `training: ${s.run_id || "active run"}`;
      }
      if (s.system) updateSystemMetrics(s.system);
    });
})();
