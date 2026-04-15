const els = {
  gestureText: document.getElementById("gestureText"),
  fpsText: document.getElementById("fpsText"),
  detectionText: document.getElementById("detectionText"),
  eventsList: document.getElementById("eventsList"),
  counts: document.getElementById("counts"),
  resetBtn: document.getElementById("resetBtn"),
  sliders: {
    detection: document.getElementById("detection"),
    tracking: document.getElementById("tracking"),
    hold: document.getElementById("hold"),
    cooldown: document.getElementById("cooldown"),
    alpha: document.getElementById("alpha"),
  },
  values: {
    detection: document.getElementById("detectionValue"),
    tracking: document.getElementById("trackingValue"),
    hold: document.getElementById("holdValue"),
    cooldown: document.getElementById("cooldownValue"),
    alpha: document.getElementById("alphaValue"),
  },
};

const formatNumber = (value, decimals = 2) => Number(value).toFixed(decimals);

let pendingConfigTimer = null;

function renderConfig(config) {
  if (!config) return;
  els.sliders.detection.value = config.min_detection_confidence;
  els.sliders.tracking.value = config.min_tracking_confidence;
  els.sliders.hold.value = config.gesture_hold_frames;
  els.sliders.cooldown.value = config.action_cooldown_ms;
  els.sliders.alpha.value = config.smoothing_alpha;

  els.values.detection.textContent = formatNumber(config.min_detection_confidence);
  els.values.tracking.textContent = formatNumber(config.min_tracking_confidence);
  els.values.hold.textContent = config.gesture_hold_frames;
  els.values.cooldown.textContent = config.action_cooldown_ms;
  els.values.alpha.textContent = formatNumber(config.smoothing_alpha);
}

function renderCounts(counts = {}) {
  const keys = Object.keys(counts);
  if (!keys.length) {
    els.counts.innerHTML = '<span class="count-chip">No gestures recorded yet</span>';
    return;
  }

  els.counts.innerHTML = keys
    .sort((a, b) => counts[b] - counts[a])
    .map((key) => `<span class="count-chip">${key}: ${counts[key]}</span>`)
    .join("");
}

function renderEvents(events = []) {
  if (!events.length) {
    els.eventsList.innerHTML = "<li><span>Waiting for gestures</span><span>-</span></li>";
    return;
  }

  els.eventsList.innerHTML = events
    .slice(0, 8)
    .map((event) => `<li><span>${event.gesture}</span><span>${event.time}</span></li>`)
    .join("");
}

async function fetchStatus() {
  try {
    const res = await fetch("/api/status", { cache: "no-store" });
    if (!res.ok) return;
    const data = await res.json();

    els.gestureText.textContent = data.gesture;
    els.fpsText.textContent = data.fps.toFixed(1);
    els.detectionText.textContent = `${data.hand_detection_rate}%`;

    renderConfig(data.config);
    renderCounts(data.gesture_counts);
    renderEvents(data.recent_events);
  } catch (err) {
    els.gestureText.textContent = "Disconnected";
  }
}

async function pushConfig() {
  const payload = {
    min_detection_confidence: Number(els.sliders.detection.value),
    min_tracking_confidence: Number(els.sliders.tracking.value),
    gesture_hold_frames: Number(els.sliders.hold.value),
    action_cooldown_ms: Number(els.sliders.cooldown.value),
    smoothing_alpha: Number(els.sliders.alpha.value),
  };

  renderConfig(payload);

  try {
    await fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (_) {
    // UI continues operating even if network request fails.
  }
}

function onSliderInput() {
  els.values.detection.textContent = formatNumber(els.sliders.detection.value);
  els.values.tracking.textContent = formatNumber(els.sliders.tracking.value);
  els.values.hold.textContent = els.sliders.hold.value;
  els.values.cooldown.textContent = els.sliders.cooldown.value;
  els.values.alpha.textContent = formatNumber(els.sliders.alpha.value);

  if (pendingConfigTimer) {
    clearTimeout(pendingConfigTimer);
  }
  pendingConfigTimer = setTimeout(pushConfig, 180);
}

els.resetBtn.addEventListener("click", async () => {
  await fetch("/api/reset", { method: "POST" });
  fetchStatus();
});

Object.values(els.sliders).forEach((slider) => {
  slider.addEventListener("input", onSliderInput);
});

setInterval(fetchStatus, 400);
fetchStatus();
