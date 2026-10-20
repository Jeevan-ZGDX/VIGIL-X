// Pedestrian Safety System - Real-time Dashboard Controller

// Web Audio API Beep-Beep Sound Synthesizer
class AudioAlertManager {
  constructor() {
    this.audioCtx = null;
    this.isMuted = false;
    this.currentAlert = 'NORMAL';
    this.intervalId = null;
    this.isBeeping = false;
  }

  init() {
    if (!this.audioCtx) {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      if (AudioContext) {
        this.audioCtx = new AudioContext();
      }
    }
    if (this.audioCtx && this.audioCtx.state === 'suspended') {
      this.audioCtx.resume();
    }
  }

  playBeep(frequency = 800, durationMs = 120) {
    if (this.isMuted || !this.audioCtx || this.audioCtx.state !== 'running') return;
    try {
      const osc = this.audioCtx.createOscillator();
      const gain = this.audioCtx.createGain();

      osc.type = 'sine';
      osc.frequency.setValueAtTime(frequency, this.audioCtx.currentTime);

      gain.gain.setValueAtTime(0.15, this.audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, this.audioCtx.currentTime + durationMs / 1000.0);

      osc.connect(gain);
      gain.connect(this.audioCtx.destination);

      osc.start();
      osc.stop(this.audioCtx.currentTime + durationMs / 1000.0);
    } catch (e) {
      console.warn("Audio playback error:", e);
    }
  }

  setAlert(state) {
    if (state === this.currentAlert) return;
    this.currentAlert = state;
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }

    if (state === 'DANGER') {
      // Urgent, high-frequency double beep (1200 Hz every 180ms)
      this.playBeep(1200, 90);
      this.intervalId = setInterval(() => {
        this.playBeep(1200, 90);
      }, 180);
    } else if (state === 'CAUTION') {
      // Standard caution warning beep (800 Hz every 450ms)
      this.playBeep(800, 140);
      this.intervalId = setInterval(() => {
        this.playBeep(800, 140);
      }, 450);
    }
  }

  toggleMute() {
    this.init();
    this.isMuted = !this.isMuted;
    const btn = document.getElementById('btn-audio-toggle');
    const icon = document.getElementById('audio-icon');
    const label = document.getElementById('audio-label');
    if (this.isMuted) {
      if (icon) icon.textContent = '🔇';
      if (label) label.textContent = 'BEEP ALARM: MUTED';
      if (btn) btn.style.borderColor = '#ef4444';
      if (this.intervalId) {
        clearInterval(this.intervalId);
        this.intervalId = null;
      }
    } else {
      if (icon) icon.textContent = '🔊';
      if (label) label.textContent = 'BEEP ALARM: ARMED';
      if (btn) btn.style.borderColor = '#38bdf8';
      this.playBeep(1000, 80);
      const prevAlert = this.currentAlert;
      this.currentAlert = 'NORMAL';
      this.setAlert(prevAlert);
    }
  }
}

const audioAlerts = new AudioAlertManager();

// Unlock audio context on first user interaction
window.addEventListener('click', () => { audioAlerts.init(); }, { once: false });

async function fetchStatus() {
  try {
    const res = await fetch('/api/status');
    if (!res.ok) return;
    const data = await res.json();
    renderStatus(data);
  } catch (err) {
    console.error("Status fetch error:", err);
  }
}

async function fetchEvents() {
  try {
    const res = await fetch('/api/events?limit=15');
    if (!res.ok) return;
    const events = await res.json();
    renderEvents(events);
  } catch (err) {
    console.error("Events fetch error:", err);
  }
}

function renderStatus(data) {
  const assessment = data.assessment || {};
  const health = data.health || {};
  const telemetry = data.telemetry || {};
  const sim = data.simulator_outputs || {};

  // Alert State Banner
  const banner = document.getElementById('alert-banner');
  const alertText = document.getElementById('alert-text');
  const state = assessment.alert_state || 'NORMAL';

  banner.className = 'alert-banner ' + state.toLowerCase();
  alertText.textContent = state;

  // Trigger real Web Audio beep-beep alarms based on alert state
  audioAlerts.setAlert(state);

  // Reason
  document.getElementById('reason-text').textContent = assessment.reason || 'Normal operation';

  // Performance & System telemetry
  document.getElementById('val-fps').textContent = (health.fps || 0.0).toFixed(1);
  document.getElementById('val-latency').textContent = (health.latency_total_ms || 0.0).toFixed(1);
  document.getElementById('val-p95').textContent = (health.latency_p95_ms || 0.0).toFixed(1);
  document.getElementById('val-cpu').textContent = (health.cpu_percent || 0.0).toFixed(1);
  document.getElementById('val-ram').textContent = (health.memory_mb || 0.0).toFixed(0);

  // Sensor badges
  const sensors = health.sensors || {};
  updateBadge('status-rgb', sensors['cam_rgb'] || 'UNKNOWN');
  updateBadge('status-thermal', sensors['cam_thermal'] || 'UNKNOWN');

  // KPI Standard Safety Measures
  const speedMps = telemetry.speed_mps || 0.0;
  const speedKmh = Math.abs(speedMps * 3.6).toFixed(1);
  document.getElementById('kpi-speed-mps').textContent = speedMps.toFixed(1);
  document.getElementById('kpi-speed-kmh').textContent = speedKmh;
  document.getElementById('kpi-gear').textContent = telemetry.gear || 'FORWARD';

  const stopDist = assessment.stopping_distance_m !== null && assessment.stopping_distance_m !== undefined
    ? assessment.stopping_distance_m.toFixed(2)
    : '1.0';
  document.getElementById('kpi-stop-dist').textContent = stopDist;

  if (assessment.min_distance_m !== null && assessment.min_distance_m !== undefined) {
    document.getElementById('kpi-closest-dist').textContent = assessment.min_distance_m.toFixed(2);
    document.getElementById('kpi-closest-cm').textContent = (assessment.min_distance_m * 100).toFixed(0);
  } else {
    document.getElementById('kpi-closest-dist').textContent = '--';
    document.getElementById('kpi-closest-cm').textContent = '--';
  }

  if (assessment.time_to_collision_sec !== null && assessment.time_to_collision_sec !== undefined) {
    document.getElementById('kpi-ttc').textContent = assessment.time_to_collision_sec.toFixed(1);
    const ttc = assessment.time_to_collision_sec;
    const ttcLabel = document.getElementById('kpi-ttc-label');
    if (ttc <= 2.0) {
      ttcLabel.textContent = 'IMMINENT IMPACT HAZARD';
      ttcLabel.style.color = '#ef4444';
    } else if (ttc <= 4.0) {
      ttcLabel.textContent = 'APPROACHING BRAKE THRESHOLD';
      ttcLabel.style.color = '#f59e0b';
    } else {
      ttcLabel.textContent = 'MONITORING PATH';
      ttcLabel.style.color = '#38bdf8';
    }
  } else {
    document.getElementById('kpi-ttc').textContent = '--';
    document.getElementById('kpi-ttc-label').textContent = 'No trajectory conflict';
    document.getElementById('kpi-ttc-label').style.color = '#94a3b8';
  }

  const spotTemp = assessment.thermal_spot_celsius !== null && assessment.thermal_spot_celsius !== undefined
    ? assessment.thermal_spot_celsius.toFixed(1)
    : '24.2';
  document.getElementById('kpi-spot-temp').textContent = spotTemp;

  // Enunciators & Cabin Display
  document.getElementById('val-strobe').textContent = sim.visual_strobe ? 'ACTIVE' : 'OFF';
  document.getElementById('val-buzzer').textContent = sim.buzzer_hz > 0 ? `${sim.buzzer_hz} Hz` : 'SILENT';
  document.getElementById('cabin-display-text').textContent = sim.cabin_display || 'READY';

  // Tracks Table
  renderTracks(data.tracks || []);
}

function updateBadge(elId, status) {
  const el = document.getElementById(elId);
  if (!el) return;
  el.textContent = status;
  if (status === 'HEALTHY') {
    el.className = 'badge badge-ok';
  } else if (status === 'STALE') {
    el.className = 'badge badge-stale';
  } else {
    el.className = 'badge badge-disconnected';
  }
}

function renderTracks(tracks) {
  const tbody = document.getElementById('tracks-body');
  if (!tracks || tracks.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; color:#888;">No active path targets detected</td></tr>';
    return;
  }

  tbody.innerHTML = tracks.map(t => {
    const distM = t.ground_pos_m ? Math.hypot(t.ground_pos_m[0], t.ground_pos_m[1]) : 0;
    const distStr = `${distM.toFixed(2)} m (${Math.round(distM * 100)} cm)`;
    const speed = ((t.estimated_speed_mps || 0.0) * 3.6).toFixed(1);
    const conf = ((t.confidence || 0.0) * 100).toFixed(0) + '%';
    const clsName = (t.class_name || 'pedestrian').toUpperCase();

    let badgeClass = 'badge-ok';
    let irTemp = '36.2 °C';
    if (t.class_name === 'pothole') {
      badgeClass = 'badge-disconnected';
      irTemp = '18.8 °C';
    } else if (t.class_name === 'paper') {
      badgeClass = 'badge-stale';
      irTemp = '21.5 °C';
    } else if (t.class_name === 'obstacle') {
      badgeClass = 'badge-stale';
      irTemp = '22.0 °C';
    }

    return `
      <tr>
        <td><strong>${t.track_id}</strong></td>
        <td><span class="badge ${badgeClass}">${clsName}</span></td>
        <td><strong>${distStr}</strong></td>
        <td>${speed} km/h</td>
        <td><span style="color:#f59e0b; font-weight:600;">${irTemp}</span></td>
        <td>${conf}</td>
      </tr>
    `;
  }).join('');
}

// Audio Toggle Button Listener
const audioBtn = document.getElementById('btn-audio-toggle');
if (audioBtn) {
  audioBtn.addEventListener('click', () => {
    audioAlerts.toggleMute();
  });
}

function renderEvents(events) {
  const tbody = document.getElementById('events-body');
  if (!events || events.length === 0) return;

  tbody.innerHTML = events.slice(-10).reverse().map(ev => {
    const timeStr = ev.timestamp_iso ? ev.timestamp_iso.split('T')[1].replace('Z', '') : '-';
    const dist = ev.min_distance_m !== null ? `${ev.min_distance_m}m` : '-';
    const ttc = ev.time_to_collision_sec !== null ? `${ev.time_to_collision_sec}s` : '-';
    return `
      <tr>
        <td>${timeStr}</td>
        <td><code>${ev.event_id}</code></td>
        <td><strong>${ev.alert_state}</strong></td>
        <td>${ev.reason}</td>
        <td>${dist}</td>
        <td>${ttc}</td>
      </tr>
    `;
  }).join('');
}

// Source Selection Handler
document.getElementById('select-source').addEventListener('change', async (e) => {
  const source = e.target.value;
  await fetch('/api/set_source', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ source_type: source })
  });
  // Reset video streams to reconnect to new source
  document.getElementById('stream-rgb').src = '/stream/rgb?t=' + Date.now();
  document.getElementById('stream-thermal').src = '/stream/thermal?t=' + Date.now();
  fetchStatus();
});

// Fault Injection Handlers
document.getElementById('btn-recover').addEventListener('click', async () => {
  await fetch('/api/fault_injection', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sensor: 'rgb', action: 'recover' })
  });
  await fetch('/api/fault_injection', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sensor: 'thermal', action: 'recover' })
  });
  await fetch('/api/fault_injection', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sensor: 'telemetry', action: 'recover' })
  });
  fetchStatus();
});

document.getElementById('btn-fault-rgb').addEventListener('click', async () => {
  await fetch('/api/fault_injection', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sensor: 'rgb', action: 'disconnect' })
  });
  fetchStatus();
});

document.getElementById('btn-fault-thermal').addEventListener('click', async () => {
  await fetch('/api/fault_injection', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sensor: 'thermal', action: 'stale', stale_duration_sec: 1.5 })
  });
  fetchStatus();
});

document.getElementById('btn-fault-telem').addEventListener('click', async () => {
  await fetch('/api/fault_injection', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sensor: 'telemetry', action: 'disconnect' })
  });
  fetchStatus();
});

const speedSlider = document.getElementById('slider-speed');
speedSlider.addEventListener('input', async (e) => {
  const speed = parseFloat(e.target.value);
  document.getElementById('label-speed-slider').textContent = speed.toFixed(1);
  document.getElementById('label-speed-kmh').textContent = Math.abs(speed * 3.6).toFixed(1);
  const gear = speed < 0 ? 'REVERSE' : 'FORWARD';
  await fetch('/api/telemetry', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ speed_mps: Math.abs(speed), gear: gear, direction_deg: 0.0 })
  });
  fetchStatus();
});

// Real-time polling loop
setInterval(fetchStatus, 200);
setInterval(fetchEvents, 1000);
fetchStatus();
fetchEvents();
