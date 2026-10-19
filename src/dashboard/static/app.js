// Pedestrian Safety System - Dashboard Frontend Controller

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

  // Reason
  document.getElementById('reason-text').textContent = assessment.reason || 'Normal operation';

  // Performance
  document.getElementById('val-fps').textContent = (health.fps || 0.0).toFixed(1);
  document.getElementById('val-latency').textContent = (health.latency_total_ms || 0.0).toFixed(1);

  // Sensor badges
  const sensors = health.sensors || {};
  updateBadge('status-rgb', sensors['cam_rgb'] || 'UNKNOWN');
  updateBadge('status-thermal', sensors['cam_thermal'] || 'UNKNOWN');

  // Telemetry & Simulator
  document.getElementById('val-speed').textContent = (telemetry.speed_mps || 0.0).toFixed(1) + ' m/s';
  document.getElementById('val-gear').textContent = telemetry.gear || 'FORWARD';
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
    tbody.innerHTML = '<tr><td colspan="5" style="text-align:center; color:#888;">No active pedestrian tracks</td></tr>';
    return;
  }

  tbody.innerHTML = tracks.map(t => {
    const dist = t.ground_pos_m ? Math.hypot(t.ground_pos_m[0], t.ground_pos_m[1]).toFixed(1) : '-';
    const speed = (t.estimated_speed_mps || 0.0).toFixed(1);
    const conf = ((t.confidence || 0.0) * 100).toFixed(0) + '%';
    const sources = (t.sensor_sources || []).join(', ');
    return `
      <tr>
        <td><strong>${t.track_id}</strong></td>
        <td>${dist} m</td>
        <td>${speed} m/s</td>
        <td>${conf}</td>
        <td>${sources}</td>
      </tr>
    `;
  }).join('');
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

// Event Listeners for Simulator Controls
document.getElementById('btn-step').addEventListener('click', async () => {
  await fetch('/api/step', { method: 'POST' });
  fetchStatus();
  fetchEvents();
});

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
  const gear = speed < 0 ? 'REVERSE' : 'FORWARD';
  await fetch('/api/telemetry', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ speed_mps: Math.abs(speed), gear: gear, direction_deg: 0.0 })
  });
  fetchStatus();
});

// Polling interval loops
setInterval(fetchStatus, 250);
setInterval(fetchEvents, 1000);
fetchStatus();
fetchEvents();
