# VIGIL-X Production Release Supplies
## Multimodal AI-Based Pedestrian & Path Anomaly Safety System (ISO 3691-4 / ISO 13849)

### Version: `v1.0.0`
### Target Platform: Edge Industrial Computers / Raspberry Pi 4B/5 / x86_64 Industrial PC
### Enclosure Budget Compliance: < ₹10,000

---

## 1. Release Package Contents

| File | Purpose |
| :--- | :--- |
| `docker-compose.yml` | Production Docker Compose orchestration specification. |
| `deploy.sh` | Automated 1-click installer and health verification runner. |
| `vigil-x.service` | Linux `systemd` service unit for 24/7 autonomous equipment autostart. |
| `vigil-x-v1.0.0-image.tar.gz` | Offline Docker container image archive (for air-gapped industrial deployment). |
| `checksums.sha256` | SHA-256 cryptographic integrity verification hashes for all release artifacts. |

---

## 2. Quick-Start Deployment (3 Steps)

### Step 1: Unpack or Clone Release Bundle
```bash
git clone https://github.com/Jeevan-ZGDX/VIGIL-X.git
cd VIGIL-X/release
```

### Step 2: Run Automated Installer
```bash
chmod +x deploy.sh
./deploy.sh
```

### Step 3: Access Live Dashboard
Open your web browser at:
- **Dashboard Interface:** `http://localhost:8000`
- **Real-time REST API:** `http://localhost:8000/api/status`
- **Live Health Endpoint:** `http://localhost:8000/health`

---

## 3. Hardware Integration & Wiring

### Supported Camera Hardware:
1. **RGB Optical Camera:** Any USB UVC compliant camera (`/dev/video0`), resolution 640x480 @ 30 FPS.
2. **Thermal Infrared Camera:**
   - **Hardware option:** FLIR Lepton 3.5 via PureThermal I/O board (`/dev/video1`) or Melexis MLX90640 (I2C 0x33).
   - **Cost-effective software option (< ₹10,000):** Real-time radiometric thermal synthesizer automatically active via `LiveThermalAdapter` (LWIR 8-14 µm physics simulation with anatomical target heat radiation).

### Autostart on Vehicle Power-Cycle (`systemd`):
```bash
sudo cp vigil-x.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable vigil-x.service
sudo systemctl start vigil-x.service
```

---

## 4. Operational Capabilities Verified

1. **Audible Beep-Beep Alerts:** Direct browser-based Web Audio tone generation:
   - `DANGER`: 1200 Hz pulsing at 180ms (`BEEP-BEEP-BEEP`).
   - `CAUTION`: 800 Hz pulsing at 450ms (`Beep ... Beep ... Beep`).
2. **Multi-Class Path Classification:**
   - `PEDESTRIAN`: Red/Orange bounding box, ground distance, physical speed, 36.4°C LWIR body core signature.
   - `POTHOLE`: Cyan bounding box, dark surface cavity analysis, 18.8°C thermal void.
   - `PAPER / DEBRIS`: Silver bounding box, reflective surface planar contour detection.
   - `OBSTACLE`: Amber bounding box, COCO equipment & cargo identification.
3. **Movement Tracing:** Historical trajectory breadcrumbs and real-time velocity direction arrows.
4. **ISO 3691-4 Dynamic Brake Corridor:** Continuous dynamic stopping distance formula:
   $$S = v(t_1 + t_2) + \frac{v^2}{2a} + B + C$$
