# VisionPark AI

VisionPark AI is a local-first vehicle intelligence and parking occupancy workspace. It pairs a React/TypeScript dashboard with a FastAPI/OpenCV service. Video is routed to vehicle tracking and line-crossing counts; images are routed to user-configured parking slot analysis.

## Requirements

- Windows 10/11
- Node.js 20.19+ or 22.12+ and npm
- Python 3.10–3.12 (Python 3.11 is a good default)
- Internet access on the first inference run so Ultralytics can fetch its model weights, unless you already have the weights locally

## Run on Windows

Open two PowerShell terminals in the project directory.

### 1. Start the backend

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

If PowerShell blocks activation, use `Set-ExecutionPolicy -Scope Process Bypass` in that terminal and activate again. The API is at http://localhost:8000 and interactive docs are at http://localhost:8000/docs.

The included `backend\yolo11n.pt` model is loaded once on first processing. It is a small pretrained Ultralytics model. Set `YOLO_MODEL` to a compatible local weight path or model name to change it. Inference selects CUDA when PyTorch reports a CUDA device and otherwise uses CPU. CPU processing is slower, especially for long videos.

### 2. Start the frontend

In a second PowerShell terminal, from the project root:

```powershell
npm install
npm run dev
```

Open http://localhost:5173. Sign in with a valid email and any non-empty password. This is prototype-only session authentication stored in `sessionStorage`.

## Use the workspace

- **Vehicle video:** drop or select MP4, AVI, MOV, or MKV. The app automatically opens Vehicle Counter and submits a background job. YOLO vehicle detections are associated by Ultralytics ByteTrack; a vehicle is counted once when its tracked center crosses the virtual line at 58% of frame height. The result includes counts by category, annotated MP4, and CSV.
- **Parking image:** drop or select JPG, JPEG, PNG, or WEBP. The app automatically opens Parking Intelligence. The app automatically analyzes the image and estimates occupied bays plus visible gaps between parked-vehicle rows. Detected vehicles are outlined red; estimated free gaps are green. Because ordinary vehicle models do not identify painted stall boundaries, this automatic count is a visual estimate and does not guess empty spaces at row ends. If the image does not show clear rows, draw polygons around spaces and choose Analyze occupancy. The result includes free/occupied totals, an annotated image, and a CSV. Each image is a single-time snapshot.
- **Analytics / Reports:** only actual completed processing results are displayed. Parking images represent a single moment; continuous monitoring requires a video or camera integration.

## Configuration

Backend settings are environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `YOLO_MODEL` | `yolo11n.pt` | Ultralytics model name or local weights path |
| `MAX_UPLOAD_MB` | `500` | Maximum upload size |
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated allowed frontend origins |
| `LOG_LEVEL` | `INFO` | Python logging level |

Frontend API URL defaults to `http://127.0.0.1:8000`; override at build time using `VITE_API_URL`.

Uploads are temporary and removed after processing. Results live under `backend/results` and are ignored by Git. This local prototype does not include database accounts, cloud storage, authentication hardening, job persistence across restarts, or live camera capture.

## Architecture

```text
src/                    React app, UI, routing-by-file-type, charts
backend/app/main.py     FastAPI endpoints, job state, processing routes
backend/uploads/        Temporary source uploads
backend/results/        Annotated media and CSV reports
```

The API exposes `/api/health`, `/api/vehicle/process`, `/api/parking/process`, `/api/jobs/{job_id}`, `/api/files/{job_id}/{kind}`, and `/api/history`.

## Troubleshooting

- **Model load fails:** confirm `backend\yolo11n.pt` exists and the Python environment has PyTorch and Ultralytics installed. Set `YOLO_MODEL` to a compatible local weight file if using a different model. The API returns an explicit processing error; it never substitutes generated detections.
- **CUDA unavailable:** CPU fallback is automatic. For GPU inference, install the PyTorch build compatible with your CUDA setup before installing/using Ultralytics.
- **Video codec error:** use a decodable MP4/AVI/MOV/MKV file. Some containers use codecs unavailable in the local OpenCV build.
- **Browser says API offline:** start the backend and confirm `http://localhost:8000/api/health` responds.
- **Parking polygons:** images are analyzed automatically first. If automatic row/spacing estimation is not available, click at least three points around each slot and choose Save slot before analyzing. Use Clear slots to redraw.

## Build

```powershell
npm run build
```
