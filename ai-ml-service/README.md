#  AI & Aadhaar Verification Service

This service is the cognitive engine of DocumentVerify, built with **FastAPI** and **Python**. It provides cryptographic Aadhaar verification and forensic image analysis.

##  Core Modules

### 1. Smart Forensic Verifier (`utils/smart_verifier.py`)
A 5-layer analysis pipeline for high-fidelity verification:
- **Layer 1: Forensic Scan** (ELA + Moire pattern detection).
- **Layer 2: Crypto Scan** (Decodes Aadhaar Secure QR).
- **Layer 3: Visual Layer** (OCR Extraction via EasyOCR).
- **Layer 4: Structural Scan** (Aadhaar number validation).
- **Layer 5: Cross-Validation** (Matches QR data vs OCR text).

### 2. Offline Aadhaar Verifier (`aadhaar_verifier/`)
A modular system specifically for cryptographic validation:
- Offline RSA signature verification.
- Demographic data extraction from V2 Secure QRs.
- QR code image enhancement pipeline.

---

##  Setup Instructions

### Prerequisites
- **Python 3.10 to 3.12** (Recommended).
- **Windows Users**: You MUST install the [Microsoft Visual C++ Redistributable (2015-2022) x64](https://aka.ms/vs/17/release/vc_redist.x64.exe) for OpenCV and QR components to work.

### Installation
1. Navigate to the directory:
   ```bash
   cd ai-ml-service
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Service
```bash
uvicorn app:app --reload --port 8000
```
The API will be available at `http://localhost:8000`. You can view the interactive documentation at `http://localhost:8000/docs`.

---

##  Model Management & Transfer

If you are moving this project to a new machine:

### 1. Library-Based Models (Auto-Download)
The following models will **automatically download** when you run the service for the first time:
- **EasyOCR**: Downloads ~100MB of detection/recognition models.
- **ResNet-18**: Used as a backbone for seal detection.
- **DeepFace/FaceRecognition**: Downloads facial feature extractors.

### 2. Custom Trained Models
If you have trained custom classifiers (SVM, XGBoost, or CNNs), ensure the following files are placed in the `ai-ml-service/models/` directory:
- `logo_cnn.pth`
- `svm_document_classifier.joblib`
- `xgboost_document_classifier.joblib`

### 3. Zipping for Migration
When you zip this folder:
- **Included**: All source code, configuration, and any files saved in `models/`.
- **Not Included**: The `.venv` folder (should be recreated on new machine) and auto-downloaded cache (usually stored in your user profile path like `~/.EasyOCR`).

---

##  API Endpoints

- `POST /api/v1/verify-aadhaar-offline`: The primary deterministic pipeline (ACCEPT/REJECT).
- `POST /api/v1/verify-aadhaar-quick`: Fast cryptographic-only check.
- `GET /api/v1/aadhaar-verification-info`: System capabilities and limitations info.

---

##  Important Note regarding `pyzbar`
This service includes fallback logic for `pyzbar`. If you encounter "DLL not found" errors on Windows, ensure the `zbar` shared library is in your system path, or the system will automatically fall back to the **OpenCV QRCodeDetector** included in the code.
