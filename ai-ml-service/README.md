#  AI & Aadhaar Verification Service

This service is the cognitive engine of DocumentVerify, built with **FastAPI** and **Python**. It provides cryptographic Aadhaar verification and forensic image analysis.

## 🚀 Aadhaar Verification Flow (File Path)
When a user uploads an Aadhaar card, the following files execute in order:
1. **Node.js Gateway**: `server/controllers/documentController.js` receives the file and calls the AI service.
2. **API Endpoint**: `ai-ml-service/routes/aadhaar.py` handles the incoming request and triggers the verifier.
3. **Smart Verifier (The Brain)**: `ai-ml-service/utils/smart_verifier.py` manages the 5-layer analysis.
4. **Forensic Analysis**: `ai-ml-service/core/forensics.py` checks for editing/transparency issues (using **Pillow**).
5. **QR Code Engine**: `ai-ml-service/core/crypto.py` rotates, enhances, and decodes the QR (using **OpenCV**).
6. **Visual Analysis (OCR)**: `ai-ml-service/utils/smart_verifier.py` performs a multi-pass OCR scan using **EasyOCR**.
7. **Plausibility Check**: `ai-ml-service/core/plausibility_engine.py` validates the Aadhaar 12-digit pattern (Verhoeff).
8. **Verdict Implementation**: The logic in `smart_verifier.py` applies the "Client Mode" logic to ensure real cards pass.

## 🛡️ "Client Mode" (Zero-Friction)
This system has been optimized for client demonstrations:
- **Low Thresholds**: Rejects only obvious fakes; real cards pass even with low-quality photos.
- **Rotation Support**: Works perfectly if the Aadhaar is uploaded horizontally or vertically.
- **Smart OCR**: Automatically reads ID numbers even if they have spaces (e.g., `1234 5678 9012`).
- **Auto-Boost**: If a valid 12-digit pattern is found, the system automatically trusts the card.

---

## ⚙️ Setup Instructions (Client Machine)

### Prerequisites
- **Python 3.10 to 3.12** (Mandatory).
- **Windows Users**: You MUST install the [Microsoft Visual C++ Redistributable (2015-2022) x64](https://aka.ms/vs/17/release/vc_redist.x64.exe). Without this, the QR and Image processing libraries will fail.

### Installation
1.  **Navigate to directory**:
    ```bash
    cd ai-ml-service
    ```
2.  **Create Virtual Environment**:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```
3.  **Install Requirements**:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Service
```bash
python app.py
```
*Note: Using `app.py` ensures the server starts correctly on the configured port 8000.*

---

## 🛠️ Key Libraries Used
- **EasyOCR**: For high-accuracy text extraction (Aadhaar numbers).
- **OpenCV**: For rotation, QR code enhancement, and image processing.
- **Pillow**: For forensic ELA analysis and format handling.
- **Pyzbar/Pyaadhaar**: (Included as fallbacks for raw cryptographic data).

---

## 📦 Migration & Zipping
When moving this to a new machine:
1. **Do NOT zip the `.venv` folder**.
2. Zip the entire `ai-ml-service` folder.
3. On the new machine, follow the **Installation** steps above to recreate the environment.
4. The first run will automatically download ~100MB of AI models for EasyOCR.
