# DocumentVerify – AI-Powered Smart Verification System

DocumentVerify is a state-of-the-art digital solution designed to streamline and secure document verification in educational institutions and organizations. It uses AI and forensics to detect tampered documents, verified Aadhaar cards offline, and manage student records efficiently.

---

## Key Features

- **AI-Powered Analysis**: Detects re-photography, digital tampering (ELA), and structural inconsistencies.
- **Offline Aadhaar Verification**: Securely decodes Aadhaar QR codes and verifies digital signatures without internet access.
- **Forensic Layer**: Integrated Moire pattern detection and Error Level Analysis (ELA).
- **Teacher Dashboard**: Real-time management and tracking of document verification statuses.
- **Paperless Workflow**: Replaces manual storage with a secure, searchable digital repository.
- **Privacy First**: Aadhaar data is processed locally; no sensitive data is sent to external servers.

---

##  Tech Stack

### Frontend
- **React.js**: Modern component-based UI.
- **Tailwind CSS**: Sleek, responsive design.
- **Framer Motion**: Smooth animations and micro-interactions.

### Backend (Node.js)
- **Express.js**: Robust API routing.
- **MongoDB**: Flexible document storage.
- **JWT**: Secure session-based authentication.

### AI/ML Service (Python)
- **FastAPI**: High-performance asynchronous API.
- **OpenCV**: Advanced image processing and QR detection.
- **EasyOCR**: Multi-lingual text extraction.
- **Scikit-Image**: Forensic image analysis.

---

## Project Structure

```bash
DocumentVerify/
├── client/           # React frontend
├── server/           # Node.js backend
├── ai-ml-service/    # AI/ML & Aadhaar Verification Service (Python)
├── uploads/          # Temporary file storage
└── README.md         # Main documentation
```

---

##  Installation & Setup

### 1. AI/ML Service (Python 3.10+)
```bash
cd ai-ml-service
# Use a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
```
*Note: For Windows users, the **Microsoft Visual C++ Redistributable (2015-2022) x64** is required for OpenCV and QR components. Download it here: [vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe).*

### 2. Backend (Node.js)
```bash
cd server
npm install
# Configure your .env file with MONGODB_URI and JWT_SECRET
npm start
```

### 3. Frontend (React)
```bash
cd client
npm install
npm start
```

### One-Command Setup
To install all requirements (Frontend, Backend, and AI Service) at once, run this in the root directory:
```bash
npm run install:all
```

---

### Shortcuts (Root Directory)
- **Start Backend**: `npm run start:server`
- **Start Frontend**: `npm run start:client`
- **Start AI Service**: `npm run start:ai-ml`

---

## Use Case

1. **Institutions**: Prevent fake medical certificates and achievement awards.
2. **Admissions**: Automate the verification of ID proofs and academic transcripts.
3. **HR**: Streamline employee onboarding with instant document validation.

---

> **Empower your institution with a secure, digital-first document verification solution!**
