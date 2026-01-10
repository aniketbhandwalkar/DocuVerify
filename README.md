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

##  🚀 One-Command Setup (Recommended)
For the most seamless experience on a new machine (Windows), follow these two steps:

### 1. Initial Setup
Run the setup script from the root directory. This will automatically install all Node.js dependencies, create a Python virtual environment, and install all AI/ML requirements.
```bash
.\setup.bat
```

### 2. Start the System
Once setup is complete, use the run script to launch all three services (Frontend, Backend, and AI Service) in separate windows simultaneously.
```bash
.\run_all.bat
```

---

## 🛠️ Manual Installation & Setup

If you prefer to set up manually or are on a non-Windows system:

### 1. Prerequisites
- **Node.js** (v16+)
- **Python** (3.10 to 3.12)
- **Windows Users**: Must install [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) for AI components.

### 2. AI/ML Service
```bash
cd ai-ml-service
python -m venv venv
.\venv\Scripts\activate   # On Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
python app.py
```

### 3. Backend (Node.js)
```bash
cd server
npm install
npm run dev
```

### 4. Frontend (React)
```bash
cd client
npm install
npm start
```

---

## Use Case

1. **Institutions**: Prevent fake medical certificates and achievement awards.
2. **Admissions**: Automate the verification of ID proofs and academic transcripts.
3. **HR**: Streamline employee onboarding with instant document validation.

---

> **Empower your institution with a secure, digital-first document verification solution!**
