from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import logging
from datetime import datetime
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Document Verification AI/ML Service",
    description="Microservice for document verification and analysis.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from routes.analysis import router as analysis_router
from routes.ocr import router as ocr_router
from routes.signature import router as signature_router
from routes.validation import router as validation_router
from routes.aadhaar import router as aadhaar_router

app.include_router(analysis_router, prefix="/api/v1")
app.include_router(ocr_router, prefix="/api/v1")
app.include_router(signature_router, prefix="/api/v1")
app.include_router(validation_router, prefix="/api/v1")
app.include_router(aadhaar_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "service": "Document Verification AI/ML Service",
        "version": "2.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/verify-aadhaar")
async def verify_aadhaar_legacy(file: UploadFile = File(...)):
    import shutil
    temp_file = None
    try:
        from aadhaar_verifier import AadhaarVerifier
        temp_file = f"temp_{file.filename}"
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        verifier = AadhaarVerifier()
        result = verifier.verify(temp_file)
        verdict = verifier.get_verdict(result)
        
        return {
            "verdict": verdict["verdict"],
            "confidence": result.confidence_score / 100,
            "reasons": result.warnings + result.errors if result.warnings or result.errors else ["Success"],
            "checks": {
                "qr_detected": result.qr_detected,
                "data_verified": result.qr_signature_valid
            },
            "aadhaar_data": {
                "name": result.extracted_name,
                "masked_aadhaar": result.masked_aadhaar
            }
        }
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return {"verdict": "error", "confidence": 0, "error": str(e)}
    finally:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == "__main__":
    def find_free_port(start_port=8000):
        import socket
        port = start_port
        while port < start_port + 100:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('localhost', port))
                    return port
            except OSError:
                port += 1
        return start_port

    port = int(os.getenv('PORT', 8000))
    try:
        uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
    except OSError:
        free_port = find_free_port(port + 1)
        uvicorn.run("app:app", host="0.0.0.0", port=free_port, reload=True)
