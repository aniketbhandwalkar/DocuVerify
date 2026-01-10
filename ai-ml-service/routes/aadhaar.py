

from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import the verifier - may fail on Windows due to DLL issues
VERIFIER_AVAILABLE = False
try:
    from aadhaar_verifier import verify_aadhaar_card, AadhaarVerifier
    verifier = AadhaarVerifier()
    VERIFIER_AVAILABLE = True
    logger.info("AadhaarVerifier loaded successfully")
except ImportError as e:
    logger.warning(f"Could not import AadhaarVerifier: {e}")
    logger.warning("Running in fallback mode - some features may be limited")
    verifier = None
except Exception as e:
    logger.error(f"Error loading AadhaarVerifier: {e}")
    verifier = None

router = APIRouter(tags=["Aadhaar Verification"])


@router.post("/verify-aadhaar-offline")
async def verify_aadhaar_offline(file: UploadFile = File(...)):
    
    temp_file = None
    
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload an image file (JPEG, PNG, etc.)"
            )
        
        # Save uploaded file temporarily
        temp_file = f"temp_aadhaar_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Processing Smart Aadhaar verification for: {file.filename}")
        
        # Import Smart Verifier
        from utils.smart_verifier import smart_verifier
        
        # Execute Smart Analysis
        result = smart_verifier.analyze(temp_file)
        
        # Map to final response format
        response = {
            "success": True,
            "verdict": result["verdict"],
            "confidence": result["confidence"],
            "reason": result["reason"],
            "findings": result["findings"],
            "extracted_data": result["extracted_data"],
            "forensics": {
                "ela_score": result["ela_score"],
                "is_rephoto": result["is_rephoto"]
            },
            "timestamp": datetime.now().isoformat(),
            "api_version": "4.0.0-smart-forensics"
        }
        
        logger.info(f"Smart Verification complete: {result['verdict']} (score: {result['confidence']})")
        
        return JSONResponse(content=response, status_code=200)
        
    except Exception as e:
        logger.error(f"Aadhaar verification error: {str(e)}", exc_info=True)
        return JSONResponse(
            content={
                "success": False,
                "verdict": "ERROR",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            },
            status_code=500
        )
    finally:
        # Clean up temp file
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass


@router.post("/verify-aadhaar-quick")
async def verify_aadhaar_quick(file: UploadFile = File(...)):
    
    temp_file = None
    
    try:
        # Save file temporarily
        temp_file = f"temp_quick_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Perform verification
        result = verifier.verify(temp_file)
        verdict = verifier.get_verdict(result)
        
        # Simplified response
        return {
            "verdict": verdict["verdict"],
            "confidence": result.confidence_score,
            "qr_detected": result.qr_detected,
            "signature_valid": result.qr_signature_valid,
            "masked_aadhaar": result.masked_aadhaar,
            "name": result.extracted_name,
            "dob_or_yob": result.dob_or_yob,
            "gender": result.gender,
            "processing_time_ms": result.processing_time_ms,
            "limitations": result.limitations
        }
        
    except Exception as e:
        logger.error(f"Quick verification error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass


@router.get("/aadhaar-verification-info")
async def get_verification_info():
    
    return {
        "system": "Offline Aadhaar Card Verification System",
        "version": "2.0.0",
        "verification_method": "Secure QR Code Digital Signature Verification",
        "capabilities": [
            "Secure QR Code Detection and Decoding",
            "Digital Signature Verification (RSA-2048, SHA-256)",
            "OCR Text Extraction",
            "QR-OCR Cross-Validation",
            "Confidence Scoring"
        ],
        "supported_qr_formats": [
            "Aadhaar Secure QR V2 (Numeric/Compressed)",
            "Aadhaar Old XML QR"
        ],
        "limitations": [
            "This system verifies card authenticity OFFLINE only",
            "It does NOT verify Aadhaar existence in UIDAI database",
            "It does NOT verify card ownership",
            "Academic/demonstration purposes only"
        ],
        "legal_notice": (
            "This system is for offline verification only. "
            "It uses cryptographic methods to verify the QR code signature. "
            "It does NOT connect to UIDAI servers or databases."
        )
    }
