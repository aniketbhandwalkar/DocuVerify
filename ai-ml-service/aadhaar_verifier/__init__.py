

import logging

logger = logging.getLogger(__name__)

# Import models first (no external dependencies)
from .models import (
    AadhaarVerificationResult,
    DemographicData,
    OCRExtractedData,
    CrossValidationResult,
    ConfidenceScore,
    SignatureResult,
    QRType
)

# Import main verifier class and function (these may have external deps)
try:
    from .aadhaar_verifier import AadhaarVerifier, verify_aadhaar_card
except ImportError as e:
    logger.warning(f"Could not import AadhaarVerifier: {e}")
    # Create a stub for when dependencies are missing
    class AadhaarVerifier:
        def verify(self, image_input):
            return AadhaarVerificationResult(
                errors=["AadhaarVerifier dependencies not installed"]
            )
    
    def verify_aadhaar_card(image_input):
        return {"error": "Dependencies not installed", "confidence_score": 0}

__version__ = "2.0.0"
__all__ = [
    "AadhaarVerifier",
    "verify_aadhaar_card",
    "AadhaarVerificationResult",
    "DemographicData",
    "OCRExtractedData",
    "CrossValidationResult",
    "ConfidenceScore",
    "SignatureResult",
    "QRType"
]
