"""
Data Models for Aadhaar Verification System

Pydantic models and dataclasses for type safety and validation.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime


class QRType(Enum):
    """Type of Aadhaar QR code"""
    SECURE_QR_V2 = "secure_qr_v2"
    OLD_XML_QR = "old_xml_qr"
    UNKNOWN = "unknown"


class Gender(Enum):
    """Gender enumeration"""
    MALE = "M"
    FEMALE = "F"
    OTHER = "O"
    UNKNOWN = "U"


@dataclass
class PreprocessedImage:
    """Result of image preprocessing"""
    original_path: str
    processed_image: Any  # numpy array
    rotation_applied: float = 0.0
    contrast_enhanced: bool = False
    blur_corrected: bool = False
    qr_region: Optional[tuple] = None  # (x, y, w, h)
    text_regions: List[tuple] = field(default_factory=list)
    preprocessing_notes: List[str] = field(default_factory=list)


@dataclass
class DemographicData:
    """Demographic data extracted from QR code"""
    name: Optional[str] = None
    dob: Optional[str] = None  # Date of Birth (DD-MM-YYYY)
    yob: Optional[str] = None  # Year of Birth
    gender: Optional[str] = None
    masked_aadhaar: Optional[str] = None  # XXXX-XXXX-1234
    reference_id: Optional[str] = None
    address: Optional[str] = None
    photo_base64: Optional[str] = None
    email_hash: Optional[str] = None
    mobile_hash: Optional[str] = None
    qr_type: QRType = QRType.UNKNOWN
    raw_data: Optional[str] = None


@dataclass
class SignatureResult:
    """Result of digital signature verification"""
    is_valid: bool
    verification_method: str = ""
    hash_algorithm: str = "SHA-256"
    signature_algorithm: str = "RSA-PKCS1v15"
    certificate_used: str = ""
    error_message: Optional[str] = None
    explanation: str = ""


@dataclass
class OCRExtractedData:
    """Data extracted via OCR from the Aadhaar image"""
    raw_text: str = ""
    name: Optional[str] = None
    dob: Optional[str] = None
    yob: Optional[str] = None
    gender: Optional[str] = None
    masked_aadhaar: Optional[str] = None
    address: Optional[str] = None
    vid: Optional[str] = None  # Virtual ID if present
    ocr_confidence: float = 0.0
    extraction_notes: List[str] = field(default_factory=list)


@dataclass
class FieldMatch:
    """Result of comparing a single field"""
    field_name: str
    qr_value: Optional[str]
    ocr_value: Optional[str]
    is_match: bool
    match_score: float = 0.0  # 0-1
    notes: str = ""


@dataclass
class CrossValidationResult:
    """Result of cross-validating QR and OCR data"""
    overall_match: bool
    overall_score: float  # 0-1
    field_matches: List[FieldMatch] = field(default_factory=list)
    mismatches: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ConfidenceScore:
    """Confidence score with breakdown"""
    total_score: int  # 0-100
    breakdown: Dict[str, int] = field(default_factory=dict)
    penalties: Dict[str, int] = field(default_factory=dict)
    explanation: str = ""


@dataclass
class AadhaarVerificationResult:
    """Final verification result returned by the system"""
    # Core verification status
    qr_detected: bool = False
    qr_signature_valid: bool = False
    
    # Extracted data
    extracted_name: Optional[str] = None
    masked_aadhaar: Optional[str] = None
    dob_or_yob: Optional[str] = None
    gender: Optional[str] = None
    reference_id: Optional[str] = None
    photo_available: bool = False
    
    # Cross-validation
    ocr_qr_match: bool = False
    
    # Confidence scoring
    confidence_score: int = 0
    confidence_breakdown: Dict[str, int] = field(default_factory=dict)
    
    # Verification metadata
    verification_type: str = "Offline Aadhaar Secure QR Verification"
    checks_performed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    # Processing details
    qr_type: str = "unknown"
    signature_details: Optional[SignatureResult] = None
    cross_validation: Optional[CrossValidationResult] = None
    
    # Timing
    processing_time_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Legal disclaimer
    limitations: str = (
        "This system verifies Aadhaar card authenticity offline using QR code "
        "digital signature verification. It does NOT verify Aadhaar existence "
        "or ownership in UIDAI databases. This is for academic purposes only."
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "qr_detected": self.qr_detected,
            "qr_signature_valid": self.qr_signature_valid,
            "extracted_name": self.extracted_name,
            "masked_aadhaar": self.masked_aadhaar,
            "dob_or_yob": self.dob_or_yob,
            "gender": self.gender,
            "reference_id": self.reference_id,
            "photo_available": self.photo_available,
            "ocr_qr_match": self.ocr_qr_match,
            "confidence_score": self.confidence_score,
            "confidence_breakdown": self.confidence_breakdown,
            "verification_type": self.verification_type,
            "checks_performed": self.checks_performed,
            "warnings": self.warnings,
            "errors": self.errors,
            "qr_type": self.qr_type,
            "processing_time_ms": self.processing_time_ms,
            "timestamp": self.timestamp,
            "limitations": self.limitations
        }
        
        if self.signature_details:
            result["signature_verification"] = {
                "is_valid": self.signature_details.is_valid,
                "method": self.signature_details.verification_method,
                "explanation": self.signature_details.explanation
            }
        
        if self.cross_validation:
            result["cross_validation_details"] = {
                "overall_match": self.cross_validation.overall_match,
                "score": self.cross_validation.overall_score,
                "mismatches": self.cross_validation.mismatches
            }
        
        return result
