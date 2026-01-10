

import time
import logging
from typing import Union, Optional
import numpy as np

from .models import (
    AadhaarVerificationResult,
    DemographicData,
    QRType
)
from .image_preprocessor import image_preprocessor
from .qr_detector import qr_detector
from .signature_verifier import signature_verifier
from .ocr_extractor import ocr_extractor
from .cross_validator import cross_validator
from .confidence_scorer import confidence_scorer
from .suspicious_image_detector import suspicious_detector

logger = logging.getLogger(__name__)


class AadhaarVerifier:
    
    
    def __init__(self):
        self.checks_performed = []
    
    def verify(self, image_input: Union[str, bytes, np.ndarray]) -> AadhaarVerificationResult:
        
        start_time = time.time()
        result = AadhaarVerificationResult()
        self.checks_performed = []
        
        try:
            # Step 1: Image Preprocessing
            logger.info("Step 1: Image Preprocessing")
            self.checks_performed.append("Image Preprocessing")
            
            try:
                preprocessed = image_preprocessor.preprocess(image_input)
                image = preprocessed.processed_image
                result.warnings.extend([f"Preprocessing: {note}" for note in preprocessed.preprocessing_notes[:3]])
            except Exception as e:
                logger.error(f"Preprocessing failed: {e}")
                result.errors.append(f"Image preprocessing failed: {str(e)}")
                result.processing_time_ms = (time.time() - start_time) * 1000
                return result
            
            # Step 1.5: Suspicious Image Detection (deity/celebrity/cartoon)
            logger.info("Step 1.5: Suspicious Image Detection")
            self.checks_performed.append("Suspicious Image Detection")
            
            try:
                suspicious_result = suspicious_detector.analyze_image(image)
                result.suspicious_image = suspicious_result
                
                if suspicious_result.get('is_suspicious'):
                    score_penalty = suspicious_result.get('suspicion_score', 0)
                    result.warnings.extend([f"Suspicious: {r}" for r in suspicious_result.get('reasons', [])[:3]])
                    logger.warning(f"Suspicious image detected: score={score_penalty}, reasons={suspicious_result.get('reasons')}")
            except Exception as e:
                logger.error(f"Suspicious detection failed: {e}")
                result.warnings.append(f"Suspicious detection error: {str(e)}")
            
            # Step 2: QR Detection and Decoding
            logger.info("Step 2: QR Code Detection")
            self.checks_performed.append("QR Code Detection")
            
            qr_data: Optional[DemographicData] = None
            raw_qr_bytes = b""
            
            try:
                qr_data, raw_qr_bytes = qr_detector.detect_and_decode(image)
                
                if qr_data:
                    result.qr_detected = True
                    result.qr_type = qr_data.qr_type.value if qr_data.qr_type else "unknown"
                    
                    # Extract demographic data
                    result.extracted_name = qr_data.name
                    result.masked_aadhaar = qr_data.masked_aadhaar
                    result.dob_or_yob = qr_data.dob or qr_data.yob
                    result.gender = qr_data.gender
                    result.reference_id = qr_data.reference_id
                    result.photo_available = bool(qr_data.photo_base64)
                    
                    logger.info(f"QR detected: type={result.qr_type}, name={result.extracted_name}")
                else:
                    result.qr_detected = False
                    result.warnings.append("No Aadhaar QR code detected in the image")
                    logger.warning("No QR code detected")
                    
            except Exception as e:
                logger.error(f"QR detection failed: {e}")
                result.qr_detected = False
                result.errors.append(f"QR detection error: {str(e)}")
            
            # Step 3: Digital Signature Verification (only if QR detected)
            logger.info("Step 3: Digital Signature Verification")
            self.checks_performed.append("Digital Signature Verification")
            
            if result.qr_detected and raw_qr_bytes:
                try:
                    # Extract signature and data from QR
                    signature_bytes = qr_detector.extract_signature_bytes(raw_qr_bytes)
                    signed_data = qr_detector.extract_signed_data(raw_qr_bytes)
                    
                    if signature_bytes and signed_data:
                        sig_result = signature_verifier.verify_signature(signature_bytes, signed_data)
                        result.qr_signature_valid = sig_result.is_valid
                        result.signature_details = sig_result
                        
                        if sig_result.is_valid:
                            logger.info("Digital signature verified successfully")
                        else:
                            logger.warning(f"Signature verification failed: {sig_result.error_message}")
                            result.warnings.append("Digital signature verification failed")
                    else:
                        # Try raw verification
                        sig_result = signature_verifier.verify_qr_raw(raw_qr_bytes)
                        result.qr_signature_valid = sig_result.is_valid
                        result.signature_details = sig_result
                        
                except Exception as e:
                    logger.error(f"Signature verification error: {e}")
                    result.qr_signature_valid = False
                    result.errors.append(f"Signature verification error: {str(e)}")
            else:
                result.qr_signature_valid = False
                if not result.qr_detected:
                    result.warnings.append("Cannot verify signature - no QR code detected")
            
            # Step 4: OCR Text Extraction
            logger.info("Step 4: OCR Text Extraction")
            self.checks_performed.append("OCR Text Extraction")
            
            ocr_data = None
            ocr_extracted = False
            
            try:
                ocr_data = ocr_extractor.extract(image)
                ocr_extracted = bool(ocr_data and ocr_data.raw_text)
                
                if ocr_extracted:
                    logger.info(f"OCR extracted, confidence: {ocr_data.ocr_confidence:.2f}")
                    
                    # If no QR data, use OCR data (with lower confidence)
                    if not result.qr_detected:
                        if ocr_data.name and not result.extracted_name:
                            result.extracted_name = ocr_data.name
                        if ocr_data.masked_aadhaar and not result.masked_aadhaar:
                            result.masked_aadhaar = ocr_data.masked_aadhaar
                        if (ocr_data.dob or ocr_data.yob) and not result.dob_or_yob:
                            result.dob_or_yob = ocr_data.dob or ocr_data.yob
                        if ocr_data.gender and not result.gender:
                            result.gender = ocr_data.gender
                        
                        result.warnings.append("Using OCR data only (less reliable than QR)")
                else:
                    result.warnings.append("OCR text extraction failed or yielded no text")
                    
            except Exception as e:
                logger.error(f"OCR extraction failed: {e}")
                result.warnings.append(f"OCR extraction error: {str(e)}")
            
            # Step 5: Cross-Validation
            logger.info("Step 5: Cross-Validation")
            self.checks_performed.append("Cross-Validation")
            
            try:
                cv_result = cross_validator.validate(qr_data, ocr_data)
                result.cross_validation = cv_result
                result.ocr_qr_match = cv_result.overall_match
                
                if cv_result.mismatches:
                    result.warnings.extend(cv_result.mismatches)
                    logger.warning(f"Cross-validation mismatches: {cv_result.mismatches}")
                    
            except Exception as e:
                logger.error(f"Cross-validation failed: {e}")
                result.warnings.append(f"Cross-validation error: {str(e)}")
            
            # Step 6: Confidence Scoring
            logger.info("Step 6: Confidence Scoring")
            self.checks_performed.append("Confidence Scoring")
            
            try:
                # Get full OCR result for new confidence scoring
                ocr_full_result = None
                if ocr_data and hasattr(ocr_data, '_full_result'):
                    ocr_full_result = ocr_data._full_result
                
                conf_score = confidence_scorer.calculate(
                    qr_detected=result.qr_detected,
                    qr_data=qr_data,
                    signature_result=result.signature_details,
                    ocr_extracted=ocr_extracted,
                    cross_validation=result.cross_validation,
                    ocr_result=ocr_full_result  # Pass full OCR result for new algorithm
                )
                
                result.confidence_score = conf_score.total_score
                result.confidence_breakdown = conf_score.breakdown
                
                logger.info(f"Final confidence score: {result.confidence_score}")
                
            except Exception as e:
                logger.error(f"Confidence scoring failed: {e}")
                result.confidence_score = 0
            
            # Set checks performed
            result.checks_performed = self.checks_performed
            
        except Exception as e:
            logger.error(f"Verification pipeline error: {e}")
            result.errors.append(f"Verification error: {str(e)}")
            result.confidence_score = 0
        
        # Calculate processing time
        result.processing_time_ms = round((time.time() - start_time) * 1000, 2)
        
        return result
    
    def get_verdict(self, result: AadhaarVerificationResult) -> dict:
        
        verdict = confidence_scorer.get_verdict(result.confidence_score)
        
        return {
            "verdict": verdict,
            "confidence": result.confidence_score,
            "summary": confidence_scorer.get_verdict_description(verdict),
            "qr_verified": result.qr_detected and result.qr_signature_valid,
            "processing_time_ms": result.processing_time_ms
        }


# Singleton instance and convenience function
aadhaar_verifier = AadhaarVerifier()


def verify_aadhaar_card(image_input: Union[str, bytes, np.ndarray]) -> dict:
    
    result = aadhaar_verifier.verify(image_input)
    return result.to_dict()
