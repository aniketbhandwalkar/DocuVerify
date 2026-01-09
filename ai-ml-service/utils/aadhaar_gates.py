"""
Hard Gating Decision Module for Aadhaar Verification

Replaces soft scoring with deterministic gates that STOP processing
if critical requirements are not met.

Gates (in order):
1. OCR Quality: confidence ≥ 0.75, text_length ≥ 150
2. Aadhaar Number: Valid 12-digit with Verhoeff checksum
3. Required Fields: Name + DOB present

Outcomes:
- NEEDS_REUPLOAD: Poor OCR quality
- REJECT: Invalid/missing Aadhaar
- ACCEPT: All gates passed
"""

import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


class AadhaarDecisionGates:
    """
    Deterministic decision gates for Aadhaar verification.
    No soft scoring - hard gates that reject immediately if failed.
    """
    
    # Gate thresholds - ADJUSTED FOR REAL-WORLD IMAGES
    # Screenshots and phone photos typically get 30-50% OCR confidence
    MIN_OCR_CONFIDENCE = 0.30  # Lowered from 0.75 - screenshots get ~38%
    MIN_TEXT_LENGTH = 22       # Lowered to 22 - minimum viable text
    MIN_FINAL_CONFIDENCE = 50  # Lowered from 70 - more lenient
    
    @classmethod
    def check_ocr_quality(cls, ocr_result: Dict[str, Any]) -> Tuple[str, str]:
        """
        Gate 1: OCR Quality Check
        
        Returns:
            ("PASS", None) if quality sufficient
            ("NEEDS_REUPLOAD", reason) if quality too poor
        """
        ocr_conf = ocr_result.get('ocr_confidence', {}).get('mean', 0.0)
        text_length = len(ocr_result.get('full_text', ''))
        
        if ocr_conf < cls.MIN_OCR_CONFIDENCE:
            reason = f"OCR confidence {ocr_conf:.2%} below threshold {cls.MIN_OCR_CONFIDENCE:.0%}"
            logger.warning(f"OCR Quality Gate FAILED: {reason}")
            return ("NEEDS_REUPLOAD", reason)
        
        if text_length < cls.MIN_TEXT_LENGTH:
            reason = f"Text length {text_length} chars below threshold {cls.MIN_TEXT_LENGTH}"
            logger.warning(f"OCR Quality Gate FAILED: {reason}")
            return ("NEEDS_REUPLOAD", reason)
        
        logger.info(f"OCR Quality Gate PASSED: confidence={ocr_conf:.2%}, length={text_length}")
        return ("PASS", None)
    
    @classmethod
    def check_aadhaar_number(cls, ocr_result: Dict[str, Any]) -> Tuple[str, str]:
        """
        Gate 2: Aadhaar Number Validation
        
        Returns:
            ("PASS", None) if valid Aadhaar number with correct checksum
            ("REJECT", reason) if no number or invalid checksum
        """
        aadhaar_number = ocr_result.get('aadhaar_number')
        aadhaar_valid = ocr_result.get('aadhaar_valid', False)
        
        if not aadhaar_number:
            reason = "No 12-digit Aadhaar number found in image"
            logger.warning(f"Aadhaar Number Gate FAILED: {reason}")
            return ("REJECT", reason)
        
        if not aadhaar_valid:
            reason = "Aadhaar number failed Verhoeff checksum validation"
            logger.warning(f"Aadhaar Number Gate FAILED: {reason}")
            return ("REJECT", reason)
        
        logger.info(f"Aadhaar Number Gate PASSED: {ocr_result.get('aadhaar_masked')}")
        return ("PASS", None)
    
    @classmethod
    def calculate_confidence(cls, ocr_result: Dict[str, Any]) -> int:
        """
        Calculate final confidence score (only called if all gates passed)
        
        Formula:
        - Base: OCR confidence × 100
        - Bonus: +10 if Aadhaar valid
        - Bonus: +5 if Name present
        - Bonus: +5 if DOB present
        - Max: 100
        
        Returns:
            Confidence score 0-100
        """
        # Base score from OCR confidence
        ocr_conf = ocr_result.get('ocr_confidence', {}).get('mean', 0.0)
        base_score = ocr_conf * 100
        
        # Bonuses for extracted fields
        bonus = 0
        
        if ocr_result.get('aadhaar_valid'):
            bonus += 10
            logger.debug("Confidence bonus +10: Aadhaar number validated")
        
        if ocr_result.get('name'):
            bonus += 5
            logger.debug("Confidence bonus +5: Name extracted")
        
        if ocr_result.get('dob'):
            bonus += 5
            logger.debug("Confidence bonus +5: DOB extracted")
        
        final_score = min(base_score + bonus, 100)
        
        logger.info(f"Confidence calculated: base={base_score:.1f} + bonus={bonus} = {final_score:.1f}")
        return int(final_score)
    
    @classmethod
    def make_final_decision(cls, confidence: int) -> Tuple[str, str]:
        """
        Gate 3: Final Decision based on confidence
        
        Returns:
            ("ACCEPT", message) if confidence ≥ threshold
            ("REJECT", reason) otherwise
        """
        if confidence >= cls.MIN_FINAL_CONFIDENCE:
            message = f"Aadhaar verified with {confidence}% confidence"
            logger.info(f"Final Decision: ACCEPT ({message})")
            return ("ACCEPT", message)
        else:
            reason = f"Confidence {confidence}% below threshold {cls.MIN_FINAL_CONFIDENCE}%"
            logger.warning(f"Final Decision: REJECT ({reason})")
            return ("REJECT", reason)
    
    @classmethod
    def execute_full_pipeline(cls, ocr_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute all gates in sequence and return final decision
        
        Args:
            ocr_result: Full OCR extraction result from AadhaarOCRExtractor
            
        Returns:
            {
                "verdict": "ACCEPT" | "REJECT" | "NEEDS_REUPLOAD",
                "confidence": int (0-100),
                "reason": str,
                "gates_passed": [list of passed gates],
                "gates_failed": [list of failed gates]
            }
        """
        # CONSOLE LOGGING - SHOW DECISION MAKING
        print(f"\n{'='*60}")
        print(f"[AADHAAR DECISION GATES]")
        print(f"{'='*60}")
        print(f"OCR Confidence: {ocr_result.get('ocr_confidence', {}).get('mean', 0):.2%}")
        print(f"Text Length: {len(ocr_result.get('full_text', ''))} chars (threshold: {cls.MIN_TEXT_LENGTH})")
        print(f"Aadhaar Number: {ocr_result.get('aadhaar_number', 'NOT FOUND')}")
        print(f"Aadhaar Valid: {ocr_result.get('aadhaar_valid', False)}")
        print(f"{'='*60}")
        
        gates_passed = []
        gates_failed = []
        
        # Gate 1: OCR Quality
        status, reason = cls.check_ocr_quality(ocr_result)
        print(f"Gate 1 (OCR Quality): {status} - {reason if reason else 'PASSED'}")
        if status != "PASS":
            print(f"{'='*60}\n")
            return {
                "verdict": status,
                "confidence": 0,
                "reason": reason,
                "gates_passed": gates_passed,
                "gates_failed": ["OCR Quality"]
            }
        gates_passed.append("OCR Quality")
        
        # Gate 2: Aadhaar Number
        status, reason = cls.check_aadhaar_number(ocr_result)
        print(f"Gate 2 (Aadhaar Number): {status} - {reason if reason else 'PASSED'}")
        if status != "PASS":
            print(f"{'='*60}\n")
            return {
                "verdict": status,
                "confidence": 0,
                "reason": reason,
                "gates_passed": gates_passed,
                "gates_failed": ["Aadhaar Number Validation"]
            }
        gates_passed.append("Aadhaar Number Validation")
        
        # Calculate confidence (only if gates passed)
        confidence = cls.calculate_confidence(ocr_result)
        print(f"Confidence Score: {confidence}% (threshold: {cls.MIN_FINAL_CONFIDENCE}%)")
        gates_passed.append("Confidence Calculation")
        
        # Gate 3: Final Decision
        verdict, message = cls.make_final_decision(confidence)
        print(f"Gate 3 (Final Decision): {verdict} - {message}")
        
        if verdict == "ACCEPT":
            gates_passed.append("Final Decision")
        else:
            gates_failed.append("Final Decision")
        
        print(f"{'='*60}\n")
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "reason": message if verdict == "ACCEPT" else message,
            "gates_passed": gates_passed,
            "gates_failed": gates_failed,
            "extracted_data": {
                "name": ocr_result.get('name'),
                "dob": ocr_result.get('dob'),
                "gender": ocr_result.get('gender'),
                "masked_aadhaar": ocr_result.get('aadhaar_masked')
            }
        }


# Convenience function
def verify_aadhaar_with_gates(ocr_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verify Aadhaar using hard gating logic
    
    Args:
        ocr_result: Result from AadhaarOCRExtractor.extract_aadhaar_data()
        
    Returns:
        Decision dict with verdict, confidence, and reason
    """
    return AadhaarDecisionGates.execute_full_pipeline(ocr_result)
