import logging
from typing import Optional, Dict, Any

from .models import (
    DemographicData,
    SignatureResult,
    CrossValidationResult,
    ConfidenceScore
)

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    
    
    # Weights for new scoring algorithm
    WEIGHTS = {
        'ocr_confidence': 0.30,      # Tesseract confidence
        'checksum_valid': 0.25,      # Verhoeff validation
        'format_match': 0.15,        # Pattern matching
        'field_completeness': 0.15,  # Extracted fields ratio
        'text_quality': 0.15,        # Character quality metrics
    }
    
    # Expected fields for completeness calculation
    EXPECTED_FIELDS = ['aadhaar_number', 'name', 'dob', 'gender']
    
    # QR bonus when available (adds to final score)
    QR_BONUS = 15
    SIGNATURE_BONUS = 10
    
    def calculate_ocr_based(
        self,
        ocr_result: Dict[str, Any],
        text_quality_metrics: Optional[Dict[str, float]] = None
    ) -> ConfidenceScore:
        
        breakdown = {}
        score = 0.0
        
        # 1. OCR Confidence (30%)
        ocr_conf = ocr_result.get('ocr_confidence', {}).get('mean', 0.0)
        ocr_score = ocr_conf * self.WEIGHTS['ocr_confidence'] * 100
        score += ocr_score
        breakdown['OCR Quality'] = round(ocr_score, 2)
        
        # 2. Checksum Validation (25%)
        aadhaar_valid = ocr_result.get('aadhaar_valid', False)
        checksum_score = (1.0 if aadhaar_valid else 0.0) * self.WEIGHTS['checksum_valid'] * 100
        score += checksum_score
        breakdown['Verhoeff Checksum'] = round(checksum_score, 2)
        
        # 3. Format Matching (15%)
        format_score = self._calculate_format_score(ocr_result)
        score += format_score
        breakdown['Format Match'] = round(format_score, 2)
        
        # 4. Field Completeness (15%)
        completeness_score = self._calculate_completeness_score(ocr_result)
        score += completeness_score
        breakdown['Field Completeness'] = round(completeness_score, 2)
        
        # 5. Text Quality (15%)
        if text_quality_metrics:
            quality_score = self._calculate_text_quality_score(text_quality_metrics)
        else:
            # Estimate from OCR confidence
            quality_score = ocr_conf * self.WEIGHTS['text_quality'] * 100
        score += quality_score
        breakdown['Text Quality'] = round(quality_score, 2)
        qa
        score = max(0, min(100, score))
        
        # Generate explanation
        explanation = self._generate_explanation_new(score, breakdown, aadhaar_valid)
        
        return ConfidenceScore(
            total_score=int(score),
            breakdown=breakdown,
            penalties={},
            explanation=explanation
        )
    
    def calculate(
        self,
        qr_detected: bool,
        qr_data: Optional[DemographicData],
        signature_result: Optional[SignatureResult],
        ocr_extracted: bool,
        cross_validation: Optional[CrossValidationResult],
        ocr_result: Optional[Dict[str, Any]] = None
    ) -> ConfidenceScore:
        
        # If we have OCR results, use new algorithm
        if ocr_result:
            base_score = self.calculate_ocr_based(ocr_result)
            
            # Add QR bonus if available
            if qr_detected:
                bonus = self.QR_BONUS
                base_score.breakdown['QR Detected Bonus'] = bonus
                base_score.total_score = min(100, base_score.total_score + bonus)
            
            # Add signature bonus if valid
            if signature_result and signature_result.is_valid:
                bonus = self.SIGNATURE_BONUS
                base_score.breakdown['Valid Signature Bonus'] = bonus
                base_score.total_score = min(100, base_score.total_score + bonus)
            
            return base_score
        
        # Fallback to legacy algorithm for backward compatibility
        return self._calculate_legacy(
            qr_detected, 
            qr_data, 
            signature_result, 
            ocr_extracted, 
            cross_validation
        )
    
    def _calculate_format_score(self, ocr_result: Dict[str, Any]) -> float:
        
        max_score = self.WEIGHTS['format_match'] * 100
        score = 0.0
        
        # Check each field format
        checks = 0
        matches = 0
        
        # Aadhaar number format (12 digits)
        if ocr_result.get('extraction_details', {}).get('aadhaar_extracted'):
            checks += 1
            if ocr_result.get('aadhaar_valid'):
                matches += 1
        
        # Name format (alphabetic)
        name = ocr_result.get('name')
        if name:
            checks += 1
            if len(name) > 2 and sum(c.isalpha() or c.isspace() for c in name) / len(name) > 0.7:
                matches += 1
        
        # DOB format
        dob = ocr_result.get('dob')
        if dob:
            checks += 1
            # Has date-like pattern
            if any(sep in dob for sep in ['/', '-', '.']):
                matches += 1
        
        # Gender format
        gender = ocr_result.get('gender')
        if gender:
            checks += 1
            if gender in ['MALE', 'FEMALE']:
                matches += 1
        
        # Calculate score
        if checks > 0:
            score = (matches / checks) * max_score
        
        return score
    
    def _calculate_completeness_score(self, ocr_result: Dict[str, Any]) -> float:
        
        max_score = self.WEIGHTS['field_completeness'] * 100
        
        extracted_count = 0
        total_fields = len(self.EXPECTED_FIELDS)
        
        for field in self.EXPECTED_FIELDS:
            value = ocr_result.get(field)
            if value:
                extracted_count += 1
        
        score = (extracted_count / total_fields) * max_score
        return score
    
    def _calculate_text_quality_score(self, metrics: Dict[str, float]) -> float:
        
        max_score = self.WEIGHTS['text_quality'] * 100
        
        # Expected ratios for Aadhaar cards
        # Name heavy (lots of alpha), some digits for Aadhaar number
        alpha_ratio = metrics.get('alpha_ratio', 0.5)
        digit_ratio = metrics.get('digit_ratio', 0.2)
        
        # Good balance: 40-70% alpha, 10-30% digits
        alpha_score = 1.0 if 0.4 <= alpha_ratio <= 0.7 else alpha_ratio
        digit_score = 1.0 if 0.1 <= digit_ratio <= 0.3 else digit_ratio
        
        quality = (alpha_score + digit_score) / 2
        return quality * max_score
    
    def _generate_explanation_new(
        self, 
        score: int, 
        breakdown: dict,
        checksum_valid: bool
    ) -> str:
        
        parts = []
        
        # Overall assessment
        if score >= 80:
            parts.append("HIGH CONFIDENCE: Aadhaar number validated with strong OCR.")
        elif score >= 60:
            parts.append("MEDIUM CONFIDENCE: Good OCR extraction with some validation.")
        elif score >= 30:
            parts.append("LOW-MEDIUM CONFIDENCE: OCR extraction successful but validation limited.")
        else:
            parts.append("VERY LOW CONFIDENCE: Poor OCR quality or critical data missing.")
        
        # Key factor
        if checksum_valid:
            parts.append("Aadhaar number passed Verhoeff checksum validation.")
        else:
            parts.append("Aadhaar number not validated (checksum failed or not extracted).")
        
        # Top contributors
        top_factors = sorted(breakdown.items(), key=lambda x: x[1], reverse=True)[:2]
        if top_factors:
            factor_names = [f"{name} ({val:.0f})" for name, val in top_factors]
            parts.append(f"Top factors: {', '.join(factor_names)}")
        
        return " ".join(parts)
    
    def _calculate_legacy(
        self,
        qr_detected: bool,
        qr_data: Optional[DemographicData],
        signature_result: Optional[SignatureResult],
        ocr_extracted: bool,
        cross_validation: Optional[CrossValidationResult]
    ) -> ConfidenceScore:
        
        logger.warning("Using deprecated legacy scoring algorithm. Consider using OCR-based scoring.")
        score = 0
        breakdown = {}
        penalties = {}
        cap = 100
        
        # Legacy weights
        LEGACY_WEIGHTS = {
            'qr_detected': 30,
            'signature_valid': 40,
            'ocr_extracted': 10,
            'ocr_strong': 20,
        }
        
        # QR Detection
        if qr_detected:
            score += LEGACY_WEIGHTS['qr_detected']
            breakdown['QR Code Detected'] = LEGACY_WEIGHTS['qr_detected']
        else:
            cap = 65
            breakdown['QR Code Detected'] = 0
        
        # Signature
        if signature_result and signature_result.is_valid:
            score += LEGACY_WEIGHTS['signature_valid']
            breakdown['Digital Signature Valid'] = LEGACY_WEIGHTS['signature_valid']
        else:
            breakdown['Digital Signature Valid'] = 0
        
        # OCR
        if ocr_extracted:
            score += LEGACY_WEIGHTS['ocr_extracted']
            breakdown['OCR Extraction'] = LEGACY_WEIGHTS['ocr_extracted']
            
            if not qr_detected:
                score += LEGACY_WEIGHTS['ocr_strong']
                breakdown['OCR Strong (No QR Available)'] = LEGACY_WEIGHTS['ocr_strong']
        else:
            breakdown['OCR Extraction'] = 0
        
        score = min(score, cap)
        score = max(0, min(100, score))
        
        return ConfidenceScore(
            total_score=int(score),
            breakdown=breakdown,
            penalties=penalties,
            explanation=self._generate_explanation_legacy(score, breakdown)
        )
    
    def _generate_explanation_legacy(self, score: int, breakdown: dict) -> str:
        
        if score >= 70:
            return "GOOD CONFIDENCE: Most verification checks passed."
        elif score >= 50:
            return "MODERATE CONFIDENCE: Some verification checks passed."
        elif score >= 30:
            return "LOW CONFIDENCE: Limited verification data available."
        else:
            return "VERY LOW CONFIDENCE: Verification largely failed."
    
    def get_verdict(self, score: int) -> str:
        
        if score >= 30:
            return "accepted"
        else:
            return "rejected"
    
    def get_verdict_description(self, verdict: str) -> str:
        
        descriptions = {
            "accepted": "The Aadhaar card has been verified and accepted.",
            "rejected": "The Aadhaar card failed verification checks and has been rejected."
        }
        return descriptions.get(verdict, "Unknown verification status.")


# Singleton instance
confidence_scorer = ConfidenceScorer()
