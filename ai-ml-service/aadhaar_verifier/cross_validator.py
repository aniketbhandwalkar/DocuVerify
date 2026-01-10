

import logging
from typing import Optional
from difflib import SequenceMatcher

from .models import (
    DemographicData,
    OCRExtractedData,
    CrossValidationResult,
    FieldMatch
)

logger = logging.getLogger(__name__)


class CrossValidator:
    
    
    # Thresholds for fuzzy matching
    NAME_MATCH_THRESHOLD = 0.7  # 70% similarity
    STRICT_MATCH_THRESHOLD = 0.9  # 90% similarity
    
    def validate(
        self, 
        qr_data: Optional[DemographicData], 
        ocr_data: Optional[OCRExtractedData]
    ) -> CrossValidationResult:
        
        field_matches = []
        mismatches = []
        warnings = []
        
        # Handle missing data cases
        if not qr_data and not ocr_data:
            return CrossValidationResult(
                overall_match=False,
                overall_score=0.0,
                field_matches=[],
                mismatches=["No data available for comparison"],
                warnings=["Both QR and OCR extraction failed"]
            )
        
        if not qr_data:
            return CrossValidationResult(
                overall_match=False,
                overall_score=0.0,
                field_matches=[],
                mismatches=[],
                warnings=["QR data not available - cannot cross-validate. OCR data is untrusted."]
            )
        
        if not ocr_data or not ocr_data.raw_text:
            return CrossValidationResult(
                overall_match=True,  # QR data is available and trusted
                overall_score=0.5,  # Partial score
                field_matches=[],
                mismatches=[],
                warnings=["OCR extraction failed - using QR data only (if signature valid)"]
            )
        
        # Compare individual fields
        
        # 1. Name comparison
        name_match = self._compare_names(qr_data.name, ocr_data.name)
        field_matches.append(name_match)
        if not name_match.is_match and name_match.qr_value and name_match.ocr_value:
            mismatches.append(f"Name mismatch: QR='{name_match.qr_value}' vs OCR='{name_match.ocr_value}'")
        
        # 2. DOB comparison
        dob_match = self._compare_dates(qr_data.dob or qr_data.yob, ocr_data.dob or ocr_data.yob)
        field_matches.append(dob_match)
        if not dob_match.is_match and dob_match.qr_value and dob_match.ocr_value:
            mismatches.append(f"DOB mismatch: QR='{dob_match.qr_value}' vs OCR='{dob_match.ocr_value}'")
        
        # 3. Gender comparison
        gender_match = self._compare_gender(qr_data.gender, ocr_data.gender)
        field_matches.append(gender_match)
        if not gender_match.is_match and gender_match.qr_value and gender_match.ocr_value:
            mismatches.append(f"Gender mismatch: QR='{gender_match.qr_value}' vs OCR='{gender_match.ocr_value}'")
        
        # 4. Masked Aadhaar comparison
        aadhaar_match = self._compare_masked_aadhaar(qr_data.masked_aadhaar, ocr_data.masked_aadhaar)
        field_matches.append(aadhaar_match)
        if not aadhaar_match.is_match and aadhaar_match.qr_value and aadhaar_match.ocr_value:
            mismatches.append(f"Aadhaar mismatch: QR='{aadhaar_match.qr_value}' vs OCR='{aadhaar_match.ocr_value}'")
        
        # Calculate overall score
        matched_fields = [f for f in field_matches if f.is_match]
        comparable_fields = [f for f in field_matches if f.qr_value and f.ocr_value]
        
        if comparable_fields:
            overall_score = len(matched_fields) / len(comparable_fields)
        else:
            overall_score = 0.5  # No fields to compare
            warnings.append("Limited fields available for comparison")
        
        # Generate warnings
        if overall_score < 1.0 and overall_score > 0.5:
            warnings.append("Some fields don't match - card may be partially modified")
        
        return CrossValidationResult(
            overall_match=len(mismatches) == 0 and overall_score >= 0.5,
            overall_score=round(overall_score, 2),
            field_matches=field_matches,
            mismatches=mismatches,
            warnings=warnings
        )
    
    def _compare_names(self, qr_name: Optional[str], ocr_name: Optional[str]) -> FieldMatch:
        
        if not qr_name or not ocr_name:
            return FieldMatch(
                field_name="Name",
                qr_value=qr_name,
                ocr_value=ocr_name,
                is_match=False,
                match_score=0.0,
                notes="One or both names missing"
            )
        
        # Normalize names
        qr_normalized = self._normalize_name(qr_name)
        ocr_normalized = self._normalize_name(ocr_name)
        
        # Calculate similarity
        similarity = SequenceMatcher(None, qr_normalized, ocr_normalized).ratio()
        
        # Check for partial matches (e.g., name order differences)
        qr_parts = set(qr_normalized.split())
        ocr_parts = set(ocr_normalized.split())
        common_parts = qr_parts.intersection(ocr_parts)
        
        if common_parts:
            partial_match = len(common_parts) / max(len(qr_parts), len(ocr_parts))
            similarity = max(similarity, partial_match)
        
        is_match = similarity >= self.NAME_MATCH_THRESHOLD
        
        return FieldMatch(
            field_name="Name",
            qr_value=qr_name,
            ocr_value=ocr_name,
            is_match=is_match,
            match_score=round(similarity, 2),
            notes=f"Fuzzy match: {similarity:.0%}"
        )
    
    def _normalize_name(self, name: str) -> str:
        
        # Convert to lowercase
        name = name.lower()
        # Remove extra whitespace
        name = ' '.join(name.split())
        # Remove common prefixes/suffixes
        for prefix in ['mr.', 'mrs.', 'ms.', 'dr.', 'shri', 'smt']:
            if name.startswith(prefix + ' '):
                name = name[len(prefix) + 1:]
        return name.strip()
    
    def _compare_dates(self, qr_date: Optional[str], ocr_date: Optional[str]) -> FieldMatch:
        
        if not qr_date or not ocr_date:
            return FieldMatch(
                field_name="DOB/YOB",
                qr_value=qr_date,
                ocr_value=ocr_date,
                is_match=False,
                match_score=0.0,
                notes="One or both dates missing"
            )
        
        # Normalize dates
        qr_normalized = self._normalize_date(qr_date)
        ocr_normalized = self._normalize_date(ocr_date)
        
        # Check for exact match
        if qr_normalized == ocr_normalized:
            return FieldMatch(
                field_name="DOB/YOB",
                qr_value=qr_date,
                ocr_value=ocr_date,
                is_match=True,
                match_score=1.0,
                notes="Exact match"
            )
        
        # Check if year matches (more lenient)
        qr_year = self._extract_year(qr_date)
        ocr_year = self._extract_year(ocr_date)
        
        if qr_year and ocr_year and qr_year == ocr_year:
            return FieldMatch(
                field_name="DOB/YOB",
                qr_value=qr_date,
                ocr_value=ocr_date,
                is_match=True,
                match_score=0.8,
                notes="Year matches"
            )
        
        return FieldMatch(
            field_name="DOB/YOB",
            qr_value=qr_date,
            ocr_value=ocr_date,
            is_match=False,
            match_score=0.0,
            notes="Date mismatch"
        )
    
    def _normalize_date(self, date: str) -> str:
        
        import re
        # Remove non-alphanumeric
        date = re.sub(r'[^0-9a-zA-Z]', '', date.lower())
        return date
    
    def _extract_year(self, date: str) -> Optional[str]:
        
        import re
        match = re.search(r'(19|20)\d{2}', date)
        return match.group(0) if match else None
    
    def _compare_gender(self, qr_gender: Optional[str], ocr_gender: Optional[str]) -> FieldMatch:
        
        if not qr_gender or not ocr_gender:
            return FieldMatch(
                field_name="Gender",
                qr_value=qr_gender,
                ocr_value=ocr_gender,
                is_match=False,
                match_score=0.0,
                notes="One or both genders missing"
            )
        
        # Normalize to single character
        qr_normalized = qr_gender[0].upper() if qr_gender else ''
        ocr_normalized = ocr_gender[0].upper() if ocr_gender else ''
        
        is_match = qr_normalized == ocr_normalized
        
        return FieldMatch(
            field_name="Gender",
            qr_value=qr_gender,
            ocr_value=ocr_gender,
            is_match=is_match,
            match_score=1.0 if is_match else 0.0,
            notes="Match" if is_match else "Mismatch"
        )
    
    def _compare_masked_aadhaar(
        self, 
        qr_aadhaar: Optional[str], 
        ocr_aadhaar: Optional[str]
    ) -> FieldMatch:
        
        if not qr_aadhaar or not ocr_aadhaar:
            return FieldMatch(
                field_name="Masked Aadhaar",
                qr_value=qr_aadhaar,
                ocr_value=ocr_aadhaar,
                is_match=False,
                match_score=0.0,
                notes="One or both Aadhaar numbers missing"
            )
        
        # Extract last 4 digits
        import re
        qr_last4 = re.search(r'(\d{4})$', qr_aadhaar.replace('-', '').replace(' ', ''))
        ocr_last4 = re.search(r'(\d{4})$', ocr_aadhaar.replace('-', '').replace(' ', ''))
        
        if not qr_last4 or not ocr_last4:
            return FieldMatch(
                field_name="Masked Aadhaar",
                qr_value=qr_aadhaar,
                ocr_value=ocr_aadhaar,
                is_match=False,
                match_score=0.0,
                notes="Could not extract last 4 digits"
            )
        
        is_match = qr_last4.group(1) == ocr_last4.group(1)
        
        return FieldMatch(
            field_name="Masked Aadhaar",
            qr_value=qr_aadhaar,
            ocr_value=ocr_aadhaar,
            is_match=is_match,
            match_score=1.0 if is_match else 0.0,
            notes="Last 4 digits match" if is_match else "Last 4 digits mismatch"
        )


# Singleton instance
cross_validator = CrossValidator()
