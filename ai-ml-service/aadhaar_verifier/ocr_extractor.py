import cv2
import numpy as np
import logging
from typing import Optional

from .models import OCRExtractedData

# Import new optimized extractor
try:
    from utils.aadhaar_extractor import AadhaarOCRExtractor
    NEW_EXTRACTOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"New AadhaarOCRExtractor not available: {e}")
    NEW_EXTRACTOR_AVAILABLE = False

logger = logging.getLogger(__name__)

# Use EasyOCR for high accuracy (~90-98%)
try:
    import easyocr
    EASYOCR_AVAILABLE = True
    logger.info("EasyOCR available (primary OCR engine)")
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.error("EasyOCR not available - install with: pip install easyocr")



class OCRExtractor:
    def __init__(self):
        if NEW_EXTRACTOR_AVAILABLE:
            self.extractor = AadhaarOCRExtractor()
            logger.info("Using optimized Aadhaar OCR extractor with EasyOCR backend")
        else:
            self.extractor = None
            logger.error("AadhaarOCRExtractor not available. Please check installation.")
    
    def extract(self, image: np.ndarray) -> OCRExtractedData:           
        
        if self.extractor:
            # Use new optimized pipeline
            return self._extract_with_new_pipeline(image)
        elif self.use_legacy:
            # Fallback to legacy extraction
            return self._extract_legacy(image)
        else:
            # No OCR available
            return OCRExtractedData(
                extraction_notes=["No OCR engine available"]
            )
    
    def _extract_with_new_pipeline(self, image: np.ndarray) -> OCRExtractedData:
        try:
            # Run new extraction pipeline
            result = self.extractor.extract_aadhaar_data(image, preprocess=True)
            
            if not result.get('success'):
                return OCRExtractedData(
                    extraction_notes=[f"Extraction failed: {result.get('error', 'Unknown')}"]
                )
            
                extraction_notes = []
            extraction_notes.append("New optimized OCR pipeline")
            
            if result.get('aadhaar_valid'):
                extraction_notes.append("Aadhaar number validated (Verhoeff checksum passed)")
            
            if result.get('preprocessing', {}).get('steps_applied'):
                steps = result['preprocessing']['steps_applied']
                extraction_notes.append(f"Preprocessing: {len(steps)} steps applied")
            
            details = result.get('extraction_details', {})
            fields_extracted =sum(details.values()) if details else 0
            extraction_notes.append(f"Extracted {fields_extracted}/5 fields")
            
            return OCRExtractedData(
                raw_text=result.get('full_text', ''),
                name=result.get('name'),
                dob=result.get('dob'),
                yob=None,   
                gender=result.get('gender'),
                masked_aadhaar=result.get('aadhaar_masked'),
                address=result.get('address'),
                vid=result.get('guardian'),  
                ocr_confidence=result.get('ocr_confidence', {}).get('mean', 0.0),
                extraction_notes=extraction_notes,
                _full_result=result
            )
            
        except Exception as e:
            logger.error(f"New OCR pipeline failed: {e}", exc_info=True)
            return OCRExtractedData(
                extraction_notes=[f"OCR extraction error: {str(e)}"]
            )
    




ocr_extractor = OCRExtractor()
