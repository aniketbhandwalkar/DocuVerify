"""
Aadhaar Data Extraction using EasyOCR
with Rule-Based Pattern Matching and Validation

This module handles:
- EasyOCR with deep learning for Aadhaar cards
- Rule-based extraction of Aadhaar numbers using regex patterns
- Name, DOB, Gender extraction with position heuristics
- Verhoeff checksum validation to eliminate false positives
"""

import cv2
import numpy as np
import easyocr
import re
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from utils.verhoeff_validator import validate_aadhaar, format_aadhaar, mask_aadhaar
from utils.aadhaar_preprocessor import AadhaarPreprocessor
from utils.ocr_config import get_ocr_config

logger = logging.getLogger(__name__)

# Initialize EasyOCR reader (lazy initialization)
_easyocr_reader = None

def get_easyocr_reader():
    """Get or initialize EasyOCR reader"""
    global _easyocr_reader
    if _easyocr_reader is None:
        logger.info("Initializing EasyOCR reader...")
        _easyocr_reader = easyocr.Reader(['en', 'hi'], gpu=False)
        logger.info("EasyOCR reader initialized")
    return _easyocr_reader


class AadhaarOCRExtractor:
    """
    Optimized OCR extraction for Aadhaar cards
    Uses EasyOCR with deep learning (90-98% accuracy)
    """
    
    # Regex patterns for Aadhaar number extraction (priority order)
    AADHAAR_PATTERNS = [
        r'\b(\d{4})\s+(\d{4})\s+(\d{4})\b',  # "1234 5678 9012" (most common)
        r'\b(\d{4})\s*-\s*(\d{4})\s*-\s*(\d{4})\b',  # "1234-5678-9012"
        r'\b(\d{12})\b',  # "123456789012" (no spaces)
        r'(\d{4})[\s\-]?(\d{4})[\s\-]?(\d{4})',  # Flexible separator
    ]
    
    # DOB patterns
    DOB_PATTERNS = [
        r'\b(\d{2})[\/\-\.](\d{2})[\/\-\.](\d{4})\b',  # DD/MM/YYYY or DD-MM-YYYY
        r'\b(\d{4})[\/\-\.](\d{2})[\/\-\.](\d{2})\b',  # YYYY/MM/DD
        r'DOB[:\s]*(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})',  # "DOB: DD/MM/YYYY"
        r'Year\s+of\s+Birth[:\s]*(\d{4})',  # "Year of Birth: YYYY"
        r'YOB[:\s]*(\d{4})',  # "YOB: YYYY"
    ]
    
    # Gender patterns
    GENDER_PATTERNS = [
        r'\b(MALE|FEMALE)\b',
        r'\b(M|F)\b(?!\w)',  # M or F not followed by word char
        r'(पुरुष|महिला)',  # Hindi: Male/Female
    ]
    
    def __init__(self):
        """Initialize the OCR extractor"""
        self.preprocessor = AadhaarPreprocessor()
        logger.info("AadhaarOCRExtractor initialized")
    
    def extract_aadhaar_data(
        self, 
        image: np.ndarray,
        preprocess: bool = True
    ) -> Dict[str, Any]:
        """
        Main extraction pipeline for Aadhaar card data
        
        Args:
            image: Input image (BGR from cv2.imread)
            preprocess: Whether to apply preprocessing (default True)
            
        Returns:
            Dictionary with extracted data and confidence scores
        """
        try:
            start_time = datetime.now()
            
            # Step 1: Preprocess image if requested
            if preprocess:
                processed_image, preprocess_info = self.preprocessor.preprocess_for_aadhaar_ocr(
                    image, 
                    enhance_numbers=True
                )
            else:
                processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
                preprocess_info = {'steps_applied': []}
            
            # Step 2: Extract text using EasyOCR
            full_text, full_confidence = self._extract_text_with_confidence(
                processed_image, 
                extract_numeric_only=False
            )
            
            # Step 3: Extract numeric text (optimized for Aadhaar number)
            numeric_text, numeric_confidence = self._extract_text_with_confidence(
                processed_image,
                extract_numeric_only=True
            )
            
            # Console logging - PRINT OCR RESULTS
            print(f"\n{'='*60}")
            print(f"[AADHAAR OCR EXTRACTION]")
            print(f"{'='*60}")
            print(f"Full Text Extracted: {full_text[:200]}")
            print(f"Numeric Text: {numeric_text[:100]}")
            print(f"Full Text Confidence: {full_confidence:.2%}")
            print(f"Numeric Confidence: {numeric_confidence:.2%}")
            print(f"Text Length: {len(full_text)} characters")
            print(f"{'='*60}\n")
            
            # Step 4: Rule-based extraction
            aadhaar_number = self._extract_aadhaar_number(full_text, numeric_text)
            dob = self._extract_dob(full_text)
            gender = self._extract_gender(full_text)
            name = self._extract_name(full_text)
            
            # Step 5: Validate Aadhaar number with Verhoeff checksum
            aadhaar_valid = False
            if aadhaar_number:
                aadhaar_valid = validate_aadhaar(aadhaar_number)
                if aadhaar_valid:
                    aadhaar_number = format_aadhaar(aadhaar_number)
            
            # Step 6: Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Compile results
            result = {
                'success': True,
                'aadhaar_number': aadhaar_number if aadhaar_valid else None,
                'aadhaar_valid': aadhaar_valid,
                'aadhaar_masked': mask_aadhaar(aadhaar_number) if aadhaar_number else None,
                'name': name,
                'dob': dob,
                'gender': gender,
                'full_text': full_text[:500],  # Limit to 500 chars
                'ocr_confidence': {
                    'full_text': full_confidence,
                    'numeric': numeric_confidence,
                    'mean': (full_confidence + numeric_confidence) / 2
                },
                'preprocessing': preprocess_info,
                'processing_time_ms': processing_time,
                'extraction_details': {
                    'aadhaar_extracted': aadhaar_number is not None,
                    'checksum_validated': aadhaar_valid,
                    'name_extracted': name is not None,
                    'dob_extracted': dob is not None,
                    'gender_extracted': gender is not None,
                }
            }
            
            logger.info(f"Extraction completed in {processing_time:.1f}ms | "
                       f"Aadhaar: {'✓' if aadhaar_valid else '✗'} | "
                       f"Fields: {sum(result['extraction_details'].values())}/5")
            
            return result
            
        except Exception as e:
            logger.error(f"Aadhaar extraction error: {e}")
            return {
                'success': False,
                'error': str(e),
                'aadhaar_number': None,
                'aadhaar_valid': False
            }
    
    def _extract_text_with_confidence(
        self, 
        image: np.ndarray, 
        extract_numeric_only: bool = False
    ) -> Tuple[str, float]:
        """
        Extract text using EasyOCR with confidence calculation
        
        Args:
            image: Preprocessed grayscale/binary image
            extract_numeric_only: If True, only return numeric characters
            
        Returns:
            Tuple of (extracted_text, mean_confidence)
        """
        try:
            reader = get_easyocr_reader()
            
            # Ensure image is in BGR format for EasyOCR
            if len(image.shape) == 2:
                # Grayscale to BGR
                image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                image_bgr = image
            
            # Extract text with EasyOCR
            results = reader.readtext(image_bgr, detail=1)  # detail=1 for confidence
            
            if not results:
                return "", 0.0
            
            # Compile text and confidence
            texts = []
            confidences = []
            
            for (bbox, text, confidence) in results:
                texts.append(text)
                confidences.append(confidence)
            
            # If numeric only, filter
            if extract_numeric_only:
                texts = [''.join(c for c in t if c.isdigit()) for t in texts]
                texts = [t for t in texts if t]  # Remove empty strings
            
            full_text = ' '.join(texts)
            mean_confidence = np.mean(confidences) if confidences else 0.0
            
            return full_text, mean_confidence
            
        except Exception as e:
            logger.warning(f"EasyOCR extraction failed: {e}")
            return "", 0.0
    
    def _extract_aadhaar_number(
        self, 
        full_text: str, 
        numeric_text: str
    ) -> Optional[str]:
        """
        Extract Aadhaar number using regex patterns and validation
        
        Args:
            full_text: Full OCR text
            numeric_text: Numeric-only OCR text
            
        Returns:
            Aadhaar number if found and valid, None otherwise
        """
        try:
            # Try patterns on both text sources
            text_sources = [full_text, numeric_text]
            
            for text in text_sources:
                for pattern in self.AADHAAR_PATTERNS:
                    matches = re.finditer(pattern, text)
                    
                    for match in matches:
                        # Extract and normalize the number
                        if len(match.groups()) == 3:
                            # Pattern with groups (e.g., "1234 5678 9012")
                            number = ''.join(match.groups())
                        else:
                            # Pattern without groups (e.g., "123456789012")
                            number = match.group(0).replace(' ', '').replace('-', '')
                        
                        # Validate format
                        if len(number) == 12 and number.isdigit():
                            # Check Verhoeff checksum
                            if validate_aadhaar(number):
                                logger.debug(f"Valid Aadhaar found: {number[:4]}********")
                                return number
                            else:
                                logger.debug(f"Invalid checksum for: {number[:4]}********")
            
            logger.warning("No valid Aadhaar number found in OCR text")
            return None
            
        except Exception as e:
            logger.error(f"Aadhaar number extraction error: {e}")
            return None
    
    def _extract_dob(self, text: str) -> Optional[str]:
        """
        Extract date of birth from text
        
        Args:
            text: OCR text
            
        Returns:
            DOB string if found, None otherwise
        """
        try:
            for pattern in self.DOB_PATTERNS:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    dob_str = match.group(1) if match.lastindex >= 1 else match.group(0)
                    logger.debug(f"DOB extracted: {dob_str}")
                    return dob_str
            
            return None
            
        except Exception as e:
            logger.warning(f"DOB extraction error: {e}")
            return None
    
    def _extract_gender(self, text: str) -> Optional[str]:
        """
        Extract gender from text
        
        Args:
            text: OCR text
            
        Returns:
            Gender string ('MALE' or 'FEMALE') if found, None otherwise
        """
        try:
            for pattern in self.GENDER_PATTERNS:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    gender = match.group(1).upper()
                    
                    # Normalize
                    if gender == 'M' or gender == 'पुरुष':
                        gender = 'MALE'
                    elif gender == 'F' or gender == 'महिला':
                        gender = 'FEMALE'
                    
                    logger.debug(f"Gender extracted: {gender}")
                    return gender
            
            return None
            
        except Exception as e:
            logger.warning(f"Gender extraction error: {e}")
            return None
    
    def _extract_name(self, text: str) -> Optional[str]:
        """
        Extract name from text using heuristics
        
        Args:
            text: OCR text
            
        Returns:
            Name string if found, None otherwise
        """
        try:
            lines = text.split('\n')
            
            # Name is typically:
            # 1. In the first few lines
            # 2. Longer than 3 characters
            # 3. Mostly alphabetic
            # 4. Not DOB/Gender/Aadhaar number
            
            for i, line in enumerate(lines[:10]):  # Check first 10 lines
                line = line.strip()
                
                # Skip short lines
                if len(line) < 3:
                    continue
                
                # Skip if contains too many digits
                digit_ratio = sum(c.isdigit() for c in line) / len(line)
                if digit_ratio > 0.3:
                    continue
                
                # Skip common labels
                skip_keywords = ['dob', 'male', 'female', 'india', 'government', 
                               'uidai', 'aadhaar', 'address', 'year', 'birth']
                if any(keyword in line.lower() for keyword in skip_keywords):
                    continue
                
                # Check if mostly alphabetic
                alpha_ratio = sum(c.isalpha() or c.isspace() for c in line) / len(line)
                if alpha_ratio > 0.7:
                    logger.debug(f"Name extracted: {line}")
                    return line
            
            return None
            
        except Exception as e:
            logger.warning(f"Name extraction error: {e}")
            return None
    
    def extract_aadhaar_number_only(
        self, 
        image: np.ndarray
    ) -> Dict[str, Any]:
        """
        Fast extraction of only Aadhaar number (for quick validation)
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with aadhaar_number and validity
        """
        try:
            # Quick preprocessing
            processed, _ = self.preprocessor.preprocess_for_aadhaar_ocr(image, enhance_numbers=True)
            
            # Numeric extraction only
            numeric_text, confidence = self._extract_text_with_confidence(
                processed, 
                self.CONFIG_NUMERIC
            )
            
            # Extract number
            aadhaar_number = self._extract_aadhaar_number(numeric_text, numeric_text)
            
            # Validate
            valid = validate_aadhaar(aadhaar_number) if aadhaar_number else False
            
            return {
                'aadhaar_number': format_aadhaar(aadhaar_number) if valid else None,
                'aadhaar_valid': valid,
                'ocr_confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"Quick extraction error: {e}")
            return {
                'aadhaar_number': None,
                'aadhaar_valid': False,
                'error': str(e)
            }


if __name__ == "__main__":
    # Test the extractor
    import sys
    
    logging.basicConfig(level=logging.DEBUG)
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        print(f"\n=== Testing Aadhaar OCR Extractor ===")
        print(f"Image: {image_path}\n")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            sys.exit(1)
        
        # Extract
        extractor = AadhaarOCRExtractor()
        result = extractor.extract_aadhaar_data(image)
        
        # Display results
        print(f"Success: {result['success']}")
        print(f"Aadhaar Number: {result.get('aadhaar_masked', 'Not found')}")
        print(f"Valid: {result.get('aadhaar_valid', False)}")
        print(f"Name: {result.get('name', 'Not found')}")
        print(f"DOB: {result.get('dob', 'Not found')}")
        print(f"Gender: {result.get('gender', 'Not found')}")
        print(f"OCR Confidence: {result.get('ocr_confidence', {}).get('mean', 0):.2%}")
        print(f"Processing Time: {result.get('processing_time_ms', 0):.1f}ms")
        print(f"\nExtracted Text Preview:")
        print(result.get('full_text', '')[:200])
    else:
        print("Usage: python aadhaar_extractor.py <image_path>")
