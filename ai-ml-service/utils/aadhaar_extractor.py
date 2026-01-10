

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
    
    global _easyocr_reader
    if _easyocr_reader is None:
        logger.info("Initializing EasyOCR reader...")
        _easyocr_reader = easyocr.Reader(['en', 'hi'], gpu=False)
        logger.info("EasyOCR reader initialized")
    return _easyocr_reader


class AadhaarOCRExtractor:
    
    
    # Regex patterns for Aadhaar number extraction (priority order)
    AADHAAR_PATTERNS = [
        # Look for 12 digits with spaces (must be exactly 12 total digits if separated by spaces)
        r'\b(\d{4})\s+(\d{4})\s+(\d{4})\b',  # "1234 5678 9012"
        # Avoid picking up VID (16 digits)
        r'(?<!VID[:\s])\b(\d{12})\b',  # 12 digits not preceded by VID
        r'\b(\d{4})\s*-\s*(\d{4})\s*-\s*(\d{4})\b',  # "1234-5678-9012"
        r'(\d{4})[\s\-]?(\d{4})[\s\-]?(\d{4})',  # Flexible separator
    ]
    
    # DOB patterns
    DOB_PATTERNS = [
        r'\b(\d{2})[\/\-\.](\d{2})[\/\-\.](\d{4})\b',  # DD/MM/YYYY
        r'\b(\d{4})[\/\-\.](\d{2})[\/\-\.](\d{2})\b',  # YYYY/MM/DD
        r'(?:DOB|जन्म तिथि|Year of Birth)[:\s]*(\d{2}[\/\-\.]\d{2}[\/\-\.]\d{4})',
        r'(?:DOB|जन्म तिथि|Year of Birth)[:\s]*(\d{4})',
    ]
    
    # Gender patterns
    GENDER_PATTERNS = [
        r'\b(MALE|FEMALE|TRANSGENDER)\b',
        r'\b(M|F|T)\b(?!\w)',
        r'(पुरुष|महिला|अन्य)',  # Hindi: Male/Female/Other
    ]
    
    def __init__(self):
        
        self.preprocessor = AadhaarPreprocessor()
        logger.info("AadhaarOCRExtractor initialized")
    
    def extract_aadhaar_data(
        self, 
        image: np.ndarray,
        preprocess: bool = True
    ) -> Dict[str, Any]:
        
        try:
            start_time = datetime.now()
            
            # Step 2-4: Run extraction with fallback rotations
            result = self._extract_with_rotation_fallback(image, preprocess)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            result['processing_time_ms'] = processing_time
            
            return result
            
        except Exception as e:
            logger.error(f"Aadhaar extraction error: {e}")
            return {
                'success': False,
                'error': str(e),
                'aadhaar_number': None,
                'aadhaar_valid': False
            }

    def _extract_with_rotation_fallback(
        self, 
        image: np.ndarray, 
        preprocess: bool
    ) -> Dict[str, Any]:
        
        best_result = None
        
        # Try rotations: 0, 90, 180, 270
        for angle in [0, 90, 180, 270]:
            if angle == 0:
                rotated = image
            else:
                # Rotate clockwise
                if angle == 90:
                    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                elif angle == 180:
                    rotated = cv2.rotate(image, cv2.ROTATE_180)
                elif angle == 270:
                    rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            # Preprocess rotated image if requested
            if preprocess:
                processed_image, preprocess_info = self.preprocessor.preprocess_for_aadhaar_ocr(
                    rotated, 
                    enhance_numbers=True
                )
            else:
                processed_image = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY) if len(rotated.shape) == 3 else rotated
                preprocess_info = {'steps_applied': []}
            
            # Extract text
            full_text, full_confidence = self._extract_text_with_confidence(processed_image)
            numeric_text, numeric_confidence = self._extract_text_with_confidence(processed_image, extract_numeric_only=True)
            
            # Rule-based extraction
            aadhaar_number = self._extract_aadhaar_number(full_text, numeric_text)
            aadhaar_valid = validate_aadhaar(aadhaar_number) if aadhaar_number else False
            
            # Compiling result for this rotation
            current_result = {
                'success': True,
                'aadhaar_number': format_aadhaar(aadhaar_number) if aadhaar_valid else None,
                'aadhaar_valid': aadhaar_valid,
                'aadhaar_masked': mask_aadhaar(aadhaar_number) if aadhaar_number else None,
                'name': self._extract_name(full_text),
                'dob': self._extract_dob(full_text),
                'gender': self._extract_gender(full_text),
                'address': self._extract_address(full_text),
                'guardian': self._extract_guardian(full_text),
                'full_text': full_text[:1000],
                'ocr_confidence': {
                    'full_text': full_confidence,
                    'numeric': numeric_confidence,
                    'mean': (full_confidence + numeric_confidence) / 2
                },
                'rotation_angle': angle,
                'extraction_details': {
                    'aadhaar_extracted': aadhaar_number is not None,
                    'checksum_validated': aadhaar_valid
                }
            }
            
            # If we found a valid Aadhaar, return immediately
            if aadhaar_valid:
                logger.info(f"Valid Aadhaar found at {angle} degrees rotation")
                return current_result
                
            # Keep the result with highest confidence/most fields as backup
            if best_result is None or current_result['ocr_confidence']['mean'] > best_result['ocr_confidence']['mean']:
                best_result = current_result
                
        return best_result
    
    def _extract_text_with_confidence(
        self, 
        image: np.ndarray, 
        extract_numeric_only: bool = False
    ) -> Tuple[str, float]:
        
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
    
    def _normalize_aadhaar_chars(self, text: str) -> str:
        
        replacements = {
            'O': '0', 'o': '0',
            'I': '1', 'i': '1', 'l': '1', '|': '1',
            'S': '5', 's': '5',
            'B': '8', 'b': '8',
            'Z': '2', 'z': '2',
            'G': '6', 'g': '6',
            'T': '7', 't': '7'
        }
        for char, digit in replacements.items():
            text = text.replace(char, digit)
        return text

    def _extract_aadhaar_number(
        self, 
        full_text: str, 
        numeric_text: str
    ) -> Optional[str]:
        
        try:
            # Try raw texts first, then corrected versions
            text_sources = [
                full_text, 
                numeric_text,
                self._normalize_aadhaar_chars(full_text)
            ]
            
            for text in text_sources:
                # Pre-clean text: handle "VID"
                clean_text = re.sub(r'VID[:\s]*\d{4}\s*\d{4}\s*\d{4}\s*\d{4}', ' [VID_MASKED] ', text, flags=re.IGNORECASE)
                
                for pattern in self.AADHAAR_PATTERNS:
                    matches = re.finditer(pattern, clean_text)
                    for match in matches:
                        if len(match.groups()) == 3:
                            number = ''.join(match.groups())
                        else:
                            number = match.group(0).replace(' ', '').replace('-', '')
                        
                        if len(number) == 12 and number.isdigit():
                            if validate_aadhaar(number):
                                logger.info(f"Valid Aadhaar found in OCR: {number[:4]}********")
                                return number
            
            return None
            
        except Exception as e:
            logger.error(f"Aadhaar number extraction error: {e}")
            return None
    
    def _extract_dob(self, text: str) -> Optional[str]:
        
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
        
        try:
            # Clean text by removing noise characters
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            
            # Heuristic 1: Look for strings like "Father:" or "Care of" to identify surrounding context
            # Names are usually above DOB and Gender on the front side
            
            for i, line in enumerate(lines[:12]):
                # Skip lines that are purely government labels
                low_line = line.lower()
                if any(x in low_line for x in ['government', 'india', 'भारत', 'सरकार', 'unique identification']):
                    continue
                
                # Aadhaar names are usually ALL CAPS in English
                # and are not labels like "DOB", "Male", etc.
                if len(line) < 3 or any(d.isdigit() for d in line):
                    continue
                
                skip_keywords = ['dob', 'male', 'female', 'india', 'government', 
                               'uidai', 'aadhaar', 'address', 'year', 'birth', 'पिता', 'पति']
                if any(k in low_line for k in skip_keywords):
                    continue
                
                # Check if it's mostly uppercase or title case
                if line.isupper() or (line.istitle() and len(line.split()) >= 2):
                    return line
            
            return None
            
        except Exception as e:
            logger.warning(f"Name extraction error: {e}")
            return None

    def _extract_address(self, text: str) -> Optional[str]:
        
        try:
            # Look for markers like "Address:", "पता:", "C/O", "S/O"
            # Address usually starts after these and continues for several lines
            markers = [r'Address[:\s]*(.*)', r'पता[:\s]*(.*)', r'S/O[:\s]*(.*)', r'D/O[:\s]*(.*)', r'W/O[:\s]*(.*)', r'C/O[:\s]*(.*)']
            
            for marker in markers:
                match = re.search(marker, text, re.IGNORECASE | re.DOTALL)
                if match:
                    addr = match.group(1).strip()
                    # Clean up: stop at big gaps or common footer keywords
                    footer_keywords = ['1947', 'help@uidai', 'www.uidai']
                    for k in footer_keywords:
                        if k in addr.lower():
                            addr = addr[:addr.lower().find(k)].strip()
                    return addr if len(addr) > 5 else None
            return None
        except Exception:
            return None

    def _extract_guardian(self, text: str) -> Optional[str]:
        
        try:
            patterns = [
                r'(?:Father|पिता|Husband|पति)[:\s]*([^,\n\r]+)',
                r'[S|D|W|C]/O[:\s]*([^,\n\r]+)'
            ]
            for p in patterns:
                match = re.search(p, text, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
            return None
        except Exception:
            return None
    
    def extract_aadhaar_number_only(
        self, 
        image: np.ndarray
    ) -> Dict[str, Any]:
        
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
