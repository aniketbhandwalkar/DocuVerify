"""
QR Code Detector and Decoder for Aadhaar Cards

Handles detection and decoding of both Secure QR (V2) and Old XML-based QR codes.
Extracts embedded demographic data from Aadhaar QR codes.
"""

import cv2
import numpy as np
import zlib
import struct
import logging
from typing import Optional, Tuple, Union
import base64

from .models import DemographicData, QRType

logger = logging.getLogger(__name__)

# Try importing pyzbar - DISABLED due to Windows DLL issues
# Using OpenCV QRCodeDetector instead
PYZBAR_AVAILABLE = False
logger.warning("pyzbar disabled - using OpenCV QRCodeDetector for QR detection")

# Try importing pyaadhaar - DISABLED (depends on pyzbar which has DLL issues)
# Using manual parsing instead
PYAADHAAR_AVAILABLE = False
logger.warning("pyaadhaar disabled (depends on pyzbar) - using manual parsing")


class QRDetector:
    """
    Detects and decodes Aadhaar QR codes from images.
    Supports both Secure QR (V2) and legacy XML-based QR codes.
    """
    
    # Secure QR V2 structure constants
    SIGNATURE_LENGTH = 256  # 2048-bit RSA signature
    VERSION_BYTE = 1
    
    def __init__(self):
        self.cv_detector = cv2.QRCodeDetector()
    
    def detect_and_decode(self, image: np.ndarray) -> Tuple[Optional[DemographicData], bytes]:
        """
        Detect and decode QR code from image.
        
        Args:
            image: Preprocessed image (numpy array)
            
        Returns:
            Tuple of (DemographicData or None, raw_qr_bytes)
        """
        raw_data = b""
        
        # Strategy 1: Try OpenCV detector first (no DLL dependencies)
        result, raw_data = self._detect_with_opencv(image)
        if result:
            return result, raw_data
        
        # Strategy 2: Try with enhanced preprocessing
        result, raw_data = self._detect_with_preprocessing(image)
        if result:
            return result, raw_data
        
        # Strategy 3: Try pyzbar as fallback (if available)
        if PYZBAR_AVAILABLE:
            result, raw_data = self._detect_with_pyzbar(image)
            if result:
                return result, raw_data
        
        logger.warning("No QR code detected with any method")
        return None, b""
    
    def _detect_with_pyzbar(self, image: np.ndarray) -> Tuple[Optional[DemographicData], bytes]:
        """Detect QR using pyzbar library."""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            decoded_objects = pyzbar.decode(gray)
            
            for obj in decoded_objects:
                if obj.type == 'QRCODE':
                    raw_data = obj.data
                    logger.info(f"pyzbar detected QR code, data length: {len(raw_data)}")
                    
                    demographic_data = self._parse_qr_data(raw_data)
                    if demographic_data:
                        return demographic_data, raw_data
            
            return None, b""
            
        except Exception as e:
            logger.warning(f"pyzbar detection failed: {e}")
            return None, b""
    
    def _detect_with_opencv(self, image: np.ndarray) -> Tuple[Optional[DemographicData], bytes]:
        """Detect QR using OpenCV detector with enhanced preprocessing."""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Try original image first
            data, points, straight_qr = self.cv_detector.detectAndDecode(image)
            
            if data:
                logger.info(f"OpenCV detected QR code (original), data length: {len(data)}")
                raw_data = data.encode('utf-8') if isinstance(data, str) else data
                demographic_data = self._parse_qr_data(raw_data)
                if demographic_data:
                    return demographic_data, raw_data
            
            # Try with grayscale
            data, points, straight_qr = self.cv_detector.detectAndDecode(gray)
            if data:
                logger.info(f"OpenCV detected QR code (grayscale), data length: {len(data)}")
                raw_data = data.encode('utf-8') if isinstance(data, str) else data
                demographic_data = self._parse_qr_data(raw_data)
                if demographic_data:
                    return demographic_data, raw_data
            
            # Try with enhanced contrast (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            data, points, straight_qr = self.cv_detector.detectAndDecode(enhanced)
            if data:
                logger.info(f"OpenCV detected QR code (CLAHE enhanced), data length: {len(data)}")
                raw_data = data.encode('utf-8') if isinstance(data, str) else data
                demographic_data = self._parse_qr_data(raw_data)
                if demographic_data:
                    return demographic_data, raw_data
            
            # Try with sharpening
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(gray, -1, kernel)
            data, points, straight_qr = self.cv_detector.detectAndDecode(sharpened)
            if data:
                logger.info(f"OpenCV detected QR code (sharpened), data length: {len(data)}")
                raw_data = data.encode('utf-8') if isinstance(data, str) else data
                demographic_data = self._parse_qr_data(raw_data)
                if demographic_data:
                    return demographic_data, raw_data
            
            # Try with increased size (upscale for small QR codes)
            h, w = gray.shape[:2]
            if w < 800:
                scale = 800 / w
                upscaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                data, points, straight_qr = self.cv_detector.detectAndDecode(upscaled)
                if data:
                    logger.info(f"OpenCV detected QR code (upscaled), data length: {len(data)}")
                    raw_data = data.encode('utf-8') if isinstance(data, str) else data
                    demographic_data = self._parse_qr_data(raw_data)
                    if demographic_data:
                        return demographic_data, raw_data
            
            logger.info("OpenCV QR detection: No QR code found after all preprocessing attempts")
            return None, b""
            
        except Exception as e:
            logger.warning(f"OpenCV detection failed: {e}")
            return None, b""
    
    def _detect_with_preprocessing(self, image: np.ndarray) -> Tuple[Optional[DemographicData], bytes]:
        """Try detection with additional preprocessing steps."""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Try different preprocessing techniques
            preprocessing_methods = [
                lambda img: cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1],
                lambda img: cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                   cv2.THRESH_BINARY, 11, 2),
                lambda img: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            ]
            
            for preprocess in preprocessing_methods:
                processed = preprocess(gray)
                
                # Try OpenCV on processed image first
                data, _, _ = self.cv_detector.detectAndDecode(processed)
                if data:
                    raw_data = data.encode('utf-8') if isinstance(data, str) else data
                    demographic_data = self._parse_qr_data(raw_data)
                    if demographic_data:
                        logger.info("QR detected with OpenCV preprocessing")
                        return demographic_data, raw_data
                
                # Try pyzbar as fallback (if available)
                if PYZBAR_AVAILABLE:
                    decoded_objects = pyzbar.decode(processed)
                    for obj in decoded_objects:
                        if obj.type == 'QRCODE':
                            raw_data = obj.data
                            demographic_data = self._parse_qr_data(raw_data)
                            if demographic_data:
                                logger.info("QR detected with pyzbar preprocessing")
                                return demographic_data, raw_data
            
            return None, b""
            
        except Exception as e:
            logger.warning(f"Preprocessing-based detection failed: {e}")
            return None, b""
    
    def _parse_qr_data(self, raw_data: Union[bytes, str]) -> Optional[DemographicData]:
        """
        Parse QR data and extract demographic information.
        Handles both Secure QR (V2) and Old XML QR formats.
        """
        if isinstance(raw_data, str):
            raw_data = raw_data.encode('utf-8')
        
        # Try to determine QR type
        data_str = raw_data.decode('utf-8', errors='ignore')
        
        # Check for XML format (old QR)
        if '<?xml' in data_str.lower() or '<qr' in data_str.lower():
            return self._parse_old_xml_qr(data_str)
        
        # Check if it's numeric (Secure QR V2)
        if raw_data.isdigit() or data_str.strip().isdigit():
            return self._parse_secure_qr_v2(data_str.strip())
        
        # Try parsing as Secure QR bytes
        if len(raw_data) > 300:
            return self._parse_secure_qr_bytes(raw_data)
        
        logger.warning(f"Unknown QR format, data starts with: {data_str[:50]}")
        return None
    
    def _parse_secure_qr_v2(self, numeric_string: str) -> Optional[DemographicData]:
        """
        Parse Secure QR V2 format (large numeric string).
        
        The Secure QR contains:
        - Reference ID
        - Name
        - DOB/YOB
        - Gender
        - Address components
        - Photo (base64)
        - Digital signature
        """
        try:
            # Use pyaadhaar if available
            if PYAADHAAR_AVAILABLE:
                try:
                    qr_obj = AadhaarSecureQr(int(numeric_string))
                    decoded = qr_obj.decodeddata()
                    
                    return DemographicData(
                        name=decoded.get('name', ''),
                        dob=decoded.get('dob', ''),
                        yob=decoded.get('yob', str(decoded.get('dob', ''))[:4] if decoded.get('dob') else None),
                        gender=self._normalize_gender(decoded.get('gender', '')),
                        masked_aadhaar=self._format_masked_aadhaar(decoded.get('uid', '')),
                        reference_id=decoded.get('referenceid', ''),
                        address=self._build_address(decoded),
                        photo_base64=decoded.get('image', ''),
                        email_hash=decoded.get('email_mobile_status', {}).get('email'),
                        mobile_hash=decoded.get('email_mobile_status', {}).get('mobile'),
                        qr_type=QRType.SECURE_QR_V2,
                        raw_data=numeric_string[:100] + "..."  # Truncate for storage
                    )
                except Exception as e:
                    logger.warning(f"pyaadhaar parsing failed: {e}")
            
            # Manual parsing fallback
            return self._manual_parse_secure_qr(numeric_string)
            
        except Exception as e:
            logger.error(f"Secure QR V2 parsing failed: {e}")
            return None
    
    def _manual_parse_secure_qr(self, numeric_string: str) -> Optional[DemographicData]:
        """
        Manual parsing of Secure QR when pyaadhaar is not available.
        The secure QR is a big integer that, when converted to bytes, contains:
        - Byte 0: Version/email-mobile flags
        - Bytes 1-256: Digital signature (256 bytes)
        - Remaining: zlib-compressed data
        """
        try:
            # Convert numeric string to bytes
            big_int = int(numeric_string)
            byte_length = (big_int.bit_length() + 7) // 8
            raw_bytes = big_int.to_bytes(byte_length, byteorder='big')
            
            if len(raw_bytes) < 260:
                logger.warning("QR data too short for Secure QR format")
                return None
            
            # Extract signature (bytes 1-256)
            signature = raw_bytes[1:257]
            
            # Extract and decompress data
            compressed_data = raw_bytes[257:]
            
            try:
                decompressed = zlib.decompress(compressed_data, wbits=-15)
                data_str = decompressed.decode('utf-8', errors='replace')
            except:
                # Try without negative wbits
                try:
                    decompressed = zlib.decompress(compressed_data)
                    data_str = decompressed.decode('utf-8', errors='replace')
                except Exception as e:
                    logger.error(f"Decompression failed: {e}")
                    return None
            
            # Parse the decompressed data (typically delimited)
            # Format: RefID, Name, DOB, Gender, Address parts...
            parts = data_str.split('\x00') if '\x00' in data_str else data_str.split('\xff')
            
            if len(parts) < 4:
                logger.warning(f"Insufficient data parts: {len(parts)}")
                return None
            
            return DemographicData(
                reference_id=parts[0] if len(parts) > 0 else None,
                name=parts[1] if len(parts) > 1 else None,
                dob=parts[2] if len(parts) > 2 else None,
                gender=self._normalize_gender(parts[3]) if len(parts) > 3 else None,
                qr_type=QRType.SECURE_QR_V2,
                raw_data=numeric_string[:100] + "..."
            )
            
        except Exception as e:
            logger.error(f"Manual Secure QR parsing failed: {e}")
            return None
    
    def _parse_secure_qr_bytes(self, raw_bytes: bytes) -> Optional[DemographicData]:
        """Parse Secure QR from raw bytes (already decoded from QR)."""
        try:
            if len(raw_bytes) < 260:
                return None
            
            # Similar structure to numeric string version
            compressed_data = raw_bytes[257:]
            
            try:
                decompressed = zlib.decompress(compressed_data, wbits=-15)
            except:
                try:
                    decompressed = zlib.decompress(compressed_data)
                except:
                    return None
            
            data_str = decompressed.decode('utf-8', errors='replace')
            parts = data_str.split('\x00')
            
            if len(parts) >= 4:
                return DemographicData(
                    reference_id=parts[0],
                    name=parts[1],
                    dob=parts[2],
                    gender=self._normalize_gender(parts[3]),
                    qr_type=QRType.SECURE_QR_V2
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Bytes parsing failed: {e}")
            return None
    
    def _parse_old_xml_qr(self, xml_string: str) -> Optional[DemographicData]:
        """Parse old XML-based QR code format."""
        try:
            # Use pyaadhaar if available
            if PYAADHAAR_AVAILABLE:
                try:
                    qr_obj = AadhaarOldQr(xml_string)
                    decoded = qr_obj.decodeddata()
                    
                    return DemographicData(
                        name=decoded.get('name', ''),
                        dob=decoded.get('dob', ''),
                        yob=decoded.get('yob', ''),
                        gender=self._normalize_gender(decoded.get('gender', '')),
                        masked_aadhaar=self._format_masked_aadhaar(decoded.get('uid', '')),
                        reference_id=decoded.get('referenceid', ''),
                        address=self._build_address(decoded),
                        qr_type=QRType.OLD_XML_QR,
                        raw_data=xml_string[:100] + "..."
                    )
                except Exception as e:
                    logger.warning(f"pyaadhaar XML parsing failed: {e}")
            
            # Manual XML parsing
            import re
            
            # Extract attributes using regex
            uid_match = re.search(r'uid="(\d+)"', xml_string)
            name_match = re.search(r'name="([^"]+)"', xml_string)
            dob_match = re.search(r'(?:dob|DOB)="([^"]+)"', xml_string)
            yob_match = re.search(r'(?:yob|YOB)="([^"]+)"', xml_string)
            gender_match = re.search(r'gender="([^"]+)"', xml_string)
            
            return DemographicData(
                name=name_match.group(1) if name_match else None,
                dob=dob_match.group(1) if dob_match else None,
                yob=yob_match.group(1) if yob_match else None,
                gender=self._normalize_gender(gender_match.group(1)) if gender_match else None,
                masked_aadhaar=self._format_masked_aadhaar(uid_match.group(1)) if uid_match else None,
                qr_type=QRType.OLD_XML_QR,
                raw_data=xml_string[:100] + "..."
            )
            
        except Exception as e:
            logger.error(f"XML QR parsing failed: {e}")
            return None
    
    def _normalize_gender(self, gender: str) -> str:
        """Normalize gender string to M/F/O."""
        if not gender:
            return "U"
        
        gender_lower = gender.lower().strip()
        
        if gender_lower in ['m', 'male', 'पुरुष']:
            return "M"
        elif gender_lower in ['f', 'female', 'महिला']:
            return "F"
        elif gender_lower in ['o', 'other', 'अन्य']:
            return "O"
        
        return gender.upper()[0] if gender else "U"
    
    def _format_masked_aadhaar(self, uid: str) -> Optional[str]:
        """Format UID as masked Aadhaar (XXXX-XXXX-1234)."""
        if not uid:
            return None
        
        # Remove any existing formatting
        uid_clean = ''.join(filter(str.isdigit, str(uid)))
        
        if len(uid_clean) == 12:
            return f"XXXX-XXXX-{uid_clean[-4:]}"
        elif len(uid_clean) == 4:
            return f"XXXX-XXXX-{uid_clean}"
        elif len(uid_clean) >= 4:
            return f"XXXX-XXXX-{uid_clean[-4:]}"
        
        return None
    
    def _build_address(self, decoded: dict) -> str:
        """Build address string from decoded parts."""
        address_parts = []
        
        for key in ['co', 'house', 'street', 'lm', 'loc', 'vtc', 'subdist', 'dist', 'state', 'pc']:
            if key in decoded and decoded[key]:
                address_parts.append(str(decoded[key]))
        
        return ', '.join(address_parts) if address_parts else ''
    
    def extract_signature_bytes(self, raw_data: Union[bytes, str]) -> Optional[bytes]:
        """Extract the digital signature bytes from QR data."""
        try:
            if isinstance(raw_data, str):
                if raw_data.isdigit():
                    big_int = int(raw_data)
                    byte_length = (big_int.bit_length() + 7) // 8
                    raw_bytes = big_int.to_bytes(byte_length, byteorder='big')
                else:
                    raw_bytes = raw_data.encode('utf-8')
            else:
                raw_bytes = raw_data
            
            if len(raw_bytes) >= 257:
                return raw_bytes[1:257]  # Signature is bytes 1-256
            
            return None
            
        except Exception as e:
            logger.error(f"Signature extraction failed: {e}")
            return None
    
    def extract_signed_data(self, raw_data: Union[bytes, str]) -> Optional[bytes]:
        """Extract the data that was signed (for verification)."""
        try:
            if isinstance(raw_data, str):
                if raw_data.isdigit():
                    big_int = int(raw_data)
                    byte_length = (big_int.bit_length() + 7) // 8
                    raw_bytes = big_int.to_bytes(byte_length, byteorder='big')
                else:
                    raw_bytes = raw_data.encode('utf-8')
            else:
                raw_bytes = raw_data
            
            if len(raw_bytes) > 257:
                return raw_bytes[257:]  # Data is after signature
            
            return None
            
        except Exception as e:
            logger.error(f"Signed data extraction failed: {e}")
            return None


# Singleton instance
qr_detector = QRDetector()
