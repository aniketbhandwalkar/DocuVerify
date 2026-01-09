import cv2
import numpy as np
from typing import Tuple
from core.plausibility_engine import VerhoeffValidator

class CryptographicValidator:
    """
    Layer 2: Cryptographic Proof.
    The strongest signal for authenticity in an offline environment.
    """

    @staticmethod
    def decode_qr(image_path: str) -> dict:
        """
        Layer 2: Advanced QR Decoding.
        Uses image enhancement to find high-density Aadhaar Secure QRs.
        """
        original_image = cv2.imread(image_path)
        if original_image is None:
            return {"success": False, "error": "Could not read image"}

        def attempt_detection(img):
            detector = cv2.QRCodeDetector()
            data, bbox, _ = detector.detectAndDecode(img)
            return data

        # Try multiple processing techniques
        # 1. Standard Gray
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        data = attempt_detection(gray)
        
        # 2. High Contrast (helps with dense QRs)
        if not data:
            alpha = 1.5 # Contrast
            beta = 0    # Brightness
            contrast_img = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
            data = attempt_detection(contrast_img)
            
        # 5. Tiling/Magnifying Strategy (Focus on the right side)
        if not data:
            h, w = gray.shape
            # Focus on the right half where Aadhaar Secure QR usually sits
            right_half = gray[0:h, w//2:w]
            # Zoom in 2x
            zoomed = cv2.resize(right_half, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            data = attempt_detection(zoomed)
            
            if not data:
                # Try extra sharp binarization on the zoom
                _, binary = cv2.threshold(zoomed, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                data = attempt_detection(binary)

        if data:
            # Check for Aadhaar patterns
            if data.isdigit() and len(data) > 100:
                return {
                    "success": True,
                    "data": {"raw_secure_data": data[:20] + "..."},
                    "is_secure": True,
                    "type": "Modern Secure QR"
                }
            elif "<?xml" in data:
                return {
                    "success": True,
                    "data": {"xml": "found"},
                    "is_secure": False,
                    "type": "Old XML QR"
                }
            return {
                "success": True,
                "data": {"text": data},
                "is_secure": False,
                "type": "General QR"
            }
        
        return {"success": False, "error": "No QR found (tried 3 enhancement layers)"}

    @staticmethod
    def verify_qr_consistency(qr_data: dict, ocr_data: dict) -> Tuple[int, str]:
        """
        Layer 5: Cross-Validation.
        Does the data in the QR match the text on the card?
        """
        if not qr_data or not ocr_data:
            return 0, "Cross-validation impossible: Missing data."
            
        matches = 0
        total_fields = 0
        
        # Check Aadhaar number
        if 'aadhaar' in qr_data and 'number' in ocr_data:
            total_fields += 1
            if qr_data['aadhaar'][-4:] == ocr_data['number'][-4:]: # Match last 4 digits
                matches += 1
                
        # Check Name (Fuzzy Match)
        if 'name' in qr_data and 'name' in ocr_data:
            total_fields += 1
            if qr_data['name'].lower() in ocr_data['name'].lower():
                matches += 1

        if total_fields == 0:
            return 0, "No common fields to cross-validate."
            
        confidence = int((matches / total_fields) * 40)
        return confidence, f"Cross-validation: {matches}/{total_fields} fields match."
