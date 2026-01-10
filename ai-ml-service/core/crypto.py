import cv2
import numpy as np
from typing import Tuple
from core.plausibility_engine import VerhoeffValidator

class CryptographicValidator:
    

    @staticmethod
    def decode_qr(image_path: str) -> dict:
        
        original_image = cv2.imread(image_path)
        if original_image is None:
            # Try to handle RGBA images if imread fails (though typically it just returns None)
            from PIL import Image
            try:
                pil_img = Image.open(image_path).convert('RGB')
                original_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except:
                return {"success": False, "error": "Could not read image"}

        detector = cv2.QRCodeDetector()

        def attempt_detection(img):
            data, bbox, _ = detector.detectAndDecode(img)
            return data

        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        
        # Try different rotations
        for angle in [0, 90, 180, 270]:
            if angle == 0:
                rotated = gray
            else:
                # Rotate image
                (h, w) = gray.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(gray, M, (w, h))

            # 1. Standard Gray
            data = attempt_detection(rotated)
            if data: break

            # 2. High Contrast
            alpha = 1.5
            beta = 0
            contrast_img = cv2.convertScaleAbs(rotated, alpha=alpha, beta=beta)
            data = attempt_detection(contrast_img)
            if data: break

            # 3. Enhanced CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(rotated)
            data = attempt_detection(enhanced)
            if data: break

            # 4. Sharpening
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            data = attempt_detection(sharpened)
            if data: break

            # 5. Tiling/Magnifying (Right side)
            rh, rw = rotated.shape
            right_half = rotated[0:rh, rw//2:rw]
            zoomed = cv2.resize(right_half, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            data = attempt_detection(zoomed)
            if data: break

            # 6. Adaptive Threshold (Binarize)
            binary = cv2.adaptiveThreshold(rotated, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            data = attempt_detection(binary)
            if data: break

            # 7. Downsampling (helps with very large images)
            h, w = rotated.shape
            if w > 1000:
                small = cv2.resize(rotated, (w // 2, h // 2))
                data = attempt_detection(small)
                if data: break

            # 8. Bottom-Right Focus (Horizontal cards)
            if angle == 0 or angle == 180:
                br_corner = rotated[2*h//3:, 2*w//3:]
                br_zoomed = cv2.resize(br_corner, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                data = attempt_detection(br_zoomed)
                if data: break

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
        
        return {"success": False, "error": f"No QR found (tried {angle+90 if data else 360}deg coverage & enhancements)"}

    @staticmethod
    def verify_qr_consistency(qr_data: dict, ocr_data: dict) -> Tuple[int, str]:
        
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
