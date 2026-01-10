
import cv2
import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

class AadhaarPreprocessor:
    
    def __init__(self):
        # CLAHE settings for contrast enhancement
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        
        # Kernel sizes for morphological operations
        self.denoise_kernel = np.ones((2, 2), np.uint8)
        self.closing_kernel = np.ones((3, 3), np.uint8)
        
        logger.info("AadhaarPreprocessor initialized")
    
    def remove_moire_patterns(self, image: np.ndarray) -> np.ndarray:
        
        try:
            # Median blur is very effective against Moire dots
            filtered = cv2.medianBlur(image, 3)
            # Bilateral filter to preserve edges while smoothing
            filtered = cv2.bilateralFilter(filtered, 9, 75, 75)
            return filtered
        except Exception:
            return image

    def preprocess_for_aadhaar_ocr(
        self, 
        image: np.ndarray,
        enhance_numbers: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        
        try:
            preprocessing_info = {
                'original_size': image.shape[:2],
                'steps_applied': []
            }
            
            # Step 1: Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                preprocessing_info['steps_applied'].append('grayscale_conversion')
            else:
                gray = image.copy()
            
            # Step 2: Deskew image
            gray, skew_angle = self.detect_and_correct_skew(gray)
            preprocessing_info['skew_angle'] = skew_angle
            preprocessing_info['steps_applied'].append('deskewing')
            
            # Step 3: CLAHE enhancement
            enhanced = self.clahe.apply(gray)
            preprocessing_info['steps_applied'].append('clahe_enhancement')
            
            # Step 4: Bilateral Filter
            denoised = cv2.bilateralFilter(enhanced, d=5, sigmaColor=50, sigmaSpace=50)
            preprocessing_info['steps_applied'].append('bilateral_denoising')
            
            # Step 5: Moire removal
            denoised = self.remove_moire_patterns(denoised)
            preprocessing_info['steps_applied'].append('moire_removal')
            
            # Step 6: Adaptive Thresholding
            binary = cv2.adaptiveThreshold(
                denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            preprocessing_info['steps_applied'].append('adaptive_thresholding')
            
            # Step 7: Morphological cleanup
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.denoise_kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.closing_kernel)
            preprocessing_info['steps_applied'].append('morphological_cleanup')
            
            if enhance_numbers:
                binary = self.enhance_digits(binary)
                preprocessing_info['steps_applied'].append('digit_enhancement')
            
            return binary, preprocessing_info
            
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            return image, {'success': False, 'error': str(e)}

    def detect_and_correct_skew(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        
        try:
            # Simple deskewing using minAreaRect on non-zero pixels
            coords = np.column_stack(np.where(image > 0))
            angle = cv2.minAreaRect(coords)[-1]
            
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
                
            if abs(angle) < 0.5:
                return image, 0.0
                
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated, angle
        except Exception:
            return image, 0.0

    def enhance_digits(self, image: np.ndarray) -> np.ndarray:
        
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
            dilated = cv2.dilate(image, kernel, iterations=1)
            eroded = cv2.erode(dilated, kernel, iterations=1)
            return eroded
        except Exception:
            return image

    def resize_to_dpi(self, image: np.ndarray, target_dpi: int = 300) -> np.ndarray:
        
        try:
            h, w = image.shape[:2]
            # Assume 72 DPI if unknown
            current_dpi = 72
            scale = target_dpi / current_dpi
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        except Exception:
            return image
