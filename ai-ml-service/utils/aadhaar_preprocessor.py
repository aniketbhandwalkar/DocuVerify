import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)


class AadhaarPreprocessor:
    
    def __init__(self):
        # CLAHE settings for contrast enhancement
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        
        # Kernel sizes for morphological operations
        self.denoise_kernel = np.ones((2, 2), np.uint8)
        self.closing_kernel = np.ones((3, 3), np.uint8)
        
        logger.info("AadhaarPreprocessor initialized")
    
    def preprocess_for_aadhaar_ocr(
        self, 
        image: np.ndarray,
        enhance_numbers: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Main preprocessing pipeline for Aadhaar card OCR
        
        Args:
            image: Input BGR image (from cv2.imread)
            enhance_numbers: If True, apply extra enhancement for digit regions
            
        Returns:
            Tuple of (preprocessed_image, preprocessing_info)
        """
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
            
            # Step 2: Deskew image (correct rotation)
            gray, skew_angle = self.detect_and_correct_skew(gray)
            preprocessing_info['skew_angle'] = skew_angle
            preprocessing_info['steps_applied'].append('deskewing')
            
            # Step 3: CLAHE contrast enhancement
            enhanced = self.clahe.apply(gray)
            preprocessing_info['steps_applied'].append('clahe_enhancement')
            
            # Step 4: Bilateral filtering (denoise while preserving edges)
            denoised = cv2.bilateralFilter(enhanced, d=5, sigmaColor=50, sigmaSpace=50)
            preprocessing_info['steps_applied'].append('bilateral_denoising')
            
            # Step 5: Adaptive thresholding
            # Gaussian adaptive threshold works better for Aadhaar cards
            binary = cv2.adaptiveThreshold(
                denoised,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=11,
                C=2
            )
            preprocessing_info['steps_applied'].append('adaptive_thresholding')
            
            # Step 6: Morphological operations to clean up
            # Remove small noise
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.denoise_kernel)
            # Connect broken characters
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self.closing_kernel)
            preprocessing_info['steps_applied'].append('morphological_cleanup')
            
            # Step 7: Optional - Extra enhancement for numeric regions
            if enhance_numbers:
                # Apply additional sharpening for better digit recognition
                binary = self.enhance_digits(binary)
                preprocessing_info['steps_applied'].append('digit_enhancement')
            
            preprocessing_info['final_size'] = binary.shape[:2]
            preprocessing_info['success'] = True
            
            logger.debug(f"Preprocessing completed: {len(preprocessing_info['steps_applied'])} steps")
            
            return binary, preprocessing_info
            
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            preprocessing_info['success'] = False
            preprocessing_info['error'] = str(e)
            # Return original grayscale as fallback
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            return gray, preprocessing_info
    
    def detect_and_correct_skew(
        self, 
        image: np.ndarray, 
        max_angle: float = 10.0
    ) -> Tuple[np.ndarray, float]:
        """
        Detect and correct image skew/rotation using Hough line detection
        
        Args:
            image: Grayscale image
            max_angle: Maximum rotation angle to correct (degrees)
            
        Returns:
            Tuple of (deskewed_image, detected_angle)
        """
        try:
            # Edge detection for line detection
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            
            # Hough line detection
            lines = cv2.HoughLinesP(
                edges, 
                rho=1, 
                theta=np.pi/180, 
                threshold=100,
                minLineLength=100,
                maxLineGap=10
            )
            
            if lines is None or len(lines) == 0:
                return image, 0.0
            
            # Calculate angles of detected lines
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # Normalize to -90 to 90
                if angle < -45:
                    angle += 90
                elif angle > 45:
                    angle -= 90
                angles.append(angle)
            
            # Use median angle as skew angle
            median_angle = np.median(angles)
            
            # Only correct if angle is within reasonable range
            if abs(median_angle) > max_angle:
                logger.warning(f"Skew angle {median_angle:.2f}° exceeds max {max_angle}°, skipping")
                return image, 0.0
            
            # Rotate image to correct skew
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(
                image, 
                M, 
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            
            logger.debug(f"Corrected skew: {median_angle:.2f}°")
            return rotated, median_angle
            
        except Exception as e:
            logger.warning(f"Skew detection failed: {e}")
            return image, 0.0
    
    def enhance_digits(self, binary_image: np.ndarray) -> np.ndarray:
        """
        Apply additional enhancement specifically for digit regions
        
        Args:
            binary_image: Binary image after thresholding
            
        Returns:
            Enhanced binary image
        """
        try:
            # Apply sharpening kernel to make digits more distinct
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])
            sharpened = cv2.filter2D(binary_image, -1, kernel)
            
            # Ensure binary output
            _, sharpened = cv2.threshold(sharpened, 127, 255, cv2.THRESH_BINARY)
            
            return sharpened
            
        except Exception as e:
            logger.warning(f"Digit enhancement failed: {e}")
            return binary_image
    
    def preprocess_number_region(
        self, 
        image: np.ndarray, 
        bbox: Optional[Tuple[int, int, int, int]] = None
    ) -> np.ndarray:
        """
        Specialized preprocessing for Aadhaar number region
        Applies more aggressive enhancement for better digit recognition
        
        Args:
            image: Full image (BGR or grayscale)
            bbox: Bounding box (x, y, w, h) of number region, if known
            
        Returns:
            Preprocessed number region image
        """
        try:
            # Extract region if bbox provided
            if bbox is not None:
                x, y, w, h = bbox
                region = image[y:y+h, x:x+w]
            else:
                region = image
            
            # Convert to grayscale if needed
            if len(region.shape) == 3:
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            else:
                gray = region
            
            # Aggressive CLAHE for number region
            clahe_aggressive = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
            enhanced = clahe_aggressive.apply(gray)
            
            # Strong denoising
            denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
            
            # Otsu's thresholding for optimal binary conversion
            _, binary = cv2.threshold(
                denoised, 
                0, 
                255, 
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            
            # Remove very small noise
            kernel = np.ones((2, 2), np.uint8)
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # Dilate slightly to connect broken digits
            dilated = cv2.dilate(cleaned, kernel, iterations=1)
            
            logger.debug("Number region preprocessing completed")
            return dilated
            
        except Exception as e:
            logger.error(f"Number region preprocessing error: {e}")
            # Fallback to basic grayscale
            if len(image.shape) == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return image
    
    def remove_security_patterns(self, image: np.ndarray) -> np.ndarray:
        """
        Remove security watermarks/patterns using frequency domain filtering
        This helps OCR focus on text rather than background patterns
        
        Args:
            image: Grayscale image
            
        Returns:
            Image with reduced security patterns
        """
        try:
            # Apply FFT to frequency domain
            f_transform = np.fft.fft2(image)
            f_shift = np.fft.fftshift(f_transform)
            
            # Create a mask to filter high-frequency noise (security patterns)
            rows, cols = image.shape
            crow, ccol = rows // 2, cols // 2
            
            # High-pass filter mask (keeps text, removes fine patterns)
            mask = np.ones((rows, cols), np.uint8)
            r = 30  # Radius for filtering
            center = [crow, ccol]
            x, y = np.ogrid[:rows, :cols]
            mask_area = (x - center[0])**2 + (y - center[1])**2 <= r*r
            mask[mask_area] = 0
            
            # Apply mask and inverse FFT
            f_shift_filtered = f_shift * mask
            f_ishift = np.fft.ifftshift(f_shift_filtered)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            
            # Normalize to 0-255
            img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
            
            return img_back.astype(np.uint8)
            
        except Exception as e:
            logger.warning(f"Security pattern removal failed: {e}")
            return image
    
    def get_optimal_resize(
        self, 
        image: np.ndarray, 
        target_dpi: int = 300
    ) -> np.ndarray:
        """
        Resize image to optimal DPI for Tesseract OCR
        Tesseract works best at 300 DPI
        
        Args:
            image: Input image
            target_dpi: Target DPI (default 300)
            
        Returns:
            Resized image
        """
        try:
            h, w = image.shape[:2]
            
            # Estimate current DPI (assuming Aadhaar card is ~8.5cm x 5.4cm)
            # Standard Aadhaar dimensions
            aadhaar_width_cm = 8.56
            current_dpi = w / (aadhaar_width_cm / 2.54)  # Convert cm to inches
            
            # Calculate scaling factor
            scale = target_dpi / current_dpi
            
            # Only resize if significant difference
            if 0.8 < scale < 1.2:
                logger.debug(f"Image DPI ~{current_dpi:.0f}, no resize needed")
                return image
            
            # Resize
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            resized = cv2.resize(
                image, 
                (new_w, new_h), 
                interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
            )
            
            logger.debug(f"Resized from {w}x{h} to {new_w}x{new_h} (DPI: {current_dpi:.0f} → {target_dpi})")
            return resized
            
        except Exception as e:
            logger.warning(f"Resize failed: {e}")
            return image
