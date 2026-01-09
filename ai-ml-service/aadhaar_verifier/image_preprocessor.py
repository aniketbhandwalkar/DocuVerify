"""
Image Preprocessor for Aadhaar Verification

Handles image enhancement, rotation correction, contrast adjustment,
and region detection for optimal QR and OCR processing.
"""

import cv2
import numpy as np
from PIL import Image
import logging
from typing import Optional, List, Tuple, Union
import io

from .models import PreprocessedImage

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """
    Preprocesses Aadhaar card images for optimal QR detection and OCR extraction.
    """
    
    def __init__(self):
        self.target_width = 1200
        self.target_height = 800
    
    def preprocess(self, image_input: Union[str, bytes, np.ndarray]) -> PreprocessedImage:
        """
        Main preprocessing pipeline for Aadhaar card images.
        
        Args:
            image_input: File path, bytes, or numpy array
            
        Returns:
            PreprocessedImage with processed image and metadata
        """
        notes = []
        original_path = ""
        
        # Load image based on input type
        if isinstance(image_input, str):
            original_path = image_input
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Could not load image from path: {image_input}")
            notes.append("Loaded from file path")
        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Could not decode image bytes")
            notes.append("Loaded from bytes")
        elif isinstance(image_input, np.ndarray):
            image = image_input.copy()
            notes.append("Loaded from numpy array")
        else:
            raise ValueError(f"Unsupported input type: {type(image_input)}")
        
        # Store original dimensions
        original_height, original_width = image.shape[:2]
        notes.append(f"Original size: {original_width}x{original_height}")
        
        # Step 1: Resize if too large
        image, resized = self._resize_if_needed(image)
        if resized:
            notes.append(f"Resized to: {image.shape[1]}x{image.shape[0]}")
        
        # Step 2: Correct rotation
        image, rotation = self._correct_rotation(image)
        if rotation != 0:
            notes.append(f"Rotation corrected: {rotation}°")
        
        # Step 3: Enhance contrast
        image, contrast_enhanced = self._enhance_contrast(image)
        if contrast_enhanced:
            notes.append("Contrast enhanced (CLAHE)")
        
        # Step 4: Reduce noise/blur
        image, blur_corrected = self._reduce_blur(image)
        if blur_corrected:
            notes.append("Blur reduction applied")
        
        # Step 5: Detect QR region
        qr_region = self._detect_qr_region(image)
        if qr_region:
            notes.append(f"QR region detected at: {qr_region}")
        
        # Step 6: Detect text regions
        text_regions = self._detect_text_regions(image)
        if text_regions:
            notes.append(f"Detected {len(text_regions)} text regions")
        
        return PreprocessedImage(
            original_path=original_path,
            processed_image=image,
            rotation_applied=rotation,
            contrast_enhanced=contrast_enhanced,
            blur_corrected=blur_corrected,
            qr_region=qr_region,
            text_regions=text_regions,
            preprocessing_notes=notes
        )
    
    def _resize_if_needed(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Resize image if it's too large while maintaining aspect ratio."""
        height, width = image.shape[:2]
        
        if width <= self.target_width and height <= self.target_height:
            return image, False
        
        # Calculate scaling factor
        scale_w = self.target_width / width
        scale_h = self.target_height / height
        scale = min(scale_w, scale_h)
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized, True
    
    def _correct_rotation(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Detect and correct image rotation using edge detection."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                                     minLineLength=100, maxLineGap=10)
            
            if lines is None or len(lines) < 5:
                return image, 0.0
            
            # Calculate dominant angle
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # Normalize to -45 to 45 range
                while angle > 45:
                    angle -= 90
                while angle < -45:
                    angle += 90
                angles.append(angle)
            
            median_angle = np.median(angles)
            
            # Only correct if rotation is significant
            if abs(median_angle) < 1.0:
                return image, 0.0
            
            # Rotate image
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(image, rotation_matrix, (width, height),
                                      flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_REPLICATE)
            
            return rotated, round(median_angle, 2)
            
        except Exception as e:
            logger.warning(f"Rotation correction failed: {e}")
            return image, 0.0
    
    def _enhance_contrast(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Enhance image contrast using CLAHE."""
        try:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            
            # Merge back
            enhanced_lab = cv2.merge([l_enhanced, a, b])
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
            
            return enhanced, True
            
        except Exception as e:
            logger.warning(f"Contrast enhancement failed: {e}")
            return image, False
    
    def _reduce_blur(self, image: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Apply sharpening to reduce blur."""
        try:
            # Calculate blur metric using Laplacian variance
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur_metric = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Only sharpen if image appears blurry
            if blur_metric > 100:  # Already sharp enough
                return image, False
            
            # Apply unsharp masking
            gaussian = cv2.GaussianBlur(image, (0, 0), 3)
            sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
            
            return sharpened, True
            
        except Exception as e:
            logger.warning(f"Blur reduction failed: {e}")
            return image, False
    
    def _detect_qr_region(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect the QR code region in the image."""
        try:
            # Try OpenCV QR detector
            detector = cv2.QRCodeDetector()
            data, points, _ = detector.detectAndDecode(image)
            
            if points is not None and len(points) > 0:
                points = points[0].astype(int)
                x_min = points[:, 0].min()
                y_min = points[:, 1].min()
                x_max = points[:, 0].max()
                y_max = points[:, 1].max()
                return (x_min, y_min, x_max - x_min, y_max - y_min)
            
            # Fallback: look for square patterns
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = w / h if h > 0 else 0
                    area = w * h
                    image_area = image.shape[0] * image.shape[1]
                    
                    # QR codes are roughly square and take up reasonable portion
                    if 0.8 <= aspect_ratio <= 1.2 and 0.01 <= area / image_area <= 0.3:
                        return (x, y, w, h)
            
            return None
            
        except Exception as e:
            logger.warning(f"QR region detection failed: {e}")
            return None
    
    def _detect_text_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect text regions in the image using morphological operations."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply morphological gradient
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
            
            # Binarize
            _, bw = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Connect text regions
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
            connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                area = w * h
                
                # Filter for text-like regions
                if aspect_ratio > 1.5 and area > 500 and w > 50:
                    regions.append((x, y, w, h))
            
            # Sort by y-coordinate (top to bottom)
            regions.sort(key=lambda r: r[1])
            
            return regions[:20]  # Limit to top 20 regions
            
        except Exception as e:
            logger.warning(f"Text region detection failed: {e}")
            return []
    
    def get_qr_cropped(self, image: np.ndarray, region: Tuple[int, int, int, int], 
                        padding: int = 10) -> np.ndarray:
        """Extract QR region from image with padding."""
        x, y, w, h = region
        height, width = image.shape[:2]
        
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(width, x + w + padding)
        y2 = min(height, y + h + padding)
        
        return image[y1:y2, x1:x2]
    
    def prepare_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Prepare image specifically for OCR processing."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
        
        return denoised


# Singleton instance
image_preprocessor = ImagePreprocessor()
