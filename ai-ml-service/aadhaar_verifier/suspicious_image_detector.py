"""
Suspicious Image Detector for Aadhaar Verification

Detects images that are likely fake based on:
- Deity images (blue skin, multiple elements, gold ornaments)
- Non-photorealistic textures (drawings, paintings, cartoons)
- Unusual skin colors
- Stock photo characteristics
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


class SuspiciousImageDetector:
    """
    Detects suspicious characteristics in Aadhaar card photos
    that indicate fake documents.
    """
    
    # Color ranges for deity detection (HSV format)
    # Blue skin (Krishna, Vishnu, etc.)
    BLUE_SKIN_LOWER = np.array([100, 50, 50])
    BLUE_SKIN_UPPER = np.array([130, 255, 255])
    
    # Gold color (common in deity images - ornaments, crowns)
    GOLD_LOWER = np.array([15, 100, 100])
    GOLD_UPPER = np.array([35, 255, 255])
    
    # Normal human skin tones (for comparison)
    SKIN_LOWER = np.array([0, 20, 70])
    SKIN_UPPER = np.array([20, 150, 255])
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def analyze_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Analyze image for suspicious characteristics.
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            Dictionary with analysis results
        """
        result = {
            'is_suspicious': False,
            'suspicion_score': 0,
            'reasons': [],
            'checks': {}
        }
        
        try:
            # Convert to HSV for color analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 1. Check for blue skin (deity indicator)
            blue_check = self._check_blue_skin(hsv, image)
            result['checks']['blue_skin'] = blue_check
            if blue_check['detected']:
                result['suspicion_score'] += 40
                result['reasons'].append("Detected unusual blue skin color (possible deity image)")
            
            # 2. Check for excessive gold colors (deity ornaments)
            gold_check = self._check_gold_ornaments(hsv)
            result['checks']['gold_ornaments'] = gold_check
            if gold_check['detected']:
                result['suspicion_score'] += 20
                result['reasons'].append("Detected excessive gold/metallic colors (possible deity ornaments)")
            
            # 3. Check for non-photorealistic texture
            texture_check = self._check_texture_reality(image)
            result['checks']['texture_reality'] = texture_check
            if texture_check['is_painting']:
                result['suspicion_score'] += 30
                result['reasons'].append("Image appears to be a painting or illustration, not a photo")
            
            # 4. Check for multiple faces (deities often shown with multiple faces/heads)
            face_check = self._check_multiple_faces(image)
            result['checks']['face_count'] = face_check
            if face_check['count'] > 1:
                result['suspicion_score'] += 25
                result['reasons'].append(f"Detected {face_check['count']} faces (expected 1 for Aadhaar)")
            elif face_check['count'] == 0:
                result['suspicion_score'] += 15
                result['reasons'].append("No clear face detected in photo area")
            
            # 5. Check for unnatural saturation (common in deity images)
            saturation_check = self._check_saturation(hsv)
            result['checks']['saturation'] = saturation_check
            if saturation_check['is_oversaturated']:
                result['suspicion_score'] += 15
                result['reasons'].append("Image has unnatural color saturation")
            
            # 6. Check for cartoon/drawn characteristics
            cartoon_check = self._check_cartoon_style(image)
            result['checks']['cartoon_style'] = cartoon_check
            if cartoon_check['is_cartoon']:
                result['suspicion_score'] += 35
                result['reasons'].append("Image appears to be a cartoon or digital illustration")
            
            # Determine if image is suspicious
            result['is_suspicious'] = result['suspicion_score'] >= 30
            
            logger.info(f"Suspicious image analysis: score={result['suspicion_score']}, "
                       f"suspicious={result['is_suspicious']}, reasons={len(result['reasons'])}")
            
        except Exception as e:
            logger.error(f"Error in suspicious image detection: {e}")
            result['error'] = str(e)
        
        return result
    
    def _check_blue_skin(self, hsv: np.ndarray, bgr: np.ndarray) -> Dict[str, Any]:
        """Check for blue skin tones (common in deity images)."""
        try:
            # Create mask for blue color in skin region
            blue_mask = cv2.inRange(hsv, self.BLUE_SKIN_LOWER, self.BLUE_SKIN_UPPER)
            
            # Calculate percentage of blue pixels
            total_pixels = hsv.shape[0] * hsv.shape[1]
            blue_pixels = cv2.countNonZero(blue_mask)
            blue_percentage = (blue_pixels / total_pixels) * 100
            
            # Significant blue presence (>3% of image)
            detected = blue_percentage > 3.0
            
            return {
                'detected': detected,
                'percentage': round(blue_percentage, 2),
                'threshold': 3.0
            }
        except Exception as e:
            return {'detected': False, 'error': str(e)}
    
    def _check_gold_ornaments(self, hsv: np.ndarray) -> Dict[str, Any]:
        """Check for excessive gold colors (deity ornaments)."""
        try:
            gold_mask = cv2.inRange(hsv, self.GOLD_LOWER, self.GOLD_UPPER)
            
            total_pixels = hsv.shape[0] * hsv.shape[1]
            gold_pixels = cv2.countNonZero(gold_mask)
            gold_percentage = (gold_pixels / total_pixels) * 100
            
            # More than 8% gold is suspicious
            detected = gold_percentage > 8.0
            
            return {
                'detected': detected,
                'percentage': round(gold_percentage, 2),
                'threshold': 8.0
            }
        except Exception as e:
            return {'detected': False, 'error': str(e)}
    
    def _check_texture_reality(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Check if image has photorealistic texture or appears painted/illustrated.
        Uses edge density and gradient analysis.
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate Laplacian variance (measure of image texture/detail)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()
            
            # Paintings/illustrations often have smoother gradients and less noise
            # Real photos have more texture variance (typically > 100)
            
            # Calculate gradient magnitude
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            gradient_std = np.std(gradient_magnitude)
            
            # Paintings typically have lower gradient variation
            is_painting = variance < 50 or gradient_std < 30
            
            return {
                'is_painting': is_painting,
                'texture_variance': round(variance, 2),
                'gradient_std': round(gradient_std, 2),
                'threshold_variance': 50,
                'threshold_gradient': 30
            }
        except Exception as e:
            return {'is_painting': False, 'error': str(e)}
    
    def _check_multiple_faces(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect number of faces in the image."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            return {
                'count': len(faces),
                'faces': [{'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)} 
                         for (x, y, w, h) in faces]
            }
        except Exception as e:
            return {'count': 0, 'error': str(e)}
    
    def _check_saturation(self, hsv: np.ndarray) -> Dict[str, Any]:
        """Check for unnatural color saturation."""
        try:
            # Extract saturation channel
            saturation = hsv[:, :, 1]
            
            mean_saturation = np.mean(saturation)
            std_saturation = np.std(saturation)
            
            # Very high saturation is common in deity images and illustrations
            is_oversaturated = mean_saturation > 150 or (mean_saturation > 100 and std_saturation < 30)
            
            return {
                'is_oversaturated': is_oversaturated,
                'mean_saturation': round(float(mean_saturation), 2),
                'std_saturation': round(float(std_saturation), 2)
            }
        except Exception as e:
            return {'is_oversaturated': False, 'error': str(e)}
    
    def _check_cartoon_style(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect cartoon/animated style images using edge and color analysis.
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Cartoons typically have:
            # 1. Strong, uniform edges (black outlines)
            # 2. Flat color regions
            # 3. Low color variance within regions
            
            # Check for strong uniform edges (cartoon outlines)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = cv2.countNonZero(edges) / (edges.shape[0] * edges.shape[1])
            
            # Check color uniformity (cartoons have flat colors)
            # Quantize colors and check distribution
            quantized = (image // 64) * 64  # Reduce to 4 levels per channel
            unique_colors = len(np.unique(quantized.reshape(-1, 3), axis=0))
            
            # Normalize by image size
            pixels = image.shape[0] * image.shape[1]
            color_diversity = unique_colors / (pixels / 1000)  # Colors per 1000 pixels
            
            # Low color diversity + moderate edges = likely cartoon
            is_cartoon = color_diversity < 0.5 and edge_density > 0.05
            
            return {
                'is_cartoon': is_cartoon,
                'edge_density': round(edge_density, 4),
                'color_diversity': round(color_diversity, 4),
                'unique_colors': unique_colors
            }
        except Exception as e:
            return {'is_cartoon': False, 'error': str(e)}
    
    def extract_photo_region(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract the photo region from an Aadhaar card image.
        The photo is typically on the left side of the card.
        """
        try:
            h, w = image.shape[:2]
            
            # Photo is typically in the left 30-40% of the card
            # and in the middle-bottom region vertically
            photo_x = 0
            photo_y = int(h * 0.2)
            photo_w = int(w * 0.35)
            photo_h = int(h * 0.6)
            
            photo_region = image[photo_y:photo_y+photo_h, photo_x:photo_x+photo_w]
            
            return photo_region if photo_region.size > 0 else None
            
        except Exception as e:
            logger.error(f"Error extracting photo region: {e}")
            return None


# Singleton instance
suspicious_detector = SuspiciousImageDetector()
