import cv2
import numpy as np
from PIL import Image, ImageEnhance
import logging

logger = logging.getLogger(__name__)

def preprocess_image(image, enhancement_type="default"):
    
    try:
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if enhancement_type == "ocr":
            return preprocess_for_ocr(image)
        elif enhancement_type == "signature":
            return preprocess_for_signature(image)
        elif enhancement_type == "quality":
            return enhance_image_quality(image)
        else:
            return default_preprocessing(image)
            
    except Exception as e:
        logger.error(f"Image preprocessing error: {str(e)}")
        return image

def preprocess_for_ocr(image):
    
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Noise removal
        denoised = cv2.medianBlur(gray, 3)
        
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Thresholding
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
        
    except Exception as e:
        logger.error(f"OCR preprocessing error: {str(e)}")
        return image

def preprocess_for_signature(image):
    
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Adaptive thresholding for varying lighting conditions
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morphological closing to connect nearby components
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return closed
        
    except Exception as e:
        logger.error(f"Signature preprocessing error: {str(e)}")
        return image

def enhance_image_quality(image):
    
    try:
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_image)
        enhanced = enhancer.enhance(1.2)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(enhanced)
        enhanced = enhancer.enhance(1.1)
        
        # Convert back to numpy array
        return np.array(enhanced)
        
    except Exception as e:
        logger.error(f"Quality enhancement error: {str(e)}")
        return image

def default_preprocessing(image):
    
    try:
        # Basic noise removal
        if len(image.shape) == 3:
            # Convert to LAB color space for better processing
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # Split channels
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            # Merge channels
            enhanced = cv2.merge([l, a, b])
            
            # Convert back to RGB
            result = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            result = clahe.apply(image)
        
        return result
        
    except Exception as e:
        logger.error(f"Default preprocessing error: {str(e)}")
        return image

def resize_image(image, max_width=1200, max_height=1600):
    
    try:
        if isinstance(image, np.ndarray):
            height, width = image.shape[:2]
        else:
            width, height = image.size
        
        # Calculate scaling factor
        scale_w = max_width / width
        scale_h = max_height / height
        scale = min(scale_w, scale_h, 1.0)  # Don't upscale
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            if isinstance(image, np.ndarray):
                resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            else:
                resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            return resized
        
        return image
        
    except Exception as e:
        logger.error(f"Image resize error: {str(e)}")
        return image

def normalize_image(image):
    
    try:
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                # Normalize to 0-255 range
                normalized = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                return normalized
        
        return image
        
    except Exception as e:
        logger.error(f"Image normalization error: {str(e)}")
        return image

def detect_image_orientation(image):
    
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Simple orientation detection based on text line detection
        # This is a simplified implementation
        
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        # Count horizontal vs vertical lines
        h_count = cv2.countNonZero(horizontal_lines)
        v_count = cv2.countNonZero(vertical_lines)
        
        # Determine if rotation is needed
        if v_count > h_count * 1.5:  # More vertical than horizontal lines
            return "rotate_90"
        else:
            return "no_rotation"
            
    except Exception as e:
        logger.error(f"Orientation detection error: {str(e)}")
        return "no_rotation"

def correct_image_orientation(image, orientation):
    
    try:
        if orientation == "rotate_90":
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif orientation == "rotate_180":
            return cv2.rotate(image, cv2.ROTATE_180)
        elif orientation == "rotate_270":
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            return image
            
    except Exception as e:
        logger.error(f"Orientation correction error: {str(e)}")
        return image

def extract_text_regions(image):
    
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Create kernels for text detection
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        
        # Detect horizontal lines
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        # Detect vertical lines
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        # Combine lines to find text regions
        combined = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
        
        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small noise
                x, y, w, h = cv2.boundingRect(contour)
                text_regions.append({
                    'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                    'area': area
                })
        
        return text_regions
        
    except Exception as e:
        logger.error(f"Text region extraction error: {str(e)}")
        return []

def extract_photo_regions(image):
    
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        photo_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 5000 < area < 100000:  # Photo-like area range
                # Check if it's roughly rectangular
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                if len(approx) >= 4:  # Rectangular-ish
                    x, y, w, h = cv2.boundingRect(contour)
                    photo_regions.append({
                        'bbox': {'x': x, 'y': y, 'width': w, 'height': h},
                        'area': area,
                        'vertices': len(approx)
                    })
        
        return photo_regions
        
    except Exception as e:
        logger.error(f"Photo region extraction error: {str(e)}")
        return []
