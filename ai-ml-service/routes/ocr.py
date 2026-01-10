from fastapi import APIRouter, File, UploadFile, HTTPException
import cv2
import numpy as np
from PIL import Image
import io
import easyocr
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize EasyOCR reader (lazy initialization)
_ocr_reader = None

def get_ocr_reader():
    
    global _ocr_reader
    if _ocr_reader is None:
        try:
            _ocr_reader = easyocr.Reader(['en', 'hi'])
            logger.info("EasyOCR reader initialized")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            raise
    return _ocr_reader

@router.post("/ocr")
async def perform_ocr_analysis(file: UploadFile = File(...)):
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Only image files are supported")
        
        # Read file content
        content = await file.read()
        
        # Load image
        image = Image.open(io.BytesIO(content))
        
        # Perform OCR
        ocr_result = extract_text_with_confidence(image)
        
        return {
            "success": True,
            "text": ocr_result["text"],
            "confidence": ocr_result["confidence"],
            "word_count": len(ocr_result["text"].split()),
            "detected_languages": ocr_result.get("languages", []),
            "text_regions": ocr_result.get("regions", [])
        }
        
    except Exception as e:
        logger.error(f"OCR analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

def extract_text_with_confidence(image):
    
    try:
        # Convert PIL to numpy array
        img_array = np.array(image)
        
        # Preprocess image for better OCR results
        preprocessed = preprocess_image_for_ocr(img_array)
        
        # Get EasyOCR reader
        reader = get_ocr_reader()
        
        # Perform OCR with EasyOCR (returns list of [bbox, text, confidence])
        results = reader.readtext(preprocessed)
        
        # Extract text and confidence
        text_parts = []
        confidences = []
        regions = []
        
        for (bbox, text, confidence) in results:
            if text.strip() and confidence > 0:
                text_parts.append(text)
                confidences.append(confidence)
                
                # Convert bbox to standard format
                bbox_list = [list(point) for point in bbox]
                regions.append({
                    "text": text,
                    "confidence": round(confidence, 4),
                    "bbox": bbox_list
                })
        
        full_text = ' '.join(text_parts)
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return {
            "text": full_text,
            "confidence": round(avg_confidence, 4),
            "languages": ["en", "hi"],  # EasyOCR supports both
            "regions": regions
        }
        
    except Exception as e:
        logger.error(f"Text extraction error: {str(e)}")
        return {
            "text": "",
            "confidence": 0.0,
            "languages": [],
            "regions": []
        }

def preprocess_image_for_ocr(image):
    
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Noise removal
        denoised = cv2.medianBlur(gray, 3)
        
        # Thresholding
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
        
    except Exception as e:
        logger.error(f"Image preprocessing error: {str(e)}")
        return image
