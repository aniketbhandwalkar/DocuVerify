from fastapi import APIRouter, File, UploadFile, HTTPException
import cv2
import numpy as np
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/detect-signature")
async def detect_signature_in_document(file: UploadFile = File(...)):
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Only image files are supported")
        
        # Read file content
        content = await file.read()
        
        # Load image
        image = Image.open(io.BytesIO(content))
        img_array = np.array(image)
        
        # Detect signatures
        signature_result = find_signatures(img_array)
        
        return {
            "success": True,
            "signature_detected": signature_result["found"],
            "signature_count": signature_result["count"],
            "signature_regions": signature_result["regions"],
            "confidence": signature_result["confidence"]
        }
        
    except Exception as e:
        logger.error(f"Signature detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Signature detection failed: {str(e)}")

def find_signatures(image):
    
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        signature_regions = []
        signature_count = 0
        total_confidence = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter contours based on area (signatures typically have certain size range)
            if 1000 < area < 50000:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Calculate contour properties
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                
                # Calculate extent
                rect_area = w * h
                extent = area / rect_area if rect_area > 0 else 0
                
                # Signature characteristics:
                # - Aspect ratio between 0.5 and 4.0
                # - Solidity between 0.3 and 0.8 (irregular shape)
                # - Extent between 0.3 and 0.8
                signature_confidence = calculate_signature_confidence(
                    aspect_ratio, solidity, extent, area
                )
                
                if signature_confidence > 0.5:
                    signature_regions.append({
                        "bbox": {
                            "x": int(x),
                            "y": int(y),
                            "width": int(w),
                            "height": int(h)
                        },
                        "area": int(area),
                        "confidence": signature_confidence,
                        "properties": {
                            "aspect_ratio": aspect_ratio,
                            "solidity": solidity,
                            "extent": extent
                        }
                    })
                    signature_count += 1
                    total_confidence += signature_confidence
        
        avg_confidence = total_confidence / signature_count if signature_count > 0 else 0
        
        return {
            "found": signature_count > 0,
            "count": signature_count,
            "regions": signature_regions,
            "confidence": avg_confidence
        }
        
    except Exception as e:
        logger.error(f"Signature finding error: {str(e)}")
        return {
            "found": False,
            "count": 0,
            "regions": [],
            "confidence": 0.0
        }

def calculate_signature_confidence(aspect_ratio, solidity, extent, area):
    
    try:
        confidence = 0.0
        
        # Aspect ratio score (signatures are usually wider than tall but not extremely so)
        if 0.5 <= aspect_ratio <= 4.0:
            ar_score = 1.0 - abs(aspect_ratio - 2.0) / 2.0  # Optimal around 2.0
            confidence += ar_score * 0.3
        
        # Solidity score (signatures have irregular boundaries)
        if 0.3 <= solidity <= 0.8:
            solidity_score = 1.0 - abs(solidity - 0.55) / 0.25  # Optimal around 0.55
            confidence += solidity_score * 0.4
        
        # Extent score (signatures don't fill their bounding rectangle completely)
        if 0.3 <= extent <= 0.8:
            extent_score = 1.0 - abs(extent - 0.55) / 0.25  # Optimal around 0.55
            confidence += extent_score * 0.2
        
        # Area score (signatures have reasonable size)
        if 2000 <= area <= 30000:
            area_score = 1.0 - abs(area - 15000) / 15000  # Optimal around 15000
            confidence += area_score * 0.1
        
        return min(confidence, 1.0)
        
    except Exception as e:
        logger.error(f"Confidence calculation error: {str(e)}")
        return 0.0

@router.post("/extract-signature")
async def extract_signature_from_document(file: UploadFile = File(...)):
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Only image files are supported")
        
        # Read file content
        content = await file.read()
        
        # Load image
        image = Image.open(io.BytesIO(content))
        img_array = np.array(image)
        
        # Find signatures
        signature_result = find_signatures(img_array)
        
        # Extract signature regions
        extracted_signatures = []
        
        for i, region in enumerate(signature_result["regions"]):
            bbox = region["bbox"]
            
            # Extract region from original image
            signature_crop = img_array[
                bbox["y"]:bbox["y"] + bbox["height"],
                bbox["x"]:bbox["x"] + bbox["width"]
            ]
            
            # Convert to PIL Image
            signature_img = Image.fromarray(signature_crop)
            
            # Convert to base64 for response
            buffer = io.BytesIO()
            signature_img.save(buffer, format='PNG')
            import base64
            signature_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            extracted_signatures.append({
                "signature_id": i + 1,
                "bbox": bbox,
                "confidence": region["confidence"],
                "image_data": signature_base64
            })
        
        return {
            "success": True,
            "signature_count": len(extracted_signatures),
            "signatures": extracted_signatures
        }
        
    except Exception as e:
        logger.error(f"Signature extraction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Signature extraction failed: {str(e)}")
