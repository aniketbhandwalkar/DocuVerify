from fastapi import APIRouter, File, UploadFile, Form, HTTPException
import cv2
import numpy as np
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/validate-format")
async def validate_document_format(
    file: UploadFile = File(...),
    document_type: str = Form(...)
):
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Only image files are supported")
        
        # Read file content
        content = await file.read()
        
        # Load image
        image = Image.open(io.BytesIO(content))
        img_array = np.array(image)
        
        # Perform format validation
        validation_result = perform_format_validation(img_array, document_type)
        
        return {
            "success": True,
            "document_type": document_type,
            "is_valid_format": validation_result["is_valid"],
            "format_score": validation_result["score"],
            "validation_details": validation_result["details"],
            "recommendations": validation_result["recommendations"]
        }
        
    except Exception as e:
        logger.error(f"Format validation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Format validation failed: {str(e)}")

def perform_format_validation(image, document_type):
    
    validation_result = {
        "is_valid": False,
        "score": 0.0,
        "details": {},
        "recommendations": []
    }
    
    try:
        height, width = image.shape[:2]
        aspect_ratio = float(width) / float(height)
        
        # Get document type specifications
        doc_specs = get_document_specifications(document_type)
        
        # Validate dimensions
        dimension_score = validate_dimensions(width, height, doc_specs)
        
        # Validate aspect ratio
        aspect_score = validate_aspect_ratio(aspect_ratio, doc_specs)
        
        # Validate image quality
        quality_score = validate_image_quality(image)
        
        # Validate structure elements
        structure_score = validate_document_structure(image, document_type)
        
        # Calculate overall score
        overall_score = (
            float(dimension_score) * 0.25 +
            float(aspect_score) * 0.25 +
            float(quality_score) * 0.25 +
            float(structure_score) * 0.25
        )
        
        validation_result.update({
            "is_valid": bool(overall_score > 0.6),
            "score": float(overall_score),
            "details": {
                "dimensions": {
                    "width": int(width),
                    "height": int(height),
                    "score": float(dimension_score)
                },
                "aspect_ratio": {
                    "value": float(aspect_ratio),
                    "score": float(aspect_score)
                },
                "quality": {
                    "score": float(quality_score)
                },
                "structure": {
                    "score": float(structure_score)
                }
            }
        })
        
        # Generate recommendations
        recommendations = generate_recommendations(validation_result["details"], doc_specs)
        validation_result["recommendations"] = recommendations
        
    except Exception as e:
        logger.error(f"Format validation processing error: {str(e)}")
        validation_result["details"]["error"] = str(e)
    
    return validation_result

def get_document_specifications(document_type):
    
    specifications = {
        "passport": {
            "aspect_ratio_range": (1.3, 1.5),
            "min_dimensions": (600, 800),
            "recommended_dimensions": (1200, 1600),
            "required_elements": ["photo", "text_regions", "machine_readable_zone"],
            "color_requirements": "color_preferred"
        },
        "id-card": {
            "aspect_ratio_range": (1.5, 1.7),
            "min_dimensions": (500, 800),
            "recommended_dimensions": (1000, 1600),
            "required_elements": ["photo", "text_regions", "id_number"],
            "color_requirements": "color_preferred"
        },
        "driver-license": {
            "aspect_ratio_range": (1.5, 1.8),
            "min_dimensions": (500, 800),
            "recommended_dimensions": (1000, 1600),
            "required_elements": ["photo", "license_number", "address"],
            "color_requirements": "color_required"
        },
        "certificate": {
            "aspect_ratio_range": (1.2, 1.4),
            "min_dimensions": (800, 1000),
            "recommended_dimensions": (1600, 2000),
            "required_elements": ["title", "name", "date", "signature"],
            "color_requirements": "any"
        }
    }
    
    return specifications.get(document_type, {
        "aspect_ratio_range": (1.0, 2.0),
        "min_dimensions": (400, 600),
        "recommended_dimensions": (800, 1200),
        "required_elements": [],
        "color_requirements": "any"
    })

def validate_dimensions(width, height, specs):
    
    try:
        min_w, min_h = specs.get("min_dimensions", (400, 600))
        rec_w, rec_h = specs.get("recommended_dimensions", (800, 1200))
        
        # Check minimum requirements
        if width < min_w or height < min_h:
            return 0.0
        
        # Score based on how close to recommended dimensions
        width_score = min(float(width) / float(rec_w), 1.0)
        height_score = min(float(height) / float(rec_h), 1.0)
        
        return float((width_score + height_score) / 2)
        
    except Exception as e:
        logger.error(f"Dimension validation error: {str(e)}")
        return 0.0

def validate_aspect_ratio(aspect_ratio, specs):
    
    try:
        ar_min, ar_max = specs.get("aspect_ratio_range", (1.0, 2.0))
        
        if ar_min <= aspect_ratio <= ar_max:
            # Score based on how close to optimal (middle of range)
            optimal = (ar_min + ar_max) / 2
            deviation = abs(aspect_ratio - optimal) / (ar_max - ar_min)
            return float(1.0 - deviation)
        else:
            return 0.0
            
    except Exception as e:
        logger.error(f"Aspect ratio validation error: {str(e)}")
        return 0.0

def validate_image_quality(image):
    
    try:
        # Convert to grayscale for quality assessment
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Calculate sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 1000, 1.0)
        
        # Calculate brightness
        brightness = np.mean(gray) / 255.0
        brightness_score = 1.0 - abs(brightness - 0.5) * 2  # Optimal around 0.5
        
        # Calculate contrast
        contrast = np.std(gray) / 255.0
        contrast_score = min(contrast * 4, 1.0)  # Good contrast > 0.25
        
        # Combine quality metrics
        quality_score = (sharpness_score * 0.4 + brightness_score * 0.3 + contrast_score * 0.3)
        
        return quality_score
        
    except Exception as e:
        logger.error(f"Quality validation error: {str(e)}")
        return 0.0

def validate_document_structure(image, document_type):
    
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect text regions
        text_regions = detect_text_regions(gray)
        
        # Detect potential photo regions
        photo_regions = detect_photo_regions(gray)
        
        # Score based on expected structure
        structure_score = 0.0
        
        if document_type in ["passport", "id-card", "driver-license"]:
            # These documents should have both text and photo regions
            if text_regions > 0:
                structure_score += 0.5
            if photo_regions > 0:
                structure_score += 0.5
        elif document_type == "certificate":
            # Certificates mainly have text
            if text_regions > 0:
                structure_score += 0.8
            # May have signature regions
            structure_score += 0.2  # Bonus for any additional elements
        else:
            # Generic document
            if text_regions > 0:
                structure_score += 0.7
            structure_score += 0.3  # Bonus for any structure
        
        return min(structure_score, 1.0)
        
    except Exception as e:
        logger.error(f"Structure validation error: {str(e)}")
        return 0.0

def detect_text_regions(gray_image):
    
    try:
        # Apply threshold
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 10000:  # Potential text region size
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 0.2 < aspect_ratio < 10:  # Text-like aspect ratio
                    text_regions += 1
        
        return text_regions
        
    except Exception as e:
        logger.error(f"Text region detection error: {str(e)}")
        return 0

def detect_photo_regions(gray_image):
    
    try:
        # Look for rectangular regions with specific properties
        edges = cv2.Canny(gray_image, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        photo_regions = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 5000 < area < 100000:  # Photo-like area
                # Check if it's roughly rectangular
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                if len(approx) == 4:  # Rectangular
                    photo_regions += 1
        
        return photo_regions
        
    except Exception as e:
        logger.error(f"Photo region detection error: {str(e)}")
        return 0

def generate_recommendations(validation_details, specs):
    
    recommendations = []
    
    try:
        # Dimension recommendations
        if validation_details["dimensions"]["score"] < 0.7:
            rec_w, rec_h = specs.get("recommended_dimensions", (800, 1200))
            recommendations.append(f"Consider using higher resolution image (recommended: {rec_w}x{rec_h})")
        
        # Aspect ratio recommendations
        if validation_details["aspect_ratio"]["score"] < 0.7:
            ar_min, ar_max = specs.get("aspect_ratio_range", (1.0, 2.0))
            recommendations.append(f"Adjust aspect ratio to be between {ar_min:.1f}:{ar_max:.1f}")
        
        # Quality recommendations
        if validation_details["quality"]["score"] < 0.6:
            recommendations.append("Improve image quality: ensure good lighting and focus")
        
        # Structure recommendations
        if validation_details["structure"]["score"] < 0.5:
            recommendations.append("Ensure all document elements are visible and properly framed")
        
        if not recommendations:
            recommendations.append("Document format looks good!")
        
    except Exception as e:
        logger.error(f"Recommendation generation error: {str(e)}")
        recommendations.append("Unable to generate specific recommendations")
    
    return recommendations
