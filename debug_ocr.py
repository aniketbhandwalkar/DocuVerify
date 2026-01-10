
import cv2
import sys
import os
import numpy as np

# Add the ai-ml-service directory to path
sys.path.append('e:/DocumentVerify/ai-ml-service')

from utils.aadhaar_extractor import AadhaarOCRExtractor
from aadhaar_verifier.image_preprocessor import image_preprocessor

def debug_ocr(path):
    print(f"\n--- Debugging OCR: {path} ---")
    image = cv2.imread(path)
    if image is None:
        print("Error: Could not load image")
        return
        
    # Preprocess
    preprocessed = image_preprocessor.preprocess(image)
    proc_img = preprocessed.processed_image
    print(f"Post-preprocessing shape: {proc_img.shape}")
    print(f"Notes: {preprocessed.preprocessing_notes}")
    
    # Save processed image for manual inspection if needed
    # cv2.imwrite('debug_processed.jpg', proc_img)
    
    extractor = AadhaarOCRExtractor()
    result = extractor.extract_aadhaar_data(proc_img, preprocess=True)
    
    print("\n[OCR RESULT]")
    print(f"Success: {result.get('success')}")
    print(f"Aadhaar: {result.get('aadhaar_number')}")
    print(f"Valid: {result.get('aadhaar_valid')}")
    print(f"Name: {result.get('name')}")
    print(f"DOB: {result.get('dob')}")
    print(f"Gender: {result.get('gender')}")
    print(f"Address: {result.get('address')}")
    print(f"Full Text: {result.get('full_text')}")

if __name__ == "__main__":
    path = "C:/Users/Aniket/.gemini/antigravity/brain/69b68fe7-6179-4b60-b0eb-8402c1795d65/uploaded_image_1767986119849.jpg"
    debug_ocr(path)
