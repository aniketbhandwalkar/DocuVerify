import cv2
import numpy as np
import easyocr
import re
import os
import logging
from typing import Dict, Any, Tuple, List
from core.plausibility_engine import PlausibilityEngine
from core.crypto import CryptographicValidator
from core.forensics import ForensicAnalyzer

logger = logging.getLogger(__name__)

class SmartAadhaarVerifier:
    def __init__(self):
        # Initialize OCR (First time will download a small model)
        # We share the reader instance to save memory
        self.reader = easyocr.Reader(['en', 'hi'], gpu=False) 
        self.plausibility = PlausibilityEngine()
        self.forensics = ForensicAnalyzer()
        self.crypto = CryptographicValidator()

    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Performs the 5-layer Smart Plausibility Assessment.
        """
        total_score = 0
        findings = []
        extracted_data = {}

        # 1. Forensic Layer
        print("🔍 [Layer 1] Forensic Analysis: Detection Screen/Editing...")
        ela_score = self.forensics.detect_ela(image_path)
        is_rephoto = self.forensics.detect_rephotography(image_path)
        
        if is_rephoto:
            findings.append("🚨 RE-PHOTOGRAPHY: Moire patterns detected.")
        else:
            total_score += 15
            findings.append("✅ CAPTURE: No obvious moire patterns.")

        if ela_score > 50:
            findings.append(f"🚨 TAMPERING: High ELA Variance ({ela_score}).")
        else:
            total_score += 15
            findings.append("✅ COMPRESSION: Consistent error levels.")

        # 2. Cryptographic Layer
        print("🔍 [Layer 2] Cryptographic Scan: Decoding QR Code...")
        qr_result = self.crypto.decode_qr(image_path)
        if qr_result["success"]:
            total_score += 30
            findings.append(f"✅ QR PROOF: Found {qr_result['type']}.")
            qr_data = qr_result["data"]
            extracted_data["qr_info"] = qr_data
        else:
            findings.append(f"🚨 QR MISSING.")
            qr_data = None

        # 3. Visual Layer
        print("🔍 [Layer 3] Visual Analysis: Extracting Text (EasyOCR)...")
        results = self.reader.readtext(image_path)
        candidates = []
        for (bbox, text, prob) in results:
            clean_digits = re.sub(r'[^0-9]', '', text)
            if len(clean_digits) >= 12:
                for i in range(len(clean_digits) - 11):
                    candidates.append(clean_digits[i:i+12])

        full_text = " ".join([res[1] for res in results])
        all_potential_12 = re.findall(r'\d{12}', re.sub(r'[^0-9]', '', full_text))
        candidates.extend(all_potential_12)

        aadhaar_num = None
        valid_found = False
        
        for cand in set(candidates):
            if self.plausibility.verhoeff.validate(cand):
                aadhaar_num = cand
                valid_found = True
                break
        
        if not valid_found and candidates:
            aadhaar_num = candidates[0]

        if aadhaar_num:
            print(f"🔍 [Layer 4] Structural Analysis: Checking Aadhaar {aadhaar_num}...")
            score, msg = self.plausibility.assess_aadhaar_number(aadhaar_num)
            total_score += score
            findings.append(f"✅ STRUCTURE: {msg} ({aadhaar_num})")
            extracted_data["aadhaar_number"] = aadhaar_num
            extracted_data["number_valid"] = valid_found
        else:
            findings.append("🚨 STRUCTURE: No valid pattern.")

        # 4. Cross-Validation
        if qr_data and aadhaar_num:
            print("🔍 [Layer 5] Cross-Validation: Comparing QR vs OCR...")
            score, msg = self.crypto.verify_qr_consistency(qr_data, {"number": aadhaar_num})
            total_score += score
            findings.append(f"✅ CROSS-CHECK: {msg}")

        print(f"🏁 Analysis Complete. Final Score: {total_score}/100")

        # Verdict logic
        is_valid = total_score > 40
        verdict = "ACCEPT" if is_valid else "REJECT"
        
        return {
            "success": True,
            "verdict": verdict,
            "is_valid": bool(is_valid),
            "confidence": int(total_score),
            "reason": "; ".join(findings),
            "findings": findings,
            "extracted_data": extracted_data,
            "ela_score": float(ela_score),
            "is_rephoto": bool(is_rephoto)
        }

# Singleton
smart_verifier = SmartAadhaarVerifier()
