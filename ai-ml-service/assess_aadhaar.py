import asyncio
import sys
import easyocr
import os
from core.plausibility_engine import PlausibilityEngine
from core.crypto import CryptographicValidator
from core.forensics import ForensicAnalyzer

class AadhaarPlausibilityAssessment:
    def __init__(self):
        # Initialize OCR (First time will download a small model - ~20MB)
        self.reader = easyocr.Reader(['en', 'hi'], gpu=False) 
        self.plausibility = PlausibilityEngine()
        self.forensics = ForensicAnalyzer()
        self.crypto = CryptographicValidator()

    async def run_report(self, image_path: str):
        print(f"\n[ forensic report for: {os.path.basename(image_path)} ]")
        print("-" * 50)
        
        total_score = 0
        findings = []

        # 1. Forensic Layer (Screen/Photoshop Detection)
        print("Layer 4: Forensic Analysis...")
        ela_score = self.forensics.detect_ela(image_path)
        is_rephoto = self.forensics.detect_rephotography(image_path)
        
        if is_rephoto:
            findings.append("🚨 RE-PHOTOGRAPHY: Moire patterns detected (Possible screen capture).")
        else:
            total_score += 15
            findings.append("✅ CAPTURE: No obvious moire patterns detected.")

        if ela_score > 50:
            findings.append(f"🚨 TAMPERING: High ELA Variance ({ela_score}). Evidence of digital editing.")
        else:
            total_score += 15
            findings.append("✅ COMPRESSION: Consistent error levels (No obvious Photoshop).")

        # 2. Cryptographic Layer (QR Decode)
        print("Layer 2: Cryptographic Decoding...")
        qr_result = self.crypto.decode_qr(image_path)
        if qr_result["success"]:
            total_score += 30
            findings.append(f"✅ QR PROOF: Found {qr_result['type']}. Cryptographic data extracted.")
            qr_data = qr_result["data"]
        else:
            findings.append(f"🚨 QR MISSING: {qr_result['error']}.")
            qr_data = None

        # 3. Visual Layer (OCR)
        print("Layer 3: Visual Text Extraction...")
        results = self.reader.readtext(image_path)
        
        candidates = []
        for (bbox, text, prob) in results:
            # Clean text to just digits
            clean_digits = re.sub(r'[^0-9]', '', text)
            if len(clean_digits) >= 12:
                # Often OCR merges 12 digits or gets them in a block
                for i in range(len(clean_digits) - 11):
                    candidates.append(clean_digits[i:i+12])
            elif len(text) >= 4 and text.isdigit():
                # Sometimes they are read as 3 separate 4-digit blocks
                pass # We'll handle full_text below too

        # Also search in the merged text 
        full_text = " ".join([res[1] for res in results])
        all_potential_12 = re.findall(r'\d{12}', re.sub(r'[^0-9]', '', full_text))
        candidates.extend(all_potential_12)

        aadhaar_num = None
        valid_found = False
        
        # Test each candidate with Verhoeff
        for cand in set(candidates):
            if self.plausibility.verhoeff.validate(cand):
                aadhaar_num = cand
                valid_found = True
                break
        
        # If no valid checksum, pick the most likely 12-digit string
        if not valid_found and candidates:
            aadhaar_num = candidates[0]

        if aadhaar_num:
            score, msg = self.plausibility.assess_aadhaar_number(aadhaar_num)
            total_score += score
            findings.append(f"✅ STRUCTURE: {msg} ({aadhaar_num})")
        else:
            findings.append("🚨 STRUCTURE: No valid 12-digit Aadhaar pattern found in text.")

        # 4. Cross-Validation (Layer 5)
        if qr_data and 'aadhaar_num' in locals():
            # Basic last-digit comparison
            score, msg = self.crypto.verify_qr_consistency(qr_data, {"number": aadhaar_num})
            total_score += score
            findings.append(f"✅ CROSS-CHECK: {msg}")

        # Summary
        print("\n--- FINAL VERDICT ---")
        for f in findings:
            print(f"- {f}")
        
        # User defined threshold: > 40 is valid
        if total_score > 40:
            verdict = "VALID / LIKELY AUTHENTIC"
            color = "GREEN"
        else:
            verdict = "SUSPICIOUS / LIKELY FAKE"
            color = "RED"
        
        print("-" * 50)
        print(f"PLAUSIBILITY SCORE: {total_score}/100")
        print(f"RESULT: {verdict}")
        print("-" * 50)
        print("DISCLAIMER: NOT an official ID verification. Forensic risk assessment only.")

import re
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python assess_aadhaar.py <image_path>")
    else:
        assessment = AadhaarPlausibilityAssessment()
        asyncio.run(assessment.run_report(sys.argv[1]))
