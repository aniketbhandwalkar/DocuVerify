from PIL import Image, ImageChops
import numpy as np

class ForensicAnalyzer:
    

    @staticmethod
    def detect_ela(image_path: str, quality: int = 90) -> float:
        
        original = Image.open(image_path).convert("RGB")
        tmp_path = "tmp_resave.jpg"
        original.save(tmp_path, "JPEG", quality=quality)
        
        resaved = Image.open(tmp_path)
        ela_img = ImageChops.difference(original, resaved)
        
        extrema = ela_img.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        
        # Cleanup
        original.close()
        resaved.close()
        
        return float(max_diff)

    @staticmethod
    def detect_rephotography(image_path: str) -> bool:
        
        # Simplified logic: High frequency noise patterns usually indicate pixels
        img = Image.open(image_path).convert('L')
        img_np = np.array(img)
        
        f = np.fft.fft2(img_np)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift))
        
        # Relaxed threshold: Screen photos have distinct repetitive peaks
        # 150 was too tight, trying 200.
        score = np.mean(magnitude_spectrum)
        return score > 200 
