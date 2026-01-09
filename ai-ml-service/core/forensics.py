from PIL import Image, ImageChops
import numpy as np

class ForensicAnalyzer:
    """
    Layer 4: Physical & Digital Forensic Anomaly Detection.
    No heavy AI, uses traditional signal processing.
    """

    @staticmethod
    def detect_ela(image_path: str, quality: int = 90) -> float:
        """
        Error Level Analysis (ELA).
        Detects if parts of an image have different compression levels (Photoshop).
        Returns a score: Higher means more likely tampered.
        """
        original = Image.open(image_path)
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
        """
        Detects 'Moire Patterns' using Fast Fourier Transform.
        Proof that a photo was taken of a digital screen.
        """
        # Simplified logic: High frequency noise patterns usually indicate pixels
        img = Image.open(image_path).convert('L')
        img_np = np.array(img)
        
        f = np.fft.fft2(img_np)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift))
        
        # Heuristic: Screen photos have distinct repetitive peaks in frequency domain
        # This is a baseline - actual thresholds depend on dataset
        return np.mean(magnitude_spectrum) > 150 # Dummy threshold
