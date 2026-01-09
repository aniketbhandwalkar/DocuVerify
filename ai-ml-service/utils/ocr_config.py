"""
Unified OCR Configuration and Management

This module provides a centralized OCR configuration system to manage
multiple OCR engines (Tesseract, EasyOCR) with proper fallback mechanisms.
"""

import os
import platform
import logging
from typing import Dict, List, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class OCREngine(Enum):
    """Available OCR engines"""
    TESSERACT = "tesseract"
    EASYOCR = "easyocr"
    HYBRID = "hybrid"  # Uses Tesseract for numbers, EasyOCR for general text


class OCRConfig:
    """
    Centralized OCR configuration manager
    
    Features:
    - Automatic Tesseract path detection
    - EasyOCR initialization with fallback
    - Environment variable support
    - Health checking for OCR engines
    """
    
    def __init__(self):
        self.tesseract_available = False
        self.easyocr_available = False
        self.tesseract_path = None
        
        # Initialize configurations
        self._configure_tesseract()
        self._configure_easyocr()
        
    def _configure_tesseract(self) -> bool:
        """Configure Tesseract OCR with platform-specific handling"""
        try:
            import pytesseract
            
            if platform.system() == 'Windows':
                # Check environment variable first
                tesseract_path = os.getenv('TESSERACT_PATH')
                
                if not tesseract_path:
                    # Try common installation paths
                    common_paths = [
                        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                        r'C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe'.format(
                            os.getenv('USERNAME', 'AppData')
                        ),
                    ]
                    
                    for path in common_paths:
                        if os.path.exists(path):
                            tesseract_path = path
                            break
                
                if tesseract_path and os.path.exists(tesseract_path):
                    pytesseract.pytesseract.tesseract_cmd = tesseract_path
                    self.tesseract_path = tesseract_path
                    self.tesseract_available = True
                    logger.info(f"Tesseract configured: {tesseract_path}")
                    return True
                else:
                    logger.warning(
                        "Tesseract not found. Set TESSERACT_PATH environment variable or "
                        "install from https://github.com/UB-Mannheim/tesseract/wiki"
                    )
                    return False
            else:
                # Linux/Mac - should be in PATH
                try:
                    import subprocess
                    subprocess.run(['tesseract', '--version'], 
                                 capture_output=True, check=True)
                    self.tesseract_available = True
                    logger.info("Tesseract found in system PATH")
                    return True
                except (subprocess.CalledProcessError, FileNotFoundError):
                    logger.warning("Tesseract not found in system PATH")
                    return False
                    
        except ImportError:
            logger.warning("pytesseract not installed. Install with: pip install pytesseract")
            return False
    
    def _configure_easyocr(self) -> bool:
        """Configure EasyOCR with lazy initialization"""
        try:
            import easyocr
            logger.info("EasyOCR available (will be initialized on first use)")
            self.easyocr_available = True
            return True
        except ImportError:
            logger.warning("EasyOCR not installed. Install with: pip install easyocr")
            return False
    
    def get_available_engines(self) -> List[str]:
        """Get list of available OCR engines"""
        engines = []
        if self.tesseract_available:
            engines.append(OCREngine.TESSERACT.value)
        if self.easyocr_available:
            engines.append(OCREngine.EASYOCR.value)
        return engines
    
    def get_recommended_engine(self) -> str:
        """Get recommended OCR engine based on availability"""
        # Tesseract is faster, prefer it if available
        if self.tesseract_available:
            return OCREngine.TESSERACT.value
        elif self.easyocr_available:
            return OCREngine.EASYOCR.value
        else:
            raise RuntimeError(
                "No OCR engine available. Install Tesseract or EasyOCR. "
                "Install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki"
            )
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all OCR engines"""
        return {
            "tesseract": self.tesseract_available,
            "easyocr": self.easyocr_available,
            "has_any_engine": self.tesseract_available or self.easyocr_available
        }


# Global configuration instance
_ocr_config = None


def get_ocr_config() -> OCRConfig:
    """Get or create global OCR configuration"""
    global _ocr_config
    if _ocr_config is None:
        _ocr_config = OCRConfig()
    return _ocr_config


def get_available_ocr_engines() -> List[str]:
    """Get list of available OCR engines"""
    return get_ocr_config().get_available_engines()


def get_recommended_ocr_engine() -> str:
    """Get recommended OCR engine"""
    return get_ocr_config().get_recommended_engine()


def ocr_health_check() -> Dict[str, bool]:
    """Get OCR health check information"""
    return get_ocr_config().health_check()
