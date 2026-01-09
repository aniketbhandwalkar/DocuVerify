"""
Digital Signature Verifier for Aadhaar Secure QR

Verifies the RSA digital signature embedded in Aadhaar Secure QR codes
using UIDAI's public certificate.
"""

import os
import logging
from typing import Optional
import hashlib

from .models import SignatureResult

logger = logging.getLogger(__name__)

# Try importing cryptography
try:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.backends import default_backend
    from cryptography.exceptions import InvalidSignature
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    logger.warning("cryptography library not available - signature verification disabled")


class SignatureVerifier:
    """
    Verifies digital signatures of Aadhaar Secure QR codes using UIDAI public certificate.
    
    The verification process:
    1. Load UIDAI public certificate (X.509)
    2. Extract public key from certificate
    3. Hash the QR data using SHA-256
    4. Verify RSA-PKCS1v15 signature
    """
    
    # Path to bundled certificate
    CERT_PATH = os.path.join(os.path.dirname(__file__), 'certs', 'uidai_cert.pem')
    
    def __init__(self, cert_path: Optional[str] = None):
        """
        Initialize the signature verifier.
        
        Args:
            cert_path: Optional path to UIDAI certificate file.
                      Uses bundled certificate if not provided.
        """
        self.cert_path = cert_path or self.CERT_PATH
        self.public_key = None
        self.certificate = None
        self.cert_loaded = False
        
        self._load_certificate()
    
    def _load_certificate(self) -> bool:
        """Load the UIDAI public certificate."""
        if not CRYPTO_AVAILABLE:
            logger.error("Cannot load certificate - cryptography library not available")
            return False
        
        try:
            if os.path.exists(self.cert_path):
                with open(self.cert_path, 'rb') as f:
                    cert_data = f.read()
                
                # Try PEM format
                try:
                    self.certificate = x509.load_pem_x509_certificate(
                        cert_data, default_backend()
                    )
                except:
                    # Try DER format
                    self.certificate = x509.load_der_x509_certificate(
                        cert_data, default_backend()
                    )
                
                self.public_key = self.certificate.public_key()
                self.cert_loaded = True
                logger.info(f"UIDAI certificate loaded from: {self.cert_path}")
                return True
            else:
                logger.warning(f"Certificate file not found: {self.cert_path}")
                # Create dummy certificate for academic demonstration
                self._create_demo_key()
                return True
                
        except Exception as e:
            logger.error(f"Failed to load certificate: {e}")
            self._create_demo_key()
            return False
    
    def _create_demo_key(self):
        """Create a demo RSA key for academic demonstration when real cert is unavailable."""
        if not CRYPTO_AVAILABLE:
            return
        
        try:
            # Generate a demo RSA key (NOT for production use)
            from cryptography.hazmat.primitives.asymmetric import rsa
            self.public_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            ).public_key()
            self.cert_loaded = True
            logger.warning("Using DEMO key - Real UIDAI certificate not available")
        except Exception as e:
            logger.error(f"Failed to create demo key: {e}")
    
    def verify_signature(self, signature_bytes: bytes, data_bytes: bytes) -> SignatureResult:
        """
        Verify the digital signature of Aadhaar QR data.
        
        Args:
            signature_bytes: The 256-byte RSA signature from QR
            data_bytes: The data that was signed (compressed demographic data)
            
        Returns:
            SignatureResult with verification status and explanation
        """
        if not CRYPTO_AVAILABLE:
            return SignatureResult(
                is_valid=False,
                verification_method="unavailable",
                error_message="cryptography library not installed",
                explanation="Digital signature verification requires the 'cryptography' library."
            )
        
        if not self.cert_loaded or not self.public_key:
            return SignatureResult(
                is_valid=False,
                verification_method="unavailable",
                error_message="Certificate not loaded",
                explanation="UIDAI public certificate could not be loaded for verification."
            )
        
        if not signature_bytes or len(signature_bytes) != 256:
            return SignatureResult(
                is_valid=False,
                verification_method="RSA-PKCS1v15",
                error_message=f"Invalid signature length: {len(signature_bytes) if signature_bytes else 0}",
                explanation="Aadhaar Secure QR signature must be exactly 256 bytes (2048-bit RSA)."
            )
        
        if not data_bytes:
            return SignatureResult(
                is_valid=False,
                verification_method="RSA-PKCS1v15",
                error_message="No data to verify",
                explanation="The QR data to be verified is missing."
            )
        
        try:
            # Verify the signature using RSA-PKCS1v15 with SHA-256
            self.public_key.verify(
                signature_bytes,
                data_bytes,
                padding.PKCS1v15(),
                hashes.SHA256()
            )
            
            # If we get here, verification succeeded
            return SignatureResult(
                is_valid=True,
                verification_method="RSA-PKCS1v15",
                hash_algorithm="SHA-256",
                signature_algorithm="RSA-2048",
                certificate_used=self.cert_path if os.path.exists(self.cert_path) else "demo_key",
                explanation=self._get_success_explanation()
            )
            
        except InvalidSignature:
            return SignatureResult(
                is_valid=False,
                verification_method="RSA-PKCS1v15",
                hash_algorithm="SHA-256",
                signature_algorithm="RSA-2048",
                error_message="Signature verification failed",
                explanation=self._get_failure_explanation()
            )
            
        except Exception as e:
            logger.error(f"Signature verification error: {e}")
            return SignatureResult(
                is_valid=False,
                verification_method="RSA-PKCS1v15",
                error_message=str(e),
                explanation="An error occurred during signature verification."
            )
    
    def verify_qr_raw(self, raw_qr_data: bytes) -> SignatureResult:
        """
        Verify signature from raw QR data (extracts signature and data automatically).
        
        The Aadhaar Secure QR structure:
        - Byte 0: Version/flags
        - Bytes 1-256: Digital signature
        - Bytes 257+: Compressed demographic data
        
        Args:
            raw_qr_data: Complete raw QR data bytes
            
        Returns:
            SignatureResult with verification status
        """
        if not raw_qr_data or len(raw_qr_data) < 260:
            return SignatureResult(
                is_valid=False,
                verification_method="RSA-PKCS1v15",
                error_message=f"QR data too short: {len(raw_qr_data) if raw_qr_data else 0} bytes",
                explanation="Aadhaar Secure QR must contain at least 260 bytes (version + signature + data)."
            )
        
        # Extract signature and data
        signature = raw_qr_data[1:257]  # 256 bytes after version byte
        data = raw_qr_data[257:]  # Remaining is the signed data
        
        return self.verify_signature(signature, data)
    
    def _get_success_explanation(self) -> str:
        """Get explanation for successful verification."""
        return (
            "Digital signature verification PASSED. This cryptographic proof confirms:\n"
            "1. The QR code data was generated by UIDAI (holder of the private key)\n"
            "2. The data has not been modified since issuance\n"
            "3. The card is authentic (not necessarily that the Aadhaar is active in UIDAI DB)\n\n"
            "Technical details:\n"
            "- Hash: SHA-256 computed over the demographic data\n"
            "- Signature: RSA-2048 with PKCS#1 v1.5 padding\n"
            "- Certificate: UIDAI public key for Secure QR verification"
        )
    
    def _get_failure_explanation(self) -> str:
        """Get explanation for failed verification."""
        return (
            "Digital signature verification FAILED. Possible reasons:\n"
            "1. The QR code data has been tampered with\n"
            "2. The QR code was not generated by UIDAI\n"
            "3. The QR code is corrupted or damaged\n"
            "4. Using incorrect/outdated UIDAI public certificate\n\n"
            "This indicates the Aadhaar card may be FORGED or TAMPERED."
        )
    
    def get_certificate_info(self) -> dict:
        """Get information about the loaded certificate."""
        if not self.certificate:
            return {"status": "not_loaded"}
        
        try:
            return {
                "status": "loaded",
                "subject": str(self.certificate.subject),
                "issuer": str(self.certificate.issuer),
                "serial_number": str(self.certificate.serial_number),
                "not_valid_before": self.certificate.not_valid_before_utc.isoformat(),
                "not_valid_after": self.certificate.not_valid_after_utc.isoformat(),
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}


# Singleton instance
signature_verifier = SignatureVerifier()
