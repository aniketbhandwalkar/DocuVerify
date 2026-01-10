import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import joblib
import os
import json
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import easyocr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    import torchvision.models as models
    TORCH_AVAILABLE = True
    logger.info("PyTorch available")
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, using alternative methods")

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
    logger.info("face_recognition available")
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    logger.warning("face_recognition not available, using OpenCV face detection")

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    logger.info("DeepFace available")
except ImportError:
    DEEPFACE_AVAILABLE = False
    logger.warning("DeepFace not available, using basic face analysis")

try:
    import magic
    MAGIC_AVAILABLE = True
    logger.info("python-magic available")
except ImportError:
    MAGIC_AVAILABLE = False
    logger.warning("python-magic not available, using basic file type detection")

if TORCH_AVAILABLE:
    class LogoDetectionCNN(torch.nn.Module):
        
        
        def __init__(self, num_classes=2):
            super(LogoDetectionCNN, self).__init__()
            self.features = torch.nn.Sequential(
                # First conv block
                torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2, 2),
                
                # Second conv block
                torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2, 2),
                
                # Third conv block
                torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2, 2),
                
                # Fourth conv block
                torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(2, 2),
            )
            
            self.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.5),
                torch.nn.Linear(256 * 4 * 4, 512),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(0.5),
                torch.nn.Linear(512, 128),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(128, num_classes)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    class SealDetectionCNN(torch.nn.Module):
        
        
        def __init__(self, num_classes=3):  # authentic, fake, no_seal
            super(SealDetectionCNN, self).__init__()
            # Use a pre-trained ResNet as backbone
            if TORCH_AVAILABLE:
                self.backbone = models.resnet18(pretrained=True)
                # Modify final layer
                self.backbone.fc = torch.nn.Linear(self.backbone.fc.in_features, num_classes)
            else:
                self.backbone = None
        
        def forward(self, x):
            if self.backbone is not None:
                return self.backbone(x)
            else:
                return torch.zeros(x.size(0), 3)  
else:
    
    class LogoDetectionCNN:
        def __init__(self, *args, **kwargs):
            pass
    
    class SealDetectionCNN:
        def __init__(self, *args, **kwargs):
            pass

class AdvancedDocumentVerifier:
    
    
    def __init__(self):
        try:
            self.ocr_reader = easyocr.Reader(['en', 'hi'])  # English and Hindi
            logger.info("EasyOCR reader initialized")
        except Exception as e:
            logger.warning(f"EasyOCR initialization failed: {e}")
            self.ocr_reader = None
            
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_extractors = {}
        self.template_database = {}
        self.face_database = {}
        
        
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            logger.info("OpenCV cascades loaded")
        except Exception as e:
            logger.warning(f"OpenCV cascade loading failed: {e}")
            self.face_cascade = None
            self.eye_cascade = None
        
        self.load_models()
        self.load_templates()
        self.setup_cnn_models()
    
    def setup_cnn_models(self):
        
        try:
            if not TORCH_AVAILABLE:
                logger.warning("PyTorch not available, CNN models disabled")
                self.logo_cnn = None
                self.seal_cnn = None
                return
            
            
            self.logo_cnn = LogoDetectionCNN(num_classes=2)
            self.logo_cnn.eval()
            
            
            self.seal_cnn = SealDetectionCNN(num_classes=3)
            self.seal_cnn.eval()
            
            
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            
            logo_model_path = 'models/logo_cnn.pth'
            seal_model_path = 'models/seal_cnn.pth'
            
            if os.path.exists(logo_model_path):
                try:
                    self.logo_cnn.load_state_dict(torch.load(logo_model_path, map_location='cpu'))
                    logger.info("Loaded pre-trained logo CNN model")
                except Exception as e:
                    logger.warning(f"Failed to load logo CNN weights: {e}")
            
            if os.path.exists(seal_model_path):
                try:
                    self.seal_cnn.load_state_dict(torch.load(seal_model_path, map_location='cpu'))
                    logger.info("Loaded pre-trained seal CNN model")
                except Exception as e:
                    logger.warning(f"Failed to load seal CNN weights: {e}")
            
            logger.info("PyTorch CNN models initialized successfully")
            
        except Exception as e:
            logger.error(f"CNN setup error: {e}")
            self.logo_cnn = None
            self.seal_cnn = None
    
    def load_templates(self):
        
        try:
            
            template_dir = 'templates'
            if os.path.exists(template_dir):
                for template_file in os.listdir(template_dir):
                    if template_file.endswith(('.png', '.jpg', '.jpeg')):
                        template_name = os.path.splitext(template_file)[0]
                        template_path = os.path.join(template_dir, template_file)
                        template_img = cv2.imread(template_path)
                        if template_img is not None:
                            self.template_database[template_name] = template_img
                            logger.info(f"Loaded template: {template_name}")
            
            
            os.makedirs(template_dir, exist_ok=True)
            
        except Exception as e:
            logger.error(f"Template loading error: {e}")
    
    def load_models(self):
        
        try:
            
            os.makedirs('models', exist_ok=True)
            
            self.models['svm'] = SVC(kernel='rbf', probability=True, random_state=42, C=10)
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            self.models['random_forest'] = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42
            )
            self.models['isolation_forest'] = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            
            
            for model_name in ['svm', 'xgboost', 'random_forest']:
                model_path = f'models/{model_name}_document_classifier.joblib'
                if os.path.exists(model_path):
                    self.models[model_name] = joblib.load(model_path)
                    logger.info(f"Loaded pre-trained {model_name} model")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def extract_comprehensive_features(self, image_path: str, document_type: str = "id-card") -> Dict[str, Any]:
        
        try:
            
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not read image")
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            features = {
                'image_path': image_path,
                'document_type': document_type,
                'timestamp': datetime.now().isoformat(),
                'image_hash': self.calculate_image_hash(image)
            }
            
            logger.info(f"Extracting features for {document_type} document")
            
            
            ocr_features = self.extract_ocr_features(image, gray)
            features.update(ocr_features)
            
            
            qr_features = self.extract_qr_features(image, gray)
            features.update(qr_features)
            
            
            if qr_features.get('aadhaar_decoded', False):
                cross_verify = AadhaarVerifier.verify_aadhaar_extracted_text(
                    ocr_features.get('extracted_text', ''), 
                    qr_features.get('aadhaar_data_full', {})
                )
                features['aadhaar_cross_verify'] = cross_verify
                if not cross_verify['verified']:
                    features['anomalies'] = features.get('anomalies', []) + [f"QR/Text Mismatch: {m}" for m in cross_verify['mismatches']]
                    # Penalize confidence significantly if mismatch found
                    logger.warning(f"Aadhaar Mismatch Detected: {cross_verify}")
            
            
            
            forensics_features = self.extract_forensics_features(image, gray)
            features.update(forensics_features)
            
            
            if document_type in ['id-card', 'passport', 'driver-license', 'aadhar-card']:
                face_features = self.extract_face_features(image)
                features.update(face_features)
            
            
            logo_features = self.extract_logo_features(image, gray, document_type)
            features.update(logo_features)
            
            
            metadata_features = self.extract_metadata_features(image_path)
            features.update(metadata_features)
            
            
            texture_features = self.extract_texture_features(gray)
            features.update(texture_features)
            
            
            color_features = self.extract_color_features(image)
            features.update(color_features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return {'error': str(e)}
    
    def calculate_image_hash(self, image: np.ndarray) -> str:
        
        try:
            
            resized = cv2.resize(image, (8, 8))
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            
            
            avg = np.mean(gray)
            
            
            hash_bits = []
            for pixel in gray.flatten():
                hash_bits.append('1' if pixel > avg else '0')
            
            
            hash_str = ''.join(hash_bits)
            hash_int = int(hash_str, 2)
            return format(hash_int, '016x')
            
        except Exception as e:
            logger.error(f"Image hash calculation error: {e}")
            return "0000000000000000"
    
    def extract_ocr_features(self, image: np.ndarray, gray: np.ndarray) -> Dict[str, Any]:
        
        try:
            features = {}
            
            # EasyOCR extraction
            if self.ocr_reader is not None:
                results = self.ocr_reader.readtext(image)
                features['easyocr_regions_count'] = len(results)
                easyocr_confidences = [result[2] for result in results if result[2] > 0.1]
                features['easyocr_confidence_mean'] = np.mean(easyocr_confidences) if easyocr_confidences else 0
            else:
                features['easyocr_regions_count'] = 0
                features['easyocr_confidence_mean'] = 0            
            # Tesseract OCR
            try:
                tesseract_text = pytesseract.image_to_string(gray)
                tesseract_conf = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
                
                confidences = [conf for conf in tesseract_conf['conf'] if conf > 0]
                features['ocr_confidence_mean'] = np.mean(confidences) if confidences else 0
                features['ocr_confidence_std'] = np.std(confidences) if confidences else 0
                
            except Exception as e:
                logger.warning(f"Tesseract OCR error: {e}")
                tesseract_text = ""
                features['ocr_confidence_mean'] = 0
                features['ocr_confidence_std'] = 0
            
            # OCR Statistics
            features['ocr_text_length'] = len(tesseract_text)
            features['ocr_word_count'] = len(tesseract_text.split())
            
            # Character type analysis
            if tesseract_text:
                alpha_chars = sum(1 for c in tesseract_text if c.isalpha())
                digit_chars = sum(1 for c in tesseract_text if c.isdigit())
                special_chars = sum(1 for c in tesseract_text if not c.isalnum() and not c.isspace())
                
                total_chars = len(tesseract_text)
                features['alpha_ratio'] = alpha_chars / total_chars
                features['digit_ratio'] = digit_chars / total_chars
                features['special_ratio'] = special_chars / total_chars
            else:
                features['alpha_ratio'] = 0
                features['digit_ratio'] = 0
                features['special_ratio'] = 0
            
            # Suspicious text detection
            suspicious_keywords = ['fake', 'fraud', 'sample', 'test', 'dummy', 'specimen', 'copy']
            features['suspicious_text_detected'] = any(keyword in tesseract_text.lower() for keyword in suspicious_keywords)
            
            # Store extracted text
            features['extracted_text'] = tesseract_text[:500]  
            
            return features
            
        except Exception as e:
            logger.error(f"OCR feature extraction error: {e}")
            return {
                'ocr_text_length': 0,
                'ocr_word_count': 0,
                'ocr_confidence_mean': 0,
                'easyocr_regions_count': 0,
                'alpha_ratio': 0,
                'digit_ratio': 0,
                'special_ratio': 0,
                'suspicious_text_detected': False,
                'extracted_text': ''
            }
    
    def extract_qr_features(self, image: np.ndarray, gray: np.ndarray) -> Dict[str, Any]:
        
        try:
            features = {}
            
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tf:
                cv2.imwrite(tf.name, image)
                temp_path = tf.name
                
            try:
                # Use our new dedicated decoder
                aadhaar_result = AadhaarVerifier.decode_aadhaar_qr(temp_path)
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            
            features['qr_code_count'] = 1 if aadhaar_result.get('success') else 0
            features['qr_codes_detected'] = aadhaar_result.get('success')
            features['aadhaar_decoded'] = aadhaar_result.get('success')
            features['aadhaar_data_full'] = aadhaar_result
            
            if aadhaar_result.get('success'):
                data = aadhaar_result.get('data', {})
                features['qr_data_length'] = len(str(data))
                features['aadhaar_qr_pattern'] = True
                features['aadhaar_uid_detected'] = 'uid' in data or 'referenceid' in data
                features['aadhaar_name_detected'] = 'name' in data
                features['qr_quality_mean'] = 1.0 # High confidence if decoded
            else:
                features['qr_data_length'] = 0
                features['aadhaar_qr_pattern'] = False
                features['aadhaar_uid_detected'] = False
                
                # Check error reason
                if "No QR code found" not in aadhaar_result.get('error', ''):
                    # QR found but failed to decode -> Potential Fake
                    features['suspicious_qr'] = True
            
            return features
            
        except Exception as e:
            logger.error(f"QR feature extraction error: {e}")
            return {
                'qr_code_count': 0,
                'qr_codes_detected': False,
                'aadhaar_decoded': False,
                'error': str(e)
            }
    
    def extract_forensics_features(self, image: np.ndarray, gray: np.ndarray) -> Dict[str, Any]:
        
        try:
            features = {}
            
            # Image quality metrics
            features['image_width'] = image.shape[1]
            features['image_height'] = image.shape[0]
            features['aspect_ratio'] = image.shape[1] / image.shape[0]
            
            # Sharpness (Laplacian variance)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            features['sharpness'] = laplacian.var()
            
            # Brightness and contrast
            features['brightness'] = np.mean(gray)
            features['contrast'] = np.std(gray)
            
            # Noise analysis
            noise = gray.astype(np.float32) - cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
            features['noise_level'] = np.std(noise)
            
            # Edge density
            edges = cv2.Canny(gray, 50, 150)
            features['edge_density'] = np.sum(edges > 0) / edges.size
            
            # Gradient analysis for resampling detection
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(gx**2 + gy**2)
            features['gradient_mean'] = np.mean(gradient_magnitude)
            features['gradient_std'] = np.std(gradient_magnitude)
            
            # Copy-paste detection using template matching
            features['copy_paste_score'] = self.detect_copy_paste(gray)
            
            # Frequency domain analysis
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            features['freq_domain_energy'] = np.sum(magnitude_spectrum ** 2)
            
            return features
            
        except Exception as e:
            logger.error(f"Forensics feature extraction error: {e}")
            return {
                'image_width': 0, 'image_height': 0, 'aspect_ratio': 0,
                'sharpness': 0, 'brightness': 0, 'contrast': 0,
                'noise_level': 0, 'edge_density': 0,
                'gradient_mean': 0, 'gradient_std': 0,
                'copy_paste_score': 0, 'freq_domain_energy': 0
            }
    
    def extract_face_features(self, image: np.ndarray) -> Dict[str, Any]:
        
        try:
            features = {}
            
            if FACE_RECOGNITION_AVAILABLE:
                
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_image)
                features['face_count'] = len(face_locations)
                features['face_detected'] = len(face_locations) > 0
                
                if face_locations:
                    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
                    if face_encodings:
                        features['face_encoding_quality'] = np.mean(np.abs(face_encodings[0]))
                    else:
                        features['face_encoding_quality'] = 0
                else:
                    features['face_encoding_quality'] = 0
                
            else:
                
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                if self.face_cascade is not None:
                    faces = self.face_cascade.detectMultiScale(
                        gray, 
                        scaleFactor=1.1,
                        minNeighbors=4,
                        minSize=(30, 30),
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )
                    
                    features['face_count'] = len(faces)
                    features['face_detected'] = len(faces) > 0
                    features['face_encoding_quality'] = 0.5 if len(faces) > 0 else 0
                else:
                    features['face_count'] = 0
                    features['face_detected'] = False
                    features['face_encoding_quality'] = 0
            
            
            if features['face_detected']:
                face_qualities = []
                if FACE_RECOGNITION_AVAILABLE and 'face_locations' in locals():
                    for (top, right, bottom, left) in face_locations:
                        face_width = right - left
                        face_height = bottom - top
                        face_area = face_width * face_height
                        quality = min(face_area / 10000.0, 1.0)
                        face_qualities.append(quality)
                else:
                    
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    if self.face_cascade is not None:
                        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                        for (x, y, w, h) in faces:
                            face_area = w * h
                            quality = min(face_area / 10000.0, 1.0)
                            face_qualities.append(quality)
                
                features['face_quality_mean'] = np.mean(face_qualities) if face_qualities else 0
                features['face_quality_std'] = np.std(face_qualities) if face_qualities else 0
            else:
                features['face_quality_mean'] = 0
                features['face_quality_std'] = 0
            
            
            features['multiple_faces_detected'] = features['face_count'] > 1
            
            return features
            
        except Exception as e:
            logger.error(f"Face feature extraction error: {e}")
            return {
                'face_count': 0,
                'face_detected': False,
                'face_quality_mean': 0,
                'face_quality_std': 0,
                'face_encoding_quality': 0,
                'multiple_faces_detected': False
            }
    
    def extract_logo_features(self, image: np.ndarray, gray: np.ndarray, document_type: str) -> Dict[str, Any]:
        
        try:
            features = {}
            
            
            template_scores = self.match_templates(gray, document_type)
            features.update(template_scores)
            
            
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            logo_candidates = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 500 < area < 10000:  # Logo size range
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter ** 2)
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h
                        
                        if 0.7 < aspect_ratio < 1.3 and circularity > 0.3:
                            logo_candidates.append({
                                'area': area,
                                'circularity': circularity,
                                'aspect_ratio': aspect_ratio,
                                'x': x, 'y': y, 'w': w, 'h': h
                            })
            
            features['logo_candidates_count'] = len(logo_candidates)
            features['logo_detected'] = len(logo_candidates) > 0
            
            if logo_candidates:
                
                img_h, img_w = gray.shape
                expected_logo_position = False
                for logo in logo_candidates:
                    x, y = logo['x'], logo['y']
                    if (x < img_w * 0.3 and y < img_h * 0.3) or (x > img_w * 0.7 and y < img_h * 0.3):
                        expected_logo_position = True
                        break
                
                features['logo_expected_position'] = expected_logo_position
            else:
                features['logo_expected_position'] = False
            
            return features
            
        except Exception as e:
            logger.error(f"Logo feature extraction error: {e}")
            return {
                'logo_candidates_count': 0,
                'logo_detected': False,
                'logo_expected_position': False,
                'template_match_score': 0
            }
    
    def match_templates(self, gray: np.ndarray, document_type: str) -> Dict[str, Any]:
        
        try:
            features = {}
            best_match_score = 0
            
            for template_name, template in self.template_database.items():
                template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                
                # Multi-scale template matching
                for scale in [0.8, 1.0, 1.2]:
                    h, w = template_gray.shape
                    scaled_template = cv2.resize(template_gray, (int(w * scale), int(h * scale)))
                    
                    if (scaled_template.shape[0] <= gray.shape[0] and 
                        scaled_template.shape[1] <= gray.shape[1]):
                        
                        result = cv2.matchTemplate(gray, scaled_template, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, _ = cv2.minMaxLoc(result)
                        
                        if max_val > best_match_score:
                            best_match_score = max_val
            
            features['template_match_score'] = best_match_score
            features['expected_template_found'] = best_match_score > 0.6
            
            return features
            
        except Exception as e:
            logger.error(f"Template matching error: {e}")
            return {'template_match_score': 0, 'expected_template_found': False}
    
    def extract_metadata_features(self, image_path: str) -> Dict[str, Any]:
        
        try:
            features = {}
            
            # File metadata
            file_stats = os.stat(image_path)
            features['file_size'] = file_stats.st_size
            features['file_creation_time'] = file_stats.st_ctime
            features['file_modification_time'] = file_stats.st_mtime
            
            # EXIF data analysis
            try:
                with open(image_path, 'rb') as f:
                    tags = exifread.process_file(f)
                
                features['camera_make'] = str(tags.get('Image Make', 'Unknown'))
                features['camera_model'] = str(tags.get('Image Model', 'Unknown'))
                features['software'] = str(tags.get('Image Software', 'Unknown'))
                
                # Suspicious software detection
                suspicious_software = ['photoshop', 'gimp', 'paint.net', 'canva', 'pixlr']
                software_lower = features['software'].lower()
                features['editing_software_detected'] = any(sw in software_lower for sw in suspicious_software)
                
            except Exception as e:
                logger.warning(f"EXIF extraction error: {e}")
                features.update({
                    'camera_make': 'Unknown',
                    'camera_model': 'Unknown',
                    'software': 'Unknown',
                    'editing_software_detected': False
                })
            
            # File type verification
            if MAGIC_AVAILABLE:
                try:
                    file_type = magic.from_file(image_path, mime=True)
                    features['file_type'] = file_type
                    features['file_type_mismatch'] = not file_type.startswith('image/')
                except Exception as e:
                    features['file_type'] = 'Unknown'
                    features['file_type_mismatch'] = False
            else:
                # Simple file type check based on extension
                file_ext = os.path.splitext(image_path)[1].lower()
                expected_exts = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
                features['file_type'] = f'image/{file_ext[1:]}' if file_ext in expected_exts else 'unknown'
                features['file_type_mismatch'] = file_ext not in expected_exts
            
            return features
            
        except Exception as e:
            logger.error(f"Metadata feature extraction error: {e}")
            return {
                'file_size': 0,
                'file_creation_time': 0,
                'file_modification_time': 0,
                'camera_make': 'Unknown',
                'camera_model': 'Unknown',
                'software': 'Unknown',
                'editing_software_detected': False,
                'file_type': 'Unknown',
                'file_type_mismatch': False
            }
    
    def extract_texture_features(self, gray: np.ndarray) -> Dict[str, Any]:
        
        try:
            features = {}
            
            # Local Binary Pattern (LBP)
            lbp = feature.local_binary_pattern(gray, 8, 1, method='uniform')
            features['lbp_mean'] = np.mean(lbp)
            features['lbp_std'] = np.std(lbp)
            
            # Gray Level Co-occurrence Matrix (GLCM) features
            glcm = feature.graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4])
            features['glcm_contrast'] = feature.graycoprops(glcm, 'contrast')[0, 0]
            features['glcm_homogeneity'] = feature.graycoprops(glcm, 'homogeneity')[0, 0]
            features['glcm_energy'] = feature.graycoprops(glcm, 'energy')[0, 0]
            features['glcm_correlation'] = feature.graycoprops(glcm, 'correlation')[0, 0]
            
            
            gabor_responses = []
            for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                try:
                    filtered = filters.gabor(gray, frequency=0.1, theta=theta)[0]
                    gabor_responses.append(np.mean(np.abs(filtered)))
                except:
                    gabor_responses.append(0)
            
            features['gabor_mean'] = np.mean(gabor_responses)
            features['gabor_std'] = np.std(gabor_responses)
            
            
            features['texture_uniformity'] = np.sum(np.histogram(gray, bins=256)[0] ** 2)
            
            return features
            
        except Exception as e:
            logger.error(f"Texture feature extraction error: {e}")
            return {
                'lbp_mean': 0, 'lbp_std': 0,
                'glcm_contrast': 0, 'glcm_homogeneity': 0,
                'glcm_energy': 0, 'glcm_correlation': 0,
                'gabor_mean': 0, 'gabor_std': 0,
                'texture_uniformity': 0
            }
    
    def extract_color_features(self, image: np.ndarray) -> Dict[str, Any]:
        
        try:
            features = {}
            
            # Color space analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # HSV statistics
            h, s, v = cv2.split(hsv)
            features['hue_mean'] = np.mean(h)
            features['hue_std'] = np.std(h)
            features['saturation_mean'] = np.mean(s)
            features['saturation_std'] = np.std(s)
            features['value_mean'] = np.mean(v)
            features['value_std'] = np.std(v)
            
            # Color distribution
            for i, color in enumerate(['blue', 'green', 'red']):
                channel = image[:, :, i]
                features[f'{color}_mean'] = np.mean(channel)
                features[f'{color}_std'] = np.std(channel)
            
            # Color histogram features
            hist_b = cv2.calcHist([image], [0], None, [32], [0, 256])
            hist_g = cv2.calcHist([image], [1], None, [32], [0, 256])
            hist_r = cv2.calcHist([image], [2], None, [32], [0, 256])
            
            features['color_hist_entropy'] = -np.sum(hist_b * np.log2(hist_b + 1e-10))
            features['color_uniformity'] = np.sum(hist_b ** 2) + np.sum(hist_g ** 2) + np.sum(hist_r ** 2)
            
            return features
            
        except Exception as e:
            logger.error(f"Color feature extraction error: {e}")
            return {
                'hue_mean': 0, 'hue_std': 0,
                'saturation_mean': 0, 'saturation_std': 0,
                'value_mean': 0, 'value_std': 0,
                'blue_mean': 0, 'blue_std': 0,
                'green_mean': 0, 'green_std': 0,
                'red_mean': 0, 'red_std': 0,
                'color_hist_entropy': 0, 'color_uniformity': 0
            }
    
    def detect_copy_paste(self, gray: np.ndarray, block_size: int = 32) -> float:
        try:
            h, w = gray.shape
            similarity_scores = []
            
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = gray[i:i+block_size, j:j+block_size]
                    
                    # Compare with other blocks
                    for ii in range(i + block_size, h - block_size, block_size):
                        for jj in range(0, w - block_size, block_size):
                            compare_block = gray[ii:ii+block_size, jj:jj+block_size]
                            
                            # Calculate normalized cross-correlation
                            correlation = cv2.matchTemplate(block, compare_block, cv2.TM_CCOEFF_NORMED)
                            similarity_scores.append(correlation[0, 0])
            
            # Return mean similarity score
            return np.mean(similarity_scores) if similarity_scores else 0.0
            
        except Exception as e:
            logger.error(f"Copy-paste detection error: {e}")
            return 0.0
    
    def classify_document(self, features: Dict[str, Any]) -> Dict[str, Any]:
        try:
            feature_vector = self.prepare_feature_vector(features)
            
            if feature_vector is None:
                return {
                    'is_authentic': False,
                    'confidence': 0.1,
                    'classification_method': 'feature_extraction_failed',
                    'anomaly_score': 1.0
                }
            
            feature_vector = feature_vector.reshape(1, -1)
            
            anomaly_score = self.models['isolation_forest'].decision_function(feature_vector)[0]
            is_anomaly = self.models['isolation_forest'].predict(feature_vector)[0] == -1
            
            
            try:
                svm_prediction = self.models['svm'].predict(feature_vector)[0]
                svm_confidence = np.max(self.models['svm'].predict_proba(feature_vector)[0])
            except:
                svm_prediction = 0
                svm_confidence = 0.5
            
            
            try:
                xgb_prediction = self.models['xgboost'].predict(feature_vector)[0]
                xgb_confidence = np.max(self.models['xgboost'].predict_proba(feature_vector)[0])
            except:
                xgb_prediction = 0
                xgb_confidence = 0.5
            
            # Random Forest classification
            try:
                rf_prediction = self.models['random_forest'].predict(feature_vector)[0]
                rf_confidence = np.max(self.models['random_forest'].predict_proba(feature_vector)[0])
            except:
                rf_prediction = 0
                rf_confidence = 0.5
            
            # Rule-based classification
            rule_based_score = self.rule_based_classification(features)
            
            # Ensemble decision
            ensemble_score = (
                svm_confidence * 0.25 +
                xgb_confidence * 0.25 +
                rf_confidence * 0.25 +
                rule_based_score * 0.2 +
                (1 - abs(anomaly_score)) * 0.05
            )
            
            # Apply penalties for suspicious features
            if features.get('suspicious_text_detected', False):
                ensemble_score *= 0.3
            if features.get('editing_software_detected', False):
                ensemble_score *= 0.5
            if features.get('multiple_faces_detected', False):
                ensemble_score *= 0.7
            if features.get('copy_paste_score', 0) > 0.8:
                ensemble_score *= 0.4
            
            # Final decision with lenient threshold
            is_authentic = ensemble_score >= 0.3 and not is_anomaly
            
            return {
                'is_authentic': is_authentic,
                'confidence': float(ensemble_score),
                'classification_method': 'ensemble_ml',
                'anomaly_score': float(abs(anomaly_score)),
                'svm_prediction': int(svm_prediction),
                'xgb_prediction': int(xgb_prediction),
                'rf_prediction': int(rf_prediction),
                'rule_based_score': float(rule_based_score),
                'is_anomaly': bool(is_anomaly),
                'detailed_scores': {
                    'svm_confidence': float(svm_confidence),
                    'xgb_confidence': float(xgb_confidence),
                    'rf_confidence': float(rf_confidence),
                    'rule_based_score': float(rule_based_score),
                    'anomaly_score': float(anomaly_score)
                }
            }
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return {
                'is_authentic': False,
                'confidence': 0.1,
                'classification_method': 'classification_failed',
                'error': str(e)
            }
    
    def rule_based_classification(self, features: Dict[str, Any]) -> float:
        
        try:
            score = 0.5  # Start with neutral score
            
            # Image quality checks
            if features.get('sharpness', 0) > 100:
                score += 0.1
            if features.get('contrast', 0) > 50:
                score += 0.1
            
            # OCR quality checks
            if features.get('ocr_confidence_mean', 0) > 80:
                score += 0.1
            if features.get('easyocr_regions_count', 0) > 5:
                score += 0.05
            
            # Face detection for ID documents
            if features.get('face_detected', False):
                score += 0.1
                if features.get('face_quality_mean', 0) > 0.7:
                    score += 0.05
            
            # QR code checks (important for Aadhaar)
            if features.get('qr_codes_detected', False):
                score += 0.1
                if features.get('aadhaar_qr_pattern', False):
                    score += 0.1
            
            # Logo detection
            if features.get('logo_detected', False):
                score += 0.05
                if features.get('logo_expected_position', False):
                    score += 0.05
            
            # Template matching
            if features.get('expected_template_found', False):
                score += 0.1
            
            # Metadata checks
            if not features.get('editing_software_detected', False):
                score += 0.1
            
            # Penalty for suspicious features
            if features.get('suspicious_text_detected', False):
                score -= 0.3
            if features.get('copy_paste_score', 0) > 0.7:
                score -= 0.2
            if features.get('multiple_faces_detected', False):
                score -= 0.1
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Rule-based classification error: {e}")
            return 0.3
    
    def prepare_feature_vector(self, features: Dict[str, Any]) -> Optional[np.ndarray]:
        
        try:
            # Select numerical features
            numerical_features = [
                'ocr_text_length', 'ocr_word_count', 'ocr_confidence_mean',
                'easyocr_regions_count', 'alpha_ratio', 'digit_ratio',
                'qr_code_count', 'qr_data_length', 'qr_quality_mean',
                'image_width', 'image_height', 'aspect_ratio',
                'sharpness', 'brightness', 'contrast', 'noise_level',
                'edge_density', 'gradient_mean', 'gradient_std',
                'face_count', 'face_quality_mean', 'face_encoding_quality',
                'logo_candidates_count', 'template_match_score',
                'file_size', 'hue_mean', 'saturation_mean', 'value_mean',
                'lbp_mean', 'glcm_contrast', 'glcm_homogeneity',
                'copy_paste_score', 'freq_domain_energy'
            ]
            
            
            boolean_features = [
                'suspicious_text_detected', 'qr_codes_detected',
                'aadhaar_qr_pattern', 'face_detected', 'multiple_faces_detected',
                'logo_detected', 'logo_expected_position',
                'editing_software_detected', 'file_type_mismatch',
                'expected_template_found'
            ]
            
            
            feature_vector = []
            for feature in numerical_features:
                value = features.get(feature, 0)
                if isinstance(value, (int, float)) and not np.isnan(value):
                    feature_vector.append(float(value))
                else:
                    feature_vector.append(0.0)
            
            # Extract boolean values
            for feature in boolean_features:
                value = features.get(feature, False)
                feature_vector.append(1.0 if value else 0.0)
            
            return np.array(feature_vector)
            
        except Exception as e:
            logger.error(f"Feature vector preparation error: {e}")
            return None
    
    def save_models(self):
        
        try:
            os.makedirs('models', exist_ok=True)
            
            for model_name, model in self.models.items():
                if model_name in ['svm', 'xgboost', 'random_forest']:
                    model_path = f'models/{model_name}_document_classifier.joblib'
                    joblib.dump(model, model_path)
                    logger.info(f"Saved {model_name} model to {model_path}")
                    
        except Exception as e:
            logger.error(f"Error saving models: {e}")

ml_verifier = None

def get_ml_verifier():
    global ml_verifier
    if ml_verifier is None:
        ml_verifier = AdvancedDocumentVerifier()
    return ml_verifier


def verify_document(image_path: str, document_type: str = "id-card") -> Dict[str, Any]:
    
    try:
        # Extract features
        verifier = get_ml_verifier()
        features = verifier.extract_comprehensive_features(image_path, document_type)
        
        if 'error' in features:
            return {
                'is_authentic': False,
                'confidence': 0.1,
                'analysis': features['error'],
                'features': features
            }
        
        # Classify document
        classification_result = verifier.classify_document(features)
        
        # Combine results
        result = {
            'is_authentic': classification_result['is_authentic'],
            'confidence': classification_result['confidence'],
            'analysis': f"Document verification completed using {classification_result['classification_method']}",
            'features': features,
            'classification_details': classification_result
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Document verification error: {e}")
        return {
            'is_authentic': False,
            'confidence': 0.05,
            'analysis': f"Verification failed: {str(e)}",
            'features': {},
            'error': str(e)
        }

if __name__ == "__main__":
    logger.info("Advanced Document Verifier initialized successfully")
    logger.info(f"PyTorch available: {TORCH_AVAILABLE}")
    logger.info(f"face_recognition available: {FACE_RECOGNITION_AVAILABLE}")
    logger.info(f"DeepFace available: {DEEPFACE_AVAILABLE}")
    logger.info(f"python-magic available: {MAGIC_AVAILABLE}")
