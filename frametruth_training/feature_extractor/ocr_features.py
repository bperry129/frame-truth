"""
OCR Text Stability Features
Text analysis for AI detection
"""

import cv2
import numpy as np
from typing import Dict, List

def compute_ocr_features(frames: List[np.ndarray]) -> Dict:
    """Compute OCR text stability features from video frames"""
    try:
        # Try to import EasyOCR
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False)
        
        # Sample a few frames for OCR analysis
        sample_frames = frames[::max(1, len(frames)//4)][:4]
        
        all_texts = []
        for frame in sample_frames:
            try:
                results = reader.readtext(frame, detail=1)
                frame_texts = []
                for (bbox, text, confidence) in results:
                    if confidence > 0.4 and len(text.strip()) > 1:
                        frame_texts.append(text.strip())
                all_texts.append(frame_texts)
            except:
                all_texts.append([])
        
        # Calculate features
        all_detected_strings = [text for frame_texts in all_texts for text in frame_texts]
        
        if not all_detected_strings:
            return {
                "ocr_char_error_rate": 0.0,
                "ocr_frame_stability": 1.0,
                "ocr_mutation_rate": 0.0,
                "ocr_unique_string_count": 0,
                "ocr_total_detections": 0
            }
        
        # Character error rate
        char_error_count = 0
        total_chars = 0
        for text in all_detected_strings:
            for char in text:
                total_chars += 1
                if not char.isprintable() or ord(char) > 127:
                    char_error_count += 1
        
        ocr_char_error_rate = float(char_error_count / max(total_chars, 1))
        
        # Frame stability (simplified)
        unique_strings = list(set(all_detected_strings))
        ocr_frame_stability = 1.0 - (len(unique_strings) / max(len(all_detected_strings), 1))
        
        # Mutation rate (simplified)
        ocr_mutation_rate = min(0.5, len(unique_strings) / max(len(sample_frames), 1))
        
        return {
            "ocr_char_error_rate": ocr_char_error_rate,
            "ocr_frame_stability": float(ocr_frame_stability),
            "ocr_mutation_rate": float(ocr_mutation_rate),
            "ocr_unique_string_count": len(unique_strings),
            "ocr_total_detections": len(all_detected_strings)
        }
        
    except ImportError:
        print("⚠️ EasyOCR not available, using placeholder OCR features")
        return {
            "ocr_char_error_rate": 0.0,
            "ocr_frame_stability": 1.0,
            "ocr_mutation_rate": 0.0,
            "ocr_unique_string_count": 0,
            "ocr_total_detections": 0
        }
    except Exception as e:
        print(f"⚠️ OCR computation failed: {e}")
        return {
            "ocr_char_error_rate": 0.0,
            "ocr_frame_stability": 1.0,
            "ocr_mutation_rate": 0.0,
            "ocr_unique_string_count": 0,
            "ocr_total_detections": 0
        }
