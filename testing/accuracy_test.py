#!/usr/bin/env python3
"""
Frame Truth Accuracy Testing Framework
=====================================

This script provides comprehensive accuracy testing for the Frame Truth AI detection system.
It tests against known ground truth datasets and provides detailed accuracy metrics.
"""

import os
import sys
import json
import requests
import time
import csv
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import statistics

# Add backend to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

class AccuracyTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
        self.test_start_time = datetime.now()
        
    def test_video(self, video_path: str, ground_truth: bool, video_info: Dict) -> Dict:
        """
        Test a single video and return detailed results
        
        Args:
            video_path: Path to video file
            ground_truth: True if AI-generated, False if real
            video_info: Metadata about the video (source, model, etc.)
        
        Returns:
            Dictionary with test results
        """
        print(f"\n🧪 Testing: {video_info.get('name', os.path.basename(video_path))}")
        print(f"   Ground Truth: {'AI' if ground_truth else 'Real'}")
        print(f"   Source: {video_info.get('source', 'Unknown')}")
        
        start_time = time.time()
        
        try:
            # Upload video file
            with open(video_path, 'rb') as f:
                upload_response = requests.post(
                    f"{self.base_url}/api/upload",
                    files={"file": f}
                )
            
            if upload_response.status_code != 200:
                raise Exception(f"Upload failed: {upload_response.text}")
            
            upload_data = upload_response.json()
            filename = upload_data['filename']
            
            print(f"   ✅ Upload successful: {filename}")
            
            # Analyze video
            analyze_response = requests.post(
                f"{self.base_url}/api/analyze",
                json={
                    "filename": filename,
                    "original_url": video_info.get('original_url', '')
                }
            )
            
            if analyze_response.status_code != 200:
                raise Exception(f"Analysis failed: {analyze_response.text}")
            
            analysis_data = analyze_response.json()
            result = analysis_data['result']
            
            analysis_time = time.time() - start_time
            
            # Extract key metrics
            predicted_ai = result.get('isAi', False)
            confidence = result.get('confidence', 0)
            curvature_score = result.get('curvatureScore', 0)
            distance_score = result.get('distanceScore', 0)
            model_detected = result.get('modelDetected', 'Unknown')
            reasoning = result.get('reasoning', [])
            
            # Determine correctness
            correct = (predicted_ai == ground_truth)
            
            # Classification type
            if ground_truth and predicted_ai:
                classification = "True Positive"
            elif not ground_truth and not predicted_ai:
                classification = "True Negative"
            elif ground_truth and not predicted_ai:
                classification = "False Negative"
            else:  # not ground_truth and predicted_ai
                classification = "False Positive"
            
            print(f"   🔍 Prediction: {'AI' if predicted_ai else 'Real'} (confidence: {confidence}%)")
            print(f"   📊 Result: {classification} ({'✅ Correct' if correct else '❌ Incorrect'})")
            print(f"   ⏱️  Analysis time: {analysis_time:.1f}s")
            
            test_result = {
                'timestamp': datetime.now().isoformat(),
                'video_info': video_info,
                'ground_truth': ground_truth,
                'ground_truth_label': 'AI' if ground_truth else 'Real',
                'predicted_ai': predicted_ai,
                'predicted_label': 'AI' if predicted_ai else 'Real',
                'confidence': confidence,
                'curvature_score': curvature_score,
                'distance_score': distance_score,
                'model_detected': model_detected,
                'reasoning': reasoning,
                'correct': correct,
                'classification': classification,
                'analysis_time': analysis_time,
                'filename': filename
            }
            
            return test_result
            
        except Exception as e:
            print(f"   ❌ Test failed: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'video_info': video_info,
                'ground_truth': ground_truth,
                'error': str(e),
                'analysis_time': time.time() - start_time
            }
    
    def run_test_suite(self, test_dataset_path: str) -> Dict:
        """
        Run complete test suite from dataset configuration
        
        Args:
            test_dataset_path: Path to JSON file with test dataset configuration
        
        Returns:
            Dictionary with comprehensive results
        """
        print(f"🚀 Starting Frame Truth Accuracy Test Suite")
        print(f"📅 Test started: {self.test_start_time}")
        print(f"📁 Dataset: {test_dataset_path}")
        
        # Load test dataset
        with open(test_dataset_path, 'r') as f:
            dataset = json.load(f)
        
        total_videos = len(dataset['ai_videos']) + len(dataset['real_videos'])
        print(f"📊 Total videos to test: {total_videos}")
        print(f"   - AI videos: {len(dataset['ai_videos'])}")
        print(f"   - Real videos: {len(dataset['real_videos'])}")
        
        # Test AI videos
        print(f"\n🤖 Testing AI-generated videos...")
        for video_info in dataset['ai_videos']:
            if os.path.exists(video_info['path']):
                result = self.test_video(video_info['path'], True, video_info)
