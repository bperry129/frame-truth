"""
Main Dataset Building Pipeline for FrameTruth Training
Orchestrates the complete process: download -> extract frames -> compute features
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List
import time

from config import DATA_DIR
from download_videos import download_from_csv, generate_video_id
from frame_extractor import extract_frames_batch
from feature_extractor import extract_features_batch, save_features_to_csv

def build_dataset_from_csv(csv_path: str) -> Dict:
    """
    Complete pipeline: CSV -> Downloaded Videos -> Frames -> Features -> Dataset
    
    Args:
        csv_path: Path to CSV with columns: url,label
    
    Returns:
        Summary of the complete process
    """
    print("🚀 STARTING FRAMETRUTH TRAINING DATASET PIPELINE")
    print("=" * 60)
    
    start_time = time.time()
    
    # Step 1: Download videos
    print("\n📥 STEP 1: DOWNLOADING VIDEOS")
    print("-" * 40)
    
    download_results = download_from_csv(csv_path)
    
    if "error" in download_results:
        return {"error": f"Download failed: {download_results['error']}"}
    
    successful_downloads = download_results["successful"]
    
    if not successful_downloads:
        return {"error": "No videos were successfully downloaded"}
    
    print(f"✅ Downloaded {len(successful_downloads)} videos successfully")
    
    # Step 2: Extract frames
    print("\n🎬 STEP 2: EXTRACTING FRAMES")
    print("-" * 40)
    
    # Prepare video list for frame extraction
    video_list = []
    for download in successful_downloads:
        video_list.append({
            "video_id": download["video_id"],
            "filepath": download["filepath"],
            "label": download["label"],
            "url": download["url"]
        })
    
    # Also include skipped videos (already downloaded)
    for skipped in download_results["skipped"]:
        video_list.append({
            "video_id": skipped["video_id"],
            "filepath": skipped["existing_file"],
            "label": skipped["label"],
            "url": skipped["url"]
        })
    
    frame_results = extract_frames_batch(video_list)
    
    successful_frames = frame_results["successful"]
    
    if not successful_frames:
        return {"error": "No frames were successfully extracted"}
    
    print(f"✅ Extracted frames from {len(successful_frames)} videos")
    
    # Step 3: Extract features
    print("\n🔬 STEP 3: EXTRACTING FEATURES")
    print("-" * 40)
    
    # Prepare video list for feature extraction
    feature_video_list = []
    for frame_result in successful_frames:
        # Find corresponding video info
        video_info = next((v for v in video_list if v["video_id"] == frame_result["video_id"]), None)
        if video_info:
            feature_video_list.append({
                "video_id": frame_result["video_id"],
                "filepath": video_info["filepath"],
                "label": video_info["label"],
                "url": video_info["url"]
            })
    
    # Also include skipped frame extractions (already extracted)
    for skipped in frame_results["skipped"]:
        video_info = next((v for v in video_list if v["video_id"] == skipped["video_id"]), None)
        if video_info:
            feature_video_list.append({
                "video_id": skipped["video_id"],
                "filepath": video_info["filepath"],
                "label": video_info["label"],
                "url": video_info["url"]
            })
    
    features_list = extract_features_batch(feature_video_list)
    
    if not features_list:
        return {"error": "No features were successfully extracted"}
    
    print(f"✅ Extracted features from {len(features_list)} videos")
    
    # Step 4: Save dataset
    print("\n💾 STEP 4: SAVING DATASET")
    print("-" * 40)
    
    # Save features to CSV
    dataset_path = DATA_DIR / "dataset.csv"
    save_features_to_csv(features_list, str(dataset_path))
    
    # Save summary metadata
    summary = {
        "total_videos_requested": len(pd.read_csv(csv_path)),
        "videos_downloaded": len(successful_downloads),
        "videos_skipped_download": len(download_results["skipped"]),
        "videos_failed_download": len(download_results["failed"]),
        "frames_extracted": len(successful_frames),
        "frames_skipped": len(frame_results["skipped"]),
        "frames_failed": len(frame_results["failed"]),
        "features_extracted": len(features_list),
        "dataset_path": str(dataset_path),
        "processing_time_seconds": time.time() - start_time,
        "feature_columns": len(features_list[0]) if features_list else 0
    }
    
    summary_path = DATA_DIR / "dataset_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print final summary
    print("\n🎯 PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"📊 Total processing time: {summary['processing_time_seconds']:.1f} seconds")
    print(f"📹 Videos processed: {summary['features_extracted']}/{summary['total_videos_requested']}")
    print(f"🔬 Features per video: {summary['feature_columns']}")
    print(f"💾 Dataset saved to: {dataset_path}")
    print(f"📋 Summary saved to: {summary_path}")
    
    # Show sample of the dataset
    if features_list:
        print(f"\n📊 SAMPLE FEATURES:")
        sample_features = features_list[0]
        feature_names = [k for k in sample_features.keys() if k not in ['video_id', 'video_path', 'label']]
        for i, feature_name in enumerate(feature_names[:10]):
            print(f"  {feature_name}: {sample_features[feature_name]}")
        if len(feature_names) > 10:
            print(f"  ... and {len(feature_names) - 10} more features")
    
    return summary

def create_test_csv(output_path: str = "test_videos.csv"):
    """
    Create a test CSV with 3 sample URLs for testing
    You can modify these URLs or create your own CSV
    """
    test_data = [
        {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "label": "real"},  # Rick Roll
        {"url": "https://www.youtube.com/watch?v=9bZkp7q19f0", "label": "real"},  # Gangnam Style
        {"url": "https://www.youtube.com/watch?v=kJQP7kiw5Fk", "label": "real"},  # Despacito
    ]
    
    df = pd.DataFrame(test_data)
    df.to_csv(output_path, index=False)
    
    print(f"📝 Created test CSV with {len(test_data)} videos: {output_path}")
    print("🔧 Modify this file with your own URLs and labels before running the pipeline")
    
    return output_path

def main():
    """
    Main function - run the complete training pipeline
    """
    import sys
    
    print("🎯 FrameTruth Training Dataset Builder")
    print("=" * 50)
    
    # Check for command line argument
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        if not Path(csv_file).exists():
            print(f"❌ CSV file not found: {csv_file}")
            return
    else:
        # Check if test CSV exists
        test_csv = "test_videos.csv"
        
        if not Path(test_csv).exists():
            print(f"📝 Creating test CSV file: {test_csv}")
            create_test_csv(test_csv)
            print(f"\n⚠️ Please edit {test_csv} with your video URLs and labels")
            print(f"   Format: url,label (where label is 'real' or 'ai')")
            print(f"   Then run this script again to process the videos")
            return
        
        csv_file = test_csv
    
    # Run the complete pipeline
    print(f"📊 Processing videos from: {csv_file}")
    
    try:
        result = build_dataset_from_csv(csv_file)
        
        if "error" in result:
            print(f"❌ Pipeline failed: {result['error']}")
        else:
            print(f"\n🎉 SUCCESS! Dataset built successfully")
            print(f"📁 Check the 'frametruth_training/data/' directory for results")
            
            # Show next steps
            print(f"\n🚀 NEXT STEPS:")
            print(f"1. Review the dataset.csv file")
            print(f"2. Add more videos to your CSV and run again")
            print(f"3. When you have 100+ videos, run train_model.py")
            
    except Exception as e:
        print(f"❌ Pipeline crashed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
