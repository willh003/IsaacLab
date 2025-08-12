import pickle
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np


class DetectionDataLoader:
    """
    Loader for pre-computed hand detection data.
    Provides an interface similar to SingleHandDetector.detect() but uses saved data.
    """
    
    def __init__(self, data_dir: str ="/home/will/dex-retargeting/example/vector_retargeting/data/detections"):
        """
        Initialize the loader with a directory containing detection data.
        
        Args:
            data_dir: Path to directory containing detection data files
        """
        self.data_dir = Path(data_dir)
        self._load_metadata()
        self._load_summary()
        
    def _load_metadata(self):
        """Load video metadata."""
        metadata_path = self.data_dir / "video_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                self.video_metadata = json.load(f)
        else:
            self.video_metadata = {}
            
    def _load_summary(self):
        """Load detection summary."""
        summary_path = self.data_dir / "detection_summary.pkl"
        if summary_path.exists():
            with open(summary_path, "rb") as f:
                self.summary = pickle.load(f)
        else:
            self.summary = {}
    
    def get_frame_data(self, frame_id: int) -> Optional[Dict]:
        """
        Get detection data for a specific frame.
        
        Args:
            frame_id: Frame number to retrieve
            
        Returns:
            Dictionary containing detection data or None if frame not found
        """
        frame_filename = f"frame_{frame_id:06d}.pkl"
        frame_path = self.data_dir / frame_filename
        
        if not frame_path.exists():
            return None
            
        with open(frame_path, "rb") as f:
            return pickle.load(f)
    
    def get_all_frames(self) -> List[Dict]:
        """
        Get all frame data as a list.
        
        Returns:
            List of all frame detection data
        """
        if "frame_data" in self.summary:
            return self.summary["frame_data"]
        return []
    
    def get_detection_rate(self) -> float:
        """
        Get the overall detection rate.
        
        Returns:
            Detection rate as a float between 0 and 1
        """
        return self.summary.get("detection_rate", 0.0)
    
    def get_total_frames(self) -> int:
        """
        Get total number of frames.
        
        Returns:
            Total frame count
        """
        return self.summary.get("total_frames", 0)
    
    def get_detected_frames(self) -> int:
        """
        Get number of frames with successful hand detection.
        
        Returns:
            Number of detected frames
        """
        return self.summary.get("detected_frames", 0)
    
    def get_video_metadata(self) -> Dict:
        """
        Get video metadata.
        
        Returns:
            Dictionary containing video information
        """
        return self.video_metadata
    
    def simulate_detector_interface(self, frame_id: int) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Simulate the SingleHandDetector.detect() interface for compatibility.
        
        Args:
            frame_id: Frame number to retrieve
            
        Returns:
            Tuple of (num_box, joint_pos, keypoint_2d, mediapipe_wrist_rot)
            matching the detector.detect() return format
        """
        frame_data = self.get_frame_data(frame_id)
        if frame_data is None:
            return 0, None, None, None
            
        # Convert back to numpy arrays if they exist
        joint_pos = np.array(frame_data["joint_pos"]) if frame_data["joint_pos"] is not None else None
        mediapipe_wrist_rot = np.array(frame_data["mediapipe_wrist_rot"]) if frame_data["mediapipe_wrist_rot"] is not None else None
        
        # keypoint_2d is already in pure Python format, no conversion needed
        keypoint_2d = frame_data["keypoint_2d"]
        
        return (
            frame_data["num_box"],
            joint_pos,
            keypoint_2d,
            mediapipe_wrist_rot
        )
    
    def get_keypoint_2d_as_numpy(self, frame_id: int, img_size = None) -> Optional[np.ndarray]:
        """
        Get 2D keypoints as a numpy array, optionally scaled to image dimensions.
        
        Args:
            frame_id: Frame number to retrieve
            img_size: Optional (height, width) tuple to scale coordinates
            
        Returns:
            Numpy array of shape (21, 2) with 2D keypoint coordinates, or None if not found
        """
        frame_data = self.get_frame_data(frame_id)
        if frame_data is None or frame_data["keypoint_2d"] is None:
            return None
            
        keypoint_2d_data = frame_data["keypoint_2d"]
        landmarks = keypoint_2d_data["landmarks"]
        
        # Extract x, y coordinates
        keypoints = np.array([[landmark["x"], landmark["y"]] for landmark in landmarks])
        
        # Scale to image dimensions if provided
        if img_size is not None:
            keypoints = keypoints * np.array([img_size[1], img_size[0]])[None, :]  # width, height
            
        return keypoints
    
    def get_keypoint_3d_as_numpy(self, frame_id: int) -> Optional[np.ndarray]:
        """
        Get 3D keypoints as a numpy array.
        
        Args:
            frame_id: Frame number to retrieve
            
        Returns:
            Numpy array of shape (21, 3) with 3D keypoint coordinates, or None if not found
        """
        frame_data = self.get_frame_data(frame_id)
        if frame_data is None or frame_data["keypoint_2d"] is None:
            return None
            
        keypoint_2d_data = frame_data["keypoint_2d"]
        landmarks = keypoint_2d_data["landmarks"]
        
        # Extract x, y, z coordinates
        keypoints = np.array([[landmark["x"], landmark["y"], landmark["z"]] for landmark in landmarks])
        
        return keypoints
    
    def get_detection_timeline(self) -> List[bool]:
        """
        Get a timeline of detection success for each frame.
        
        Returns:
            List of booleans indicating detection success for each frame
        """
        if "frame_data" in self.summary:
            return [frame["detection_success"] for frame in self.summary["frame_data"]]
        return []
    
    def filter_detected_frames(self) -> List[int]:
        """
        Get list of frame IDs where hand detection was successful.
        
        Returns:
            List of frame IDs with successful detection
        """
        detected_frames = []
        for frame_data in self.get_all_frames():
            if frame_data["detection_success"]:
                detected_frames.append(frame_data["frame_id"])
        return detected_frames


def main():
    """Example usage of the DetectionDataLoader."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load and inspect saved detection data")
    parser.add_argument("--frame", type=int, help="Specific frame to inspect")
    parser.add_argument("--summary", action="store_true", help="Show summary statistics")
    
    args = parser.parse_args()
    
    # Load data
    loader = DetectionDataLoader()
    
    if args.summary:
        print("=== Detection Summary ===")
        print(f"Total frames: {loader.get_total_frames()}")
        print(f"Detected frames: {loader.get_detected_frames()}")
        print(f"Detection rate: {loader.get_detection_rate():.2%}")
        
        video_meta = loader.get_video_metadata()
        if video_meta:
            print(f"\nVideo metadata:")
            for key, value in video_meta.items():
                print(f"  {key}: {value}")
    
    if args.frame is not None:
        frame_data = loader.get_frame_data(args.frame)
        if frame_data:
            print(f"\n=== Frame {args.frame} ===")
            print(f"Detection success: {frame_data['detection_success']}")
            print(f"Num boxes: {frame_data['num_box']}")
            if frame_data['joint_pos'] is not None:
                print(f"Joint positions shape: {np.array(frame_data['joint_pos']).shape}")
        else:
            print(f"Frame {args.frame} not found")


if __name__ == "__main__":
    main() 