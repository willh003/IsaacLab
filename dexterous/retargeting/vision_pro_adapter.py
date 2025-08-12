import numpy as np
from typing import Optional, Tuple
from avp_stream import VisionProStreamer



class VisionProHandAdapter:
    """
    Adapter to convert Apple Vision Pro hand data directly to the format expected by dex-retargeting.
    Bypasses MediaPipe conversion and directly maps AVP joints to optimizer requirements.
    """
    
    def __init__(self, avp_ip: str, hand_type: str = "Right"):
        """
        Initialize the Vision Pro hand adapter.
        
        Args:
            avp_ip: IP address of the Apple Vision Pro device
            hand_type: Which hand to track ("Right" or "Left")
        """
        self.streamer = VisionProStreamer(ip=avp_ip, record=True)
        self.hand_type = hand_type.lower()
        self.first_middle_tip_vec = None

    def mediapipe_to_avp_idx(self):
        return {
            0:0,
            1:1,
            2:2,
            3:3,
            4:4,
            5:6,
            6:7,
            7:8,
            8:9,
            9:11,
            10:12,
            11:13,
            12:14,
            13:16,
            14:17,
            15:18,
            16:19,
            17:21,
            18:22,
            19:23,
            20:24,
            21:0        
        }

    def detect(self) -> Tuple[int, Optional[np.ndarray], None, None]:
        """
        Get the latest hand pose data from Vision Pro directly in AVP format.
        
        Returns:
            Tuple matching SingleHandDetector.detect() output:
            - num_hands: Number of detected hands (1 if successful, 0 if failed)
            - joint_pos: (21, 3) array of joint positions in AVP coordinates, or None if detection failed
            - keypoint_2d: None (not used with Vision Pro)
            - mediapipe_wrist_rot: None (not used with Vision Pro)
        """
        try:
            # Get latest data from Vision Pro
            data = self.streamer.latest
            
            if data is None:
                return 0, None, None, None
            

            finger_positions = data['right_fingers']

            # hard coded for vector retargeting
            joint_position_arr = np.zeros((22, 3))
            
            # Extract xyz positions from 4x4 transformation matrices
            # The translation is in the last column, first 3 elements
            for mp_idx, avp_idx in self.mediapipe_to_avp_idx().items():
                joint_position_arr[mp_idx] = finger_positions[avp_idx][:3, 3]
                
            #joint_position_arr = joint_position_arr[:, [0,2,1]] # swap y and z for sapien
            #joint_position_arr[: , 0] = -joint_position_arr[: , 0] # flip y for sapien

            # Calculate transformation to make middle finger point straight up [0,0,1]
            # Get the vector from wrist to middle finger tip
            wrist_pos = joint_position_arr[0]
            middle_tip_pos = joint_position_arr[12]
            middle_finger_vector = middle_tip_pos - wrist_pos

            if self.first_middle_tip_vec is None:
                self.first_middle_tip_vec = middle_finger_vector

            # Normalize the current middle finger vector
            current_magnitude = np.linalg.norm(self.first_middle_tip_vec)
            if current_magnitude > 1e-6:  # Avoid division by zero
                current_middle_normalized = self.first_middle_tip_vec / current_magnitude
                
                # Target vector is [0,0,1] (pointing straight up)
                target_vector = np.array([0.0, 0.0, 1.0])
                
                # Calculate rotation matrix using cross product method
                # This rotates from current_middle_normalized to target_vector
                cross_product = np.cross(current_middle_normalized, target_vector)
                dot_product = np.dot(current_middle_normalized, target_vector)
                
                if np.linalg.norm(cross_product) > 1e-6:  # If vectors are not parallel
                    # Rodrigues' rotation formula
                    K = np.array([[0, -cross_product[2], cross_product[1]],
                                  [cross_product[2], 0, -cross_product[0]],
                                  [-cross_product[1], cross_product[0], 0]])
                    
                    rotation_matrix = (np.eye(3) + 
                                     K + 
                                     K @ K * (1 - dot_product) / (1 - dot_product**2))
                else:
                    # Vectors are parallel, no rotation needed or 180 degree rotation
                    if dot_product < 0:
                        # 180 degree rotation around any perpendicular axis
                        rotation_matrix = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
                    else:
                        rotation_matrix = np.eye(3)
                
                # Apply the rotation to all fingertip positions relative to wrist
                for finger_idx in range(1, 21):  # thumb, index, middle, ring
                    # Get vector from wrist to fingertip
                    finger_vector = joint_position_arr[finger_idx] - wrist_pos
                    # Apply rotation
                    rotated_finger_vector = rotation_matrix @ finger_vector
                    # Update position
                    joint_position_arr[finger_idx] = wrist_pos + rotated_finger_vector


            return joint_position_arr
            
            
        except Exception as e:
            print(f"Error in Vision Pro detection: {e}")
            return None