#!/usr/bin/env python3
"""
YOLO26-based 6D Grasp Pose Generation System
Flow: YOLO26 detection -> 2D-3D mapping -> Point cloud cropping -> 
      Denoising & Downsampling -> PCA -> 6D pose generation -> TF transform -> Robot execution
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
import time

# Optional ROS imports - system works without ROS in simulation mode
try:
    import rospy
    from geometry_msgs.msg import PoseStamped, Pose
    from sensor_msgs.msg import PointCloud2
    import sensor_msgs.point_cloud2 as pc2
    from tf.transformations import quaternion_from_euler, translation_matrix, rotation_matrix
    from tf import TransformListener
    ROS_AVAILABLE = True
    
    # Type alias for type hints when ROS is available
    ROSEnableTime = rospy.Time
    ROSDuration = rospy.Duration
    
except ImportError:
    ROS_AVAILABLE = False
    print("ROS not available, running in simulation mode")
    
    # Mock classes for simulation
    class Pose:
        def __init__(self):
            self.position = type('obj', (object,), {'x': 0, 'y': 0, 'z': 0})()
            self.orientation = type('obj', (object,), {'x': 0, 'y': 0, 'z': 0, 'w': 1})()
        
        def __repr__(self):
            return f"Pose(position=({self.position.x:.3f}, {self.position.y:.3f}, {self.position.z:.3f}), "\
                   f"orientation=({self.orientation.x:.3f}, {self.orientation.y:.3f}, {self.orientation.z:.3f}, {self.orientation.w:.3f}))"
    
    class PoseStamped:
        def __init__(self):
            self.header = type('obj', (object,), {'stamp': None, 'frame_id': ''})()
            self.pose = Pose()
    
    def quaternion_from_euler(a, b, c):
        return [0, 0, 0, 1]
    
    class MockTime:
        @staticmethod
        def now():
            return None
    
    class MockDuration:
        def __init__(self, secs):
            self.secs = secs
    
    ROSEnableTime = type(None)
    ROSDuration = MockDuration
    rospy = type('obj', (object,), {'Time': MockTime, 'Duration': MockDuration, 'init_node': lambda *args, **kwargs: None})()
    TransformListener = None

# Try to import open3d
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Open3D not available, some features will be limited")
    
    # Mock open3d classes for simulation
    class MockVector3dVector:
        def __init__(self, data):
            self.data = np.asarray(data)
    
    class MockPointCloud:
        def __init__(self):
            self._points = np.zeros((0, 3))
            self._colors = np.zeros((0, 3))
        
        @property
        def points(self):
            return self._points
        
        @points.setter
        def points(self, vec):
            if hasattr(vec, 'data'):
                self._points = np.asarray(vec.data)
            else:
                self._points = np.asarray(vec)
        
        @property
        def colors(self):
            return self._colors
        
        @colors.setter
        def colors(self, vec):
            if hasattr(vec, 'data'):
                self._colors = np.asarray(vec.data)
            else:
                self._colors = np.asarray(vec)
        
        def __len__(self):
            return len(self._points)
        
        def remove_statistical_outlier(self, nb_neighbors, std_ratio):
            # Simple mock: keep all points
            indices = list(range(len(self._points)))
            return self, indices
        
        def voxel_down_sample(self, voxel_size):
            # Simple mock: return self
            return self
        
        def select_by_index(self, indices):
            # Return new point cloud with selected indices
            new_pc = MockPointCloud()
            new_pc._points = self._points[indices]
            if len(self._colors) > 0:
                new_pc._colors = self._colors[indices]
            return new_pc
    
    class o3d:
        class geometry:
            PointCloud = MockPointCloud
        
        class utility:
            Vector3dVector = MockVector3dVector


class YOLO26Detector:
    """YOLO26 object detector for 2D bounding box detection"""
    
    def __init__(self, model_path: str = "yolo26_weights.pt", confidence_threshold: float = 0.5):
        """
        Initialize YOLO26 detector
        
        Args:
            model_path: Path to YOLO26 model weights
            confidence_threshold: Detection confidence threshold
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = self._load_model()
        
    def _load_model(self):
        """Load YOLO26 model"""
        # Placeholder for actual YOLO26 model loading
        # In practice, use ultralytics or custom YOLO26 implementation
        print(f"Loading YOLO26 model from {self.model_path}")
        return None
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect objects in image and return 2D bounding boxes
        
        Args:
            image: Input image (H, W, 3)
            
        Returns:
            List of detections with keys: 'bbox', 'class_id', 'confidence'
        """
        if self.model is None:
            # Simulated detection for testing
            h, w = image.shape[:2]
            return [{
                'bbox': [w//4, h//4, w//2, h//2],  # [x_min, y_min, x_max, y_max]
                'class_id': 0,
                'confidence': 0.95
            }]
        
        # Actual detection logic would go here
        results = self.model(image)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.conf[0] > self.confidence_threshold:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'class_id': int(box.cls[0]),
                        'confidence': float(box.conf[0])
                    })
        
        return detections


class PointCloudProcessor:
    """Process point cloud data for grasp pose estimation"""
    
    def __init__(self, voxel_size: float = 0.005, outlier_removal_std: float = 2.0):
        """
        Initialize point cloud processor
        
        Args:
            voxel_size: Voxel size for downsampling
            outlier_removal_std: Standard deviation threshold for outlier removal
        """
        self.voxel_size = voxel_size
        self.outlier_removal_std = outlier_removal_std
    
    def project_bbox_to_3d(self, bbox_2d: List[int], depth_image: np.ndarray, 
                           camera_intrinsics: np.ndarray) -> np.ndarray:
        """
        Project 2D bounding box to 3D point cloud region
        
        Args:
            bbox_2d: 2D bounding box [x_min, y_min, x_max, y_max]
            depth_image: Depth image aligned with RGB
            camera_intrinsics: Camera intrinsic matrix (3x3)
            
        Returns:
            Mask for point cloud cropping
        """
        x_min, y_min, x_max, y_max = bbox_2d
        height, width = depth_image.shape
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        
        # Create mask for bounding box region
        mask = ((x_coords >= x_min) & (x_coords <= x_max) & 
                (y_coords >= y_min) & (y_coords <= y_max))
        
        return mask
    
    def crop_point_cloud(self, point_cloud: o3d.geometry.PointCloud, 
                         mask: np.ndarray) -> o3d.geometry.PointCloud:
        """
        Crop point cloud based on 2D projection mask
        
        Args:
            point_cloud: Full point cloud
            mask: Boolean mask for cropping
            
        Returns:
            Cropped point cloud
        """
        # Flatten mask to match point cloud points
        mask_flat = mask.flatten()
        
        # Filter points based on mask - handle both open3d and mock point clouds
        if hasattr(point_cloud, '_points'):
            # Mock point cloud
            points = point_cloud._points
            colors = point_cloud._colors
        else:
            # Real open3d point cloud
            points = np.asarray(point_cloud.points)
            colors = np.asarray(point_cloud.colors) if len(point_cloud.colors) > 0 else np.array([])
        
        # Ensure mask length matches points
        if len(mask_flat) != len(points):
            # Handle case where mask doesn't match point cloud size
            # This might happen if depth image resolution differs
            print(f"Warning: Mask size {len(mask_flat)} != Point cloud size {len(points)}")
            # Take center region as fallback
            center_idx = len(points) // 2
            crop_size = min(1000, len(points) // 4)
            start_idx = max(0, center_idx - crop_size // 2)
            end_idx = min(len(points), center_idx + crop_size // 2)
            cropped_points = points[start_idx:end_idx]
            cropped_colors = colors[start_idx:end_idx] if len(colors) > 0 else np.zeros_like(cropped_points)
        else:
            cropped_points = points[mask_flat]
            cropped_colors = colors[mask_flat] if len(colors) > 0 else np.zeros_like(cropped_points)
        
        # Create new point cloud
        cropped_pc = o3d.geometry.PointCloud()
        cropped_pc.points = o3d.utility.Vector3dVector(cropped_points)
        if len(cropped_colors) > 0:
            cropped_pc.colors = o3d.utility.Vector3dVector(cropped_colors)
        
        return cropped_pc
    
    def denoise_and_downsample(self, point_cloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        """
        Remove outliers and downsample point cloud
        
        Args:
            point_cloud: Input point cloud
            
        Returns:
            Processed point cloud
        """
        # Remove statistical outliers
        if len(point_cloud.points) > 10:
            cl, ind = point_cloud.remove_statistical_outlier(
                nb_neighbors=20,
                std_ratio=self.outlier_removal_std
            )
            point_cloud = point_cloud.select_by_index(ind)
        
        # Voxel downsampling
        if len(point_cloud.points) > 0:
            point_cloud = point_cloud.voxel_down_sample(self.voxel_size)
        
        return point_cloud
    
    def compute_pca_pose(self, point_cloud: o3d.geometry.PointCloud) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute centroid and surface normals using PCA
        
        Args:
            point_cloud: Processed point cloud
            
        Returns:
            centroid: Point cloud centroid (3,)
            orientation: Orientation matrix from PCA (3x3)
        """
        if len(point_cloud.points) < 3:
            raise ValueError("Point cloud must have at least 3 points for PCA")
        
        points = np.asarray(point_cloud.points)
        
        # Compute centroid
        centroid = np.mean(points, axis=0)
        
        # Center points
        centered_points = points - centroid
        
        # Compute covariance matrix
        cov_matrix = np.cov(centered_points.T)
        
        # Eigen decomposition for PCA
        eigen_values, eigen_vectors = np.linalg.eigh(cov_matrix)
        
        # Sort eigenvectors by eigenvalues (descending)
        sorted_indices = np.argsort(eigen_values)[::-1]
        orientation = eigen_vectors[:, sorted_indices]
        
        # Ensure right-handed coordinate system
        if np.linalg.det(orientation) < 0:
            orientation[:, -1] *= -1
        
        return centroid, orientation


class GraspPoseGenerator:
    """Generate 6D grasp poses from point cloud features"""
    
    def __init__(self, approach_distance: float = 0.1, grasp_depth: float = 0.05):
        """
        Initialize grasp pose generator
        
        Args:
            approach_distance: Distance to approach before grasping
            grasp_depth: How deep to grasp into the object
        """
        self.approach_distance = approach_distance
        self.grasp_depth = grasp_depth
    
    def generate_grasp_pose(self, centroid: np.ndarray, 
                           orientation: np.ndarray) -> Pose:
        """
        Generate 6D grasp pose in camera frame
        
        Args:
            centroid: Object centroid
            orientation: Object orientation from PCA
            
        Returns:
            geometry_msgs/Pose: Grasp pose
        """
        # Define grasp approach direction (typically along negative Z of object frame)
        # Adjust based on your gripper configuration
        approach_vector = -orientation[:, 2]  # Assuming Z is the approach direction
        
        # Calculate grasp position
        grasp_position = centroid + approach_vector * self.approach_distance
        
        # Create rotation matrix for grasp pose
        # Align gripper with object orientation
        rotation_matrix = self._compute_grasp_rotation(orientation)
        
        # Convert rotation matrix to quaternion
        quaternion = self._rotation_matrix_to_quaternion(rotation_matrix)
        
        # Create ROS Pose message
        pose = Pose()
        pose.position.x = float(grasp_position[0])
        pose.position.y = float(grasp_position[1])
        pose.position.z = float(grasp_position[2])
        pose.orientation.x = float(quaternion[0])
        pose.orientation.y = float(quaternion[1])
        pose.orientation.z = float(quaternion[2])
        pose.orientation.w = float(quaternion[3])
        
        return pose
    
    def _compute_grasp_rotation(self, object_orientation: np.ndarray) -> np.ndarray:
        """
        Compute grasp rotation matrix based on object orientation
        
        Args:
            object_orientation: Object orientation from PCA (3x3)
            
        Returns:
            Rotation matrix for grasp pose
        """
        # Standard gripper alignment (adjust based on your gripper)
        # Typically: X = approach direction, Y = parallel to gripper fingers, Z = up
        gripper_alignment = np.array([
            [0, 0, 1],   # X axis of gripper aligns with Z of object
            [0, 1, 0],   # Y axis of gripper aligns with Y of object  
            [-1, 0, 0]   # Z axis of gripper aligns with -X of object
        ])
        
        # Combine object orientation with gripper alignment
        grasp_rotation = object_orientation @ gripper_alignment.T
        
        return grasp_rotation
    
    def _rotation_matrix_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to quaternion
        
        Args:
            R: Rotation matrix (3x3)
            
        Returns:
            Quaternion [x, y, z, w]
        """
        return quaternion_from_euler(0, 0, 0)  # Placeholder, use proper conversion


class RoboticArmController:
    """Control robotic arm for grasp execution"""
    
    def __init__(self, robot_base_frame: str = "base_link", 
                 camera_frame: str = "camera_link"):
        """
        Initialize robotic arm controller
        
        Args:
            robot_base_frame: Robot base coordinate frame
            camera_frame: Camera coordinate frame
        """
        self.robot_base_frame = robot_base_frame
        self.camera_frame = camera_frame
        self.tf_listener = None
        self.pose_pub = None
        self._init_ros()
    
    def _init_ros(self):
        """Initialize ROS components"""
        if not ROS_AVAILABLE:
            print("ROS not available, running in simulation mode")
            return
        
        try:
            rospy.init_node('grasp_pose_generator', anonymous=True)
            self.tf_listener = TransformListener()
            self.pose_pub = rospy.Publisher('/grasp_pose', PoseStamped, queue_size=10)
            print("ROS initialized successfully")
        except Exception as e:
            print(f"ROS initialization failed: {e}")
            print("Running in simulation mode")
    
    def transform_pose_to_robot_frame(self, camera_pose: Pose, 
                                      timestamp=None) -> Pose:
        """
        Transform pose from camera frame to robot base frame
        
        Args:
            camera_pose: Pose in camera frame
            timestamp: Time for TF lookup
            
        Returns:
            Pose in robot base frame
        """
        if self.tf_listener is None:
            # Return same pose if TF not available (simulation)
            return camera_pose
        
        if timestamp is None:
            timestamp = rospy.Time.now()
        
        # Wait for transform to be available
        try:
            self.tf_listener.waitForTransform(
                self.robot_base_frame, 
                self.camera_frame, 
                timestamp, 
                rospy.Duration(1.0)
            )
            
            # Create PoseStamped for transformation
            camera_pose_stamped = PoseStamped()
            camera_pose_stamped.header.stamp = timestamp
            camera_pose_stamped.header.frame_id = self.camera_frame
            camera_pose_stamped.pose = camera_pose
            
            # Transform to robot base frame
            robot_pose_stamped = self.tf_listener.transformPose(
                self.robot_base_frame, 
                camera_pose_stamped
            )
            
            return robot_pose_stamped.pose
            
        except Exception as e:
            print(f"TF transformation failed: {e}")
            return camera_pose
    
    def send_grasp_command(self, grasp_pose: Pose):
        """
        Send grasp command to robotic arm
        
        Args:
            grasp_pose: Grasp pose in robot base frame
        """
        # Publish grasp pose
        if self.pose_pub is not None and ROS_AVAILABLE:
            pose_msg = PoseStamped()
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = self.robot_base_frame
            pose_msg.pose = grasp_pose
            
            self.pose_pub.publish(pose_msg)
            print(f"Grasp pose published: position={grasp_pose.position}, orientation={grasp_pose.orientation}")
        else:
            # Simulation mode - just print the pose
            print(f"[Simulation] Grasp pose generated:")
            print(f"  Position: ({grasp_pose.position.x:.3f}, {grasp_pose.position.y:.3f}, {grasp_pose.position.z:.3f})")
            print(f"  Orientation: ({grasp_pose.orientation.x:.3f}, {grasp_pose.orientation.y:.3f}, "
                  f"{grasp_pose.orientation.z:.3f}, {grasp_pose.orientation.w:.3f})")
        
        # In a real system, you would call motion planning services here
        # Example: MoveIt! integration
        # self.move_group.set_pose_target(grasp_pose)
        # self.move_group.go(wait=True)


class GraspSystem:
    """Main system integrating all components"""
    
    def __init__(self, config: Dict):
        """
        Initialize complete grasp system
        
        Args:
            config: System configuration dictionary
        """
        self.config = config
        
        # Initialize components
        self.detector = YOLO26Detector(
            model_path=config.get('model_path', 'yolo26_weights.pt'),
            confidence_threshold=config.get('confidence_threshold', 0.5)
        )
        
        self.pc_processor = PointCloudProcessor(
            voxel_size=config.get('voxel_size', 0.005),
            outlier_removal_std=config.get('outlier_std', 2.0)
        )
        
        self.pose_generator = GraspPoseGenerator(
            approach_distance=config.get('approach_distance', 0.1),
            grasp_depth=config.get('grasp_depth', 0.05)
        )
        
        self.robot_controller = RoboticArmController(
            robot_base_frame=config.get('robot_base_frame', 'base_link'),
            camera_frame=config.get('camera_frame', 'camera_link')
        )
        
        # Camera intrinsics (should be calibrated)
        self.camera_intrinsics = np.array(config.get('camera_intrinsics', [
            [500, 0, 320],
            [0, 500, 240],
            [0, 0, 1]
        ]))
    
    def process_frame(self, rgb_image: np.ndarray, depth_image: np.ndarray, 
                     point_cloud: o3d.geometry.PointCloud) -> Optional[Pose]:
        """
        Process a single frame to generate grasp pose
        
        Args:
            rgb_image: RGB image
            depth_image: Depth image
            point_cloud: 3D point cloud
            
        Returns:
            Grasp pose in robot base frame, or None if failed
        """
        try:
            # Step 1: Detect objects with YOLO26
            print("Step 1: Running YOLO26 detection...")
            detections = self.detector.detect(rgb_image)
            
            if not detections:
                print("No objects detected")
                return None
            
            # Use the highest confidence detection
            best_detection = max(detections, key=lambda x: x['confidence'])
            bbox_2d = best_detection['bbox']
            print(f"Detected object with confidence {best_detection['confidence']:.2f}, bbox: {bbox_2d}")
            
            # Step 2: Project 2D bbox to 3D and crop point cloud
            print("Step 2: Projecting 2D bbox to 3D and cropping point cloud...")
            mask = self.pc_processor.project_bbox_to_3d(bbox_2d, depth_image, self.camera_intrinsics)
            cropped_pc = self.pc_processor.crop_point_cloud(point_cloud, mask)
            
            if len(cropped_pc.points) == 0:
                print("Cropped point cloud is empty")
                return None
            
            print(f"Cropped point cloud has {len(cropped_pc.points)} points")
            
            # Step 3: Denoise and downsample
            print("Step 3: Denoising and downsampling point cloud...")
            processed_pc = self.pc_processor.denoise_and_downsample(cropped_pc)
            
            if len(processed_pc.points) < 10:
                print("Processed point cloud has too few points")
                return None
            
            print(f"Processed point cloud has {len(processed_pc.points)} points")
            
            # Step 4: Compute PCA for centroid and orientation
            print("Step 4: Computing PCA for pose estimation...")
            centroid, orientation = self.pc_processor.compute_pca_pose(processed_pc)
            print(f"Centroid: {centroid}")
            print(f"Orientation matrix:\n{orientation}")
            
            # Step 5: Generate 6D grasp pose in camera frame
            print("Step 5: Generating 6D grasp pose...")
            camera_grasp_pose = self.pose_generator.generate_grasp_pose(centroid, orientation)
            
            # Step 6: Transform to robot base frame
            print("Step 6: Transforming pose to robot base frame...")
            robot_grasp_pose = self.robot_controller.transform_pose_to_robot_frame(camera_grasp_pose)
            
            # Step 7: Send to robotic arm
            print("Step 7: Sending grasp command to robot...")
            self.robot_controller.send_grasp_command(robot_grasp_pose)
            
            print("Grasp pose generation completed successfully!")
            return robot_grasp_pose
            
        except Exception as e:
            print(f"Error in grasp pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_continuous(self, rgb_callback, depth_callback, pc_callback):
        """
        Run system continuously with callbacks
        
        Args:
            rgb_callback: Function to get RGB images
            depth_callback: Function to get depth images  
            pc_callback: Function to get point clouds
        """
        print("Starting continuous grasp system...")
        
        while True:
            try:
                rgb_image = rgb_callback()
                depth_image = depth_callback()
                point_cloud = pc_callback()
                
                grasp_pose = self.process_frame(rgb_image, depth_image, point_cloud)
                
                if grasp_pose:
                    print(f"Generated grasp pose: {grasp_pose}")
                
                time.sleep(0.1)  # Adjust rate as needed
                
            except KeyboardInterrupt:
                print("Stopping grasp system...")
                break
            except Exception as e:
                print(f"Error in continuous loop: {e}")
                time.sleep(1.0)


def main():
    """Main entry point"""
    # Configuration
    config = {
        'model_path': 'yolo26_weights.pt',
        'confidence_threshold': 0.5,
        'voxel_size': 0.005,
        'outlier_std': 2.0,
        'approach_distance': 0.1,
        'grasp_depth': 0.05,
        'robot_base_frame': 'base_link',
        'camera_frame': 'camera_link',
        'camera_intrinsics': [
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0]
        ]
    }
    
    # Initialize system
    system = GraspSystem(config)
    
    # Example usage with dummy data
    print("Initializing grasp system...")
    
    # Create sample data for testing
    rgb_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    depth_image = np.random.uniform(0.5, 2.0, (480, 640)).astype(np.float32)
    
    # Create sample point cloud
    points = np.random.randn(10000, 3) * 0.1 + np.array([0, 0, 1.0])
    sample_pc = o3d.geometry.PointCloud()
    sample_pc.points = o3d.utility.Vector3dVector(points)
    
    # Process single frame
    grasp_pose = system.process_frame(rgb_image, depth_image, sample_pc)
    
    if grasp_pose:
        print(f"\nSuccess! Generated grasp pose:")
        print(f"Position: ({grasp_pose.position.x:.3f}, {grasp_pose.position.y:.3f}, {grasp_pose.position.z:.3f})")
        print(f"Orientation: ({grasp_pose.orientation.x:.3f}, {grasp_pose.orientation.y:.3f}, "
              f"{grasp_pose.orientation.z:.3f}, {grasp_pose.orientation.w:.3f})")
    else:
        print("\nFailed to generate grasp pose")


if __name__ == "__main__":
    main()
