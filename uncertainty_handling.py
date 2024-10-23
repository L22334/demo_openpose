import numpy as np
from scipy.stats import norm
from collections import deque
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class Constants:
    SPINE_ANGLE_MEAN: float = 170
    SPINE_ANGLE_STD: float = 10
    SPINE_BEND_ANGLE_MEAN: float = 120
    SPINE_BEND_ANGLE_STD: float = 20
    SHOULDER_ANGLE_DIFF_THRESHOLD: float = 30
    HIP_ANGLE_MEAN: float = 90
    HIP_ANGLE_STD: float = 20
    KNEE_STRAIGHT_THRESHOLD: float = 160
    KNEE_BEND_ANGLE_MEAN: float = 60
    KNEE_BEND_ANGLE_STD: float = 15

class EnhancedPoseFeatureExtractor:
    @staticmethod
    def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        ba, bc = a - b, c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    @staticmethod
    def extract_static_features(pose: np.ndarray) -> Dict[str, float]:
        NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST, LEFT_HIP, RIGHT_HIP = range(9)

        mid_shoulder = (pose[LEFT_SHOULDER] + pose[RIGHT_SHOULDER]) / 2
        mid_hip = (pose[LEFT_HIP] + pose[RIGHT_HIP]) / 2
        vertical_line = mid_hip + [0, -1, 0]

        angles = {
            'spine': EnhancedPoseFeatureExtractor.calculate_angle(mid_shoulder, mid_hip, vertical_line),
            'left_shoulder': EnhancedPoseFeatureExtractor.calculate_angle(pose[LEFT_ELBOW], pose[LEFT_SHOULDER], pose[LEFT_HIP]),
            'right_shoulder': EnhancedPoseFeatureExtractor.calculate_angle(pose[RIGHT_ELBOW], pose[RIGHT_SHOULDER], pose[RIGHT_HIP]),
            'left_elbow': EnhancedPoseFeatureExtractor.calculate_angle(pose[LEFT_SHOULDER], pose[LEFT_ELBOW], pose[LEFT_WRIST]),
            'right_elbow': EnhancedPoseFeatureExtractor.calculate_angle(pose[RIGHT_SHOULDER], pose[RIGHT_ELBOW], pose[RIGHT_WRIST]),
            'left_hip': EnhancedPoseFeatureExtractor.calculate_angle(pose[LEFT_SHOULDER], pose[LEFT_HIP], vertical_line),
            'right_hip': EnhancedPoseFeatureExtractor.calculate_angle(pose[RIGHT_SHOULDER], pose[RIGHT_HIP], vertical_line)
        }

        heights = {
            'left_wrist': pose[LEFT_WRIST, 1] - pose[LEFT_SHOULDER, 1],
            'right_wrist': pose[RIGHT_WRIST, 1] - pose[RIGHT_SHOULDER, 1],
        }

        symmetry = {joint: abs(angles[f'left_{joint}'] - angles[f'right_{joint}']) for joint in ['shoulder', 'elbow', 'hip']}

        distances = {
            'hand_to_hip_left': np.linalg.norm(pose[LEFT_WRIST] - pose[LEFT_HIP]),
            'hand_to_hip_right': np.linalg.norm(pose[RIGHT_WRIST] - pose[RIGHT_HIP]),
        }

        # 添加更多相关特征
        torso_angle = EnhancedPoseFeatureExtractor.calculate_angle(mid_shoulder, mid_hip, vertical_line)
        leg_symmetry = abs(angles['left_hip'] - angles['right_hip'])
        arm_height_diff = abs(heights['left_wrist'] - heights['right_wrist'])
        
        return {**angles, **heights, **symmetry, **distances, 
                'torso_angle': torso_angle, 
                'leg_symmetry': leg_symmetry,
                'arm_height_diff': arm_height_diff}

    @staticmethod
    def extract_dynamic_features(pose_history: List[np.ndarray]) -> Dict[str, float]:
        if len(pose_history) < 2:
            return {}

        velocities = np.diff(pose_history, axis=0)
        accelerations = np.diff(velocities, axis=0)

        avg_velocity = np.mean(np.abs(velocities), axis=0)
        avg_acceleration = np.mean(np.abs(accelerations), axis=0)

        joints = ['nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                  'left_wrist', 'right_wrist', 'left_hip', 'right_hip']

        dynamic_features = {f'{joint}_{feature}': float(np.linalg.norm(data[i]))
                            for i, joint in enumerate(joints)
                            for feature, data in zip(['velocity', 'acceleration'], [avg_velocity, avg_acceleration])}

        spine_angles = [EnhancedPoseFeatureExtractor.calculate_angle(
            (pose[1] + pose[2]) / 2,  # mid_shoulder
            (pose[7] + pose[8]) / 2,  # mid_hip
            (pose[7] + pose[8]) / 2 + [0, -1, 0]  # vertical line from mid_hip
        ) for pose in pose_history]

        angular_velocities = np.diff(spine_angles)
        angular_accelerations = np.diff(angular_velocities)

        dynamic_features.update({
            'spine_angular_velocity': float(np.mean(np.abs(angular_velocities))),
            'spine_angular_acceleration': float(np.mean(np.abs(angular_accelerations)))
        })

        # 添加更多动态特征
        pose_change_rate = np.mean(np.abs(np.diff(pose_history, axis=0)))
        
        dynamic_features['pose_change_rate'] = float(np.mean(pose_change_rate))
        
        return dynamic_features

class TemporalActionRecognizer:
    def __init__(self, window_size: int = 5):  # 减小窗口大小以提高响应速度
        self.window_size = window_size
        self.pose_history = deque(maxlen=window_size)
        self.action_history = deque(maxlen=window_size)

    def update(self, pose: np.ndarray):
        self.pose_history.append(pose)

    def extract_features(self) -> Dict[str, float]:
        if len(self.pose_history) < self.window_size:
            return {}

        static_features = EnhancedPoseFeatureExtractor.extract_static_features(self.pose_history[-1])
        dynamic_features = EnhancedPoseFeatureExtractor.extract_dynamic_features(list(self.pose_history))

        return {**static_features, **dynamic_features}

    def smooth_action(self, current_action: Dict[str, int]) -> Dict[str, int]:
        self.action_history.append(current_action)
        
        if len(self.action_history) < self.window_size:
            return current_action

        smoothed_action = {}
        for group in ['group1', 'group2', 'group3']:
            actions = [action[group] for action in self.action_history]
            # 使用加权投票,最近的动作权重更大
            weighted_actions = [(action, weight) for action, weight in zip(actions, range(1, len(actions)+1))]
            smoothed_action[group] = max(weighted_actions, key=lambda x: x[1])[0]

        return smoothed_action

class ActionClassifier:
    @staticmethod
    def classify_group1(features: Dict[str, float]) -> int:
        spine_angle = features['spine']
        torso_angle = features['torso_angle']
        pose_change_rate = features.get('pose_change_rate', 0)

        if spine_angle > 160:
            return 1  # 站直
        elif spine_angle < 120:
            if abs(torso_angle - 90) > 30 and pose_change_rate > 5:
                return 4  # 弯曲同时转身
            else:
                return 2  # 弯曲
        elif abs(torso_angle - 90) > 30:
            return 3  # 转身
        else:
            return 1  # 默认为站直

    @staticmethod
    def classify_group2(features: Dict[str, float]) -> int:
        left_wrist_height = features['left_wrist']
        right_wrist_height = features['right_wrist']
        left_wrist_velocity = features.get('left_wrist_velocity', 0)
        right_wrist_velocity = features.get('right_wrist_velocity', 0)

        conditions = [
            (left_wrist_height > 0 and right_wrist_height > 0) and (left_wrist_velocity < 5 and right_wrist_velocity < 5),
            ((left_wrist_height < 0 and right_wrist_height > 0) or (left_wrist_height > 0 and right_wrist_height < 0)) and
            (left_wrist_velocity > 5 or right_wrist_velocity > 5),
            (left_wrist_height < 0 and right_wrist_height < 0) and (left_wrist_velocity > 10 and right_wrist_velocity > 10)
        ]
        return np.argmax(conditions) + 1

    @staticmethod
    def classify_group3(features: Dict[str, float]) -> int:
        left_hip_angle = features['left_hip']
        right_hip_angle = features['right_hip']
        left_hip_velocity = features.get('left_hip_velocity', 0)
        right_hip_velocity = features.get('right_hip_velocity', 0)

        probs = np.array([
            norm(Constants.HIP_ANGLE_MEAN, Constants.HIP_ANGLE_STD).pdf(left_hip_angle) *
            norm(Constants.HIP_ANGLE_MEAN, Constants.HIP_ANGLE_STD).pdf(right_hip_angle),
            (left_hip_angle > Constants.KNEE_STRAIGHT_THRESHOLD and right_hip_angle > Constants.KNEE_STRAIGHT_THRESHOLD) and
            (left_hip_velocity < 5 and right_hip_velocity < 5),
            ((left_hip_angle > Constants.KNEE_STRAIGHT_THRESHOLD > right_hip_angle) or
             (left_hip_angle < Constants.KNEE_STRAIGHT_THRESHOLD < right_hip_angle)) and
            (left_hip_velocity > 5 or right_hip_velocity > 5),
            (left_hip_angle < Constants.KNEE_STRAIGHT_THRESHOLD and right_hip_angle < Constants.KNEE_STRAIGHT_THRESHOLD) and
            (left_hip_velocity < 5 and right_hip_velocity < 5),
            ((left_hip_angle < Constants.KNEE_STRAIGHT_THRESHOLD < right_hip_angle) or
             (left_hip_angle > Constants.KNEE_STRAIGHT_THRESHOLD > right_hip_angle)) and
            (left_hip_velocity > 10 or right_hip_velocity > 10),
            norm(Constants.KNEE_BEND_ANGLE_MEAN, Constants.KNEE_BEND_ANGLE_STD).pdf(left_hip_angle) *
            norm(Constants.KNEE_BEND_ANGLE_MEAN, Constants.KNEE_BEND_ANGLE_STD).pdf(right_hip_angle) and
            (left_hip_velocity > 15 and right_hip_velocity > 15)
        ])
        return np.argmax(probs) + 1

def recognize_action(pose: np.ndarray, temporal_recognizer: TemporalActionRecognizer) -> Dict[str, int]:
    temporal_recognizer.update(pose)
    features = temporal_recognizer.extract_features()

    if not features:
        return {}

    return {
        'group1': ActionClassifier.classify_group1(features),
        'group2': ActionClassifier.classify_group2(features),
        'group3': ActionClassifier.classify_group3(features)
    }

def calculate_uncertainty(pose: np.ndarray, temporal_recognizer: TemporalActionRecognizer) -> float:
    features = temporal_recognizer.extract_features()
    if not features:
        return 1.0

    velocities = [features.get(f'{joint}_velocity', 0) for joint in
                  ['spine', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']]
    pose_stability = np.mean(velocities)

    static_features = [features['spine'], features['left_shoulder'], features['right_shoulder'],
                       features['left_hip'], features['right_hip']]
    feature_consistency = np.std(static_features)

    # 调整不确定性计算公式
    uncertainty = 1 / (1 + np.exp(-(pose_stability * 0.5 + feature_consistency * 0.5 - 1)))
    return uncertainty
