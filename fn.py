import collections
import re
import cv2
import numpy as np
import torch

string_classes = str
int_classes = int
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
ORANGE = (0, 165, 255)
PURPLE = (255, 0, 255)

COCO_PAIR = [(0, 13), (1, 2), (1, 3), (3, 5), (2, 4), (4, 6), (13, 14), (7, 8),
             (7, 9), (8, 10), (9, 11), (10, 12)]
POINT_COLORS = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),
                (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),
                (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),
                (0, 255, 255)]
LINE_COLORS = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50), (77, 255, 222),
               (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77), (77, 222, 255),
               (255, 156, 127), (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]

MPII_PAIR = [(8, 9), (11, 12), (11, 10), (2, 1), (1, 0), (13, 14), (14, 15), (3, 4), (4, 5),
             (8, 7), (7, 6), (6, 2), (6, 3), (8, 12), (8, 13)]

numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}

_use_shared_memory = True


def collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])

    if isinstance(batch[0], torch.Tensor):
        out = None
        if _use_shared_memory:
            numel = sum([x.numel() for x in batch])
            storage = batch[0].storage()._new_shared(numel)
            out = batch[0].new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [collate_fn(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))


def collate_fn_list(batch):
    img, inp, im_name = zip(*batch)
    img = collate_fn(img)
    im_name = collate_fn(im_name)

    return img, inp, im_name


def draw_single(frame, pts, joint_format='coco'):
    if joint_format == 'coco':
        l_pair = COCO_PAIR
        p_color = POINT_COLORS
        line_color = LINE_COLORS
    elif joint_format == 'mpii':
        l_pair = MPII_PAIR
        p_color = [PURPLE, BLUE, BLUE, RED, RED, BLUE, BLUE, RED, RED, PURPLE, PURPLE, PURPLE, RED, RED, BLUE, BLUE]
    else:
        raise NotImplementedError

    part_line = {}
    pts = np.concatenate((pts, np.expand_dims((pts[1, :] + pts[2, :]) / 2, 0)), axis=0)  # 追加颈部关键点
    pts = np.concatenate((pts, np.expand_dims((pts[7, :] + pts[8, :]) / 2, 0)), axis=0)  # 追加臀部关键点
    for n in range(pts.shape[0]):
        if pts[n, 2] <= 0.2:
            continue
        cor_x, cor_y = int(pts[n, 0]), int(pts[n, 1])
        part_line[n] = (cor_x, cor_y)
        cv2.circle(frame, (cor_x, cor_y), 3, p_color[n], -1)

    for i, (start_p, end_p) in enumerate(l_pair):
        if start_p in part_line and end_p in part_line:
            start_xy = part_line[start_p]
            end_xy = part_line[end_p]
            conf1 = pts[start_p, 2]
            conf2 = pts[end_p, 2]
            if conf1 < 0.2 or conf2 < 0.2:
                continue
            cv2.line(frame, start_xy, end_xy, line_color[i], int(1 * (pts[start_p, 2] + pts[end_p, 2]) + 1))
    return frame


def extrct_keypoints(pose, index: list):
    keypoint_list = []
    for key in index:
        point = np.array([int(pose[key, 0]), int(pose[key, 1])])
        keypoint_list.append(point)
    return keypoint_list


def calculate_angle(keypoint_list):
    left_upper_vector = keypoint_list[0] - keypoint_list[1]
    left_lower_vector = keypoint_list[2] - keypoint_list[1]
    right_upper_vector = keypoint_list[3] - keypoint_list[4]
    right_lower_vector = keypoint_list[5] - keypoint_list[4]

    left_angle = np.degrees(np.arccos(np.clip(np.dot(left_upper_vector, left_lower_vector) /
                                              (np.linalg.norm(left_upper_vector) * np.linalg.norm(left_lower_vector)), -1.0, 1.0)))
    right_angle = np.degrees(np.arccos(np.clip(np.dot(right_upper_vector, right_lower_vector) /
                                               (np.linalg.norm(right_upper_vector) * np.linalg.norm(right_lower_vector)), -1.0, 1.0)))

    left_point = keypoint_list[1]
    right_point = keypoint_list[4]

    return left_angle, right_angle, left_point, right_point


def draw_angle(frame, point, point_angle, flag="left", length=50):
    angle = 45
    start_point = point
    end_point = (
        int(start_point[0] + length * np.cos(np.radians(angle))),
        int(start_point[1] - length * np.sin(np.radians(angle)))
    )

    if flag == "right":
        end_point = (
            int(start_point[0] - length * np.cos(np.radians(angle))),
            int(start_point[1] - length * np.sin(np.radians(angle)))
        )

    cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

    if flag == "left":
        horizontal_line_end_point = (end_point[0] + length, end_point[1])
    else:
        horizontal_line_end_point = (end_point[0] - length, end_point[1])

    cv2.line(frame, end_point, horizontal_line_end_point, (0, 255, 0), 2)

    if flag == "left":
        text_position = (end_point[0], end_point[1] - 10)
    else:
        text_position = (horizontal_line_end_point[0], horizontal_line_end_point[1] - 10)

    cv2.putText(frame, '{0:.2f}'.format(point_angle), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(frame, point, 5, color=(0, 255, 0), thickness=-1)
    cv2.circle(frame, point, 10, color=(0, 255, 0), thickness=2)

    return frame


class KalmanFilter:
    def __init__(self, process_noise=5e-4, measurement_noise=5e-3, error_estimate=1.0):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.error_estimate = error_estimate
        self.estimate = np.array([0.0, 0.0], dtype=float)
        self.error_covariance = np.eye(2, dtype=float) * error_estimate

    def update(self, measurement):
        self.error_covariance += np.eye(2, dtype=float) * self.process_noise
        kalman_gain = self.error_covariance / (self.error_covariance + self.measurement_noise)
        self.estimate += kalman_gain @ (measurement - self.estimate)
        self.error_covariance = (np.eye(2, dtype=float) - kalman_gain) @ self.error_covariance

        return self.estimate


class PoseKalmanFilter:
    def __init__(self, num_keypoints=13):
        self.filters = [KalmanFilter() for _ in range(num_keypoints)]

    def update(self, keypoints):
        smoothed_keypoints = np.zeros_like(keypoints)
        for i, (kf, kp) in enumerate(zip(self.filters, keypoints)):
            smoothed_xy = kf.update(kp[:2])
            smoothed_keypoints[i, :2] = smoothed_xy
            smoothed_keypoints[i, 2] = kp[2]
        return smoothed_keypoints


def calculate_angles_with_points(keypoints):
    keypoints = np.concatenate((keypoints, np.expand_dims((keypoints[1, :] + keypoints[2, :]) / 2, 0)), axis=0)
    keypoints = np.concatenate((keypoints, np.expand_dims((keypoints[7, :] + keypoints[8, :]) / 2, 0)), axis=0)
    keypoints = np.around(keypoints).astype(int)

    NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST = range(7)
    LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE, NECK, MID_HIP = range(7, 15)

    neck = keypoints[NECK][:2]
    mid_hip = keypoints[MID_HIP][:2]
    left_shoulder = keypoints[LEFT_SHOULDER][:2]
    left_elbow = keypoints[LEFT_ELBOW][:2]
    left_hip = keypoints[LEFT_HIP][:2]
    right_shoulder = keypoints[RIGHT_SHOULDER][:2]
    right_elbow = keypoints[RIGHT_ELBOW][:2]
    right_hip = keypoints[RIGHT_HIP][:2]
    left_knee = keypoints[LEFT_KNEE][:2]
    right_knee = keypoints[RIGHT_KNEE][:2]
    left_ankle = keypoints[LEFT_ANKLE][:2]
    right_ankle = keypoints[RIGHT_ANKLE][:2]

    y_axis = np.array([0, -1])

    def calculate_vector(p1, p2):
        return p2 - p1

    def calculate_angle(v1, v2):
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        return np.degrees(angle)

    def determine_side(point1, point2):
        return "left" if point1[0] > point2[0] else "right"

    angles_with_points = {}

    vector_midhip_neck = calculate_vector(mid_hip, neck)
    angles_with_points['midhip_neck_y'] = {
        'angle': calculate_angle(vector_midhip_neck, y_axis),
        'point': mid_hip.tolist(),
        'side': determine_side(mid_hip, mid_hip)
    }

    vector_left_shoulder_elbow = calculate_vector(left_shoulder, left_elbow)
    vector_left_shoulder_hip = calculate_vector(left_shoulder, left_hip)
    angles_with_points['left_shoulder_elbow_hip'] = {
        'angle': calculate_angle(vector_left_shoulder_elbow, vector_left_shoulder_hip),
        'point': left_shoulder.tolist(),
        'side': determine_side(left_shoulder, right_shoulder)
    }

    vector_right_shoulder_elbow = calculate_vector(right_shoulder, right_elbow)
    vector_right_shoulder_hip = calculate_vector(right_shoulder, right_hip)
    angles_with_points['right_shoulder_elbow_hip'] = {
        'angle': calculate_angle(vector_right_shoulder_elbow, vector_right_shoulder_hip),
        'point': right_shoulder.tolist(),
        'side': determine_side(right_shoulder, left_shoulder)
    }

    vector_left_hip_shoulder = calculate_vector(left_hip, left_shoulder)
    vector_left_hip_knee = calculate_vector(left_hip, left_knee)
    angles_with_points['left_hip_shoulder_knee'] = {
        'angle': calculate_angle(vector_left_hip_shoulder, vector_left_hip_knee),
        'point': left_hip.tolist(),
        'side': determine_side(left_hip, right_hip)
    }

    vector_right_hip_shoulder = calculate_vector(right_hip, right_shoulder)
    vector_right_hip_knee = calculate_vector(right_hip, right_knee)
    angles_with_points['right_hip_shoulder_knee'] = {
        'angle': calculate_angle(vector_right_hip_shoulder, vector_right_hip_knee),
        'point': right_hip.tolist(),
        'side': determine_side(right_hip, left_hip)
    }

    vector_left_knee_hip = calculate_vector(left_knee, left_hip)
    vector_left_knee_ankle = calculate_vector(left_knee, left_ankle)
    angles_with_points['left_knee_hip_ankle'] = {
        'angle': calculate_angle(vector_left_knee_hip, vector_left_knee_ankle),
        'point': left_knee.tolist(),
        'side': determine_side(left_knee, right_knee)
    }

    vector_right_knee_hip = calculate_vector(right_knee, right_hip)
    vector_right_knee_ankle = calculate_vector(right_knee, right_ankle)
    angles_with_points['right_knee_hip_ankle'] = {
        'angle': calculate_angle(vector_right_knee_hip, vector_right_knee_ankle),
        'point': right_knee.tolist(),
        'side': determine_side(right_knee, left_knee)
    }

    return angles_with_points
