import cv2
import numpy as np
import openpose.pyopenpose as op
import torch
from fn import PoseKalmanFilter, calculate_angles_with_points, draw_angle, draw_single
from lib.model import PosePredictor
from uncertainty_handling import TemporalActionRecognizer, recognize_action, calculate_uncertainty


def initialize_openpose():
    params = {
        "model_folder": "openpose/models",
        "net_resolution": "-1x368",
        "model_pose": "COCO",
        "number_people_max": 1,
        "render_threshold": 0.05,
        "disable_blending": False,
        "tracking": 1,
        "scale_number": 1,
        "scale_gap": 0.25
    }
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    return opWrapper


def load_pose_model():
    pose_model = PosePredictor()
    pose_model.load_state_dict(torch.load('weights/pose_predictor_epoch30.pth'))
    pose_model.eval()
    return pose_model


def prepare_blank_images():
    input_img = np.zeros((640, 640, 3), dtype=np.uint8)
    pred_img = np.zeros((640, 640, 3), dtype=np.uint8)
    input_img = cv2.putText(input_img, "input_pose", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    pred_img = cv2.putText(pred_img, "pred_pose", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return input_img, pred_img


def process_frame(frame, opWrapper, pose_model, pose_kf, temporal_recognizer):
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    keypoints = datum.poseKeypoints

    frame_resized = cv2.resize(frame, (640, 640))
    frame_resized = cv2.putText(frame_resized, "source_pose", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    input_img, pred_img = prepare_blank_images()

    if keypoints is not None:
        keypoints = keypoints[:, [0, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10], :]
        keypoints[:, :, :2] /= [frame.shape[1], frame.shape[0]]
        input_data = keypoints.reshape(-1, 39)
        input_tensor = torch.tensor(input_data, dtype=torch.float32)

        with torch.no_grad():
            output = pose_model(input_tensor)

        input_nps = input_tensor.numpy().reshape(-1, 13, 3)
        output_nps = output.numpy().reshape(-1, 13, 2)

        for input_np, output_np in zip(input_nps, output_nps):
            pred_pose = np.column_stack((output_np * 640, np.ones(13)))
            pred_pose = pose_kf.update(pred_pose)
            input_pose = np.column_stack((input_np[:, :2] * 640, input_np[:, 2]))

            temporal_recognizer.update(pred_pose)
            action_recognition = recognize_action(pred_pose, temporal_recognizer)
            uncertainty = calculate_uncertainty(pred_pose, temporal_recognizer)

            input_img = draw_single(input_img, input_pose)
            pred_img = draw_single(pred_img, pred_pose)

            angles_with_points = calculate_angles_with_points(pred_pose)
            for key, data in angles_with_points.items():
                pred_img = draw_angle(pred_img, data["point"], data["angle"], data["side"])

            y_offset = 60
            for group, action_id in action_recognition.items():
                action_text = f"{group}: {action_id}"
                cv2.putText(pred_img, action_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 30

            cv2.putText(pred_img, f"uncertain: {uncertainty:.2f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)

    return np.hstack((frame_resized, input_img, pred_img))


def main():
    video_path = "video/deal2.mp4"
    opWrapper = initialize_openpose()
    pose_model = load_pose_model()
    pose_kf = PoseKalmanFilter(num_keypoints=13)
    temporal_recognizer = TemporalActionRecognizer()

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_with_action_numbers.mp4', fourcc, 20.0, (1920, 640))

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            combined_img = process_frame(frame, opWrapper, pose_model, pose_kf, temporal_recognizer)

            cv2.imshow("OpenPose with Action Recognition", combined_img)
            out.write(combined_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        out.release()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
