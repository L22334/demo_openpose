import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import cv2
import numpy as np
import openpose.pyopenpose as op
import torch
from fn import PoseKalmanFilter, calculate_angles_with_points, draw_angle, draw_single
from lib.model import PosePredictor
from uncertainty_handling import TemporalActionRecognizer, recognize_action, calculate_uncertainty, calculate_action_confidence


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

        action_descriptions = {
            'group1': {
                1: "站直",
                2: "弯曲",
                3: "转身",
                4: "弯曲同时转身"
            },
            'group2': {
                1: "双臂低于肩膀",
                2: "一个手臂高于肩膀",
                3: "双臂高于肩膀"
            },
            'group3': {
                1: "坐姿",
                2: "双腿站直",
                3: "单腿站直",
                4: "双腿弯曲",
                5: "单腿弯曲站立",
                6: "跪姿或蹲姿"
            }
        }

        # 使用支持中文的字体
        font_path = "C:/Windows/Fonts/simhei.ttf"  
        try:
            from PIL import ImageFont, ImageDraw, Image
            font = ImageFont.truetype(font_path, 20)
            
            def cv2AddChineseText(img, text, position, color):
                img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                draw.text(position, text, font=font, fill=color)
                return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                
            # 修改文本绘制部分
            y_offset = 60
            for group, action_id in action_recognition.items():
                action_desc = action_descriptions[group][action_id]
                confidence = calculate_action_confidence(
                    temporal_recognizer.extract_features(), 
                    int(group[-1]),
                    action_id
                )
                
                color = (
                    0,  # B
                    int(255 * confidence),  # G
                    int(255 * (1 - confidence))  # R
                )
                
                action_text = f"{group}: {action_desc} ({confidence:.2f})"
                pred_img = cv2AddChineseText(pred_img, action_text, (10, y_offset), color[::-1])  # 注意颜色需要反转RGB->BGR
                y_offset += 30
                
        except Exception as e:
            print(f"Error drawing Chinese text: {e}")

        uncertainty_color = (
            0,  # B
            int(255 * (1 - uncertainty)),  # G
            int(255 * uncertainty)  # R
        )
        cv2.putText(pred_img, f"Uncertainty: {uncertainty:.2f}", 
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, uncertainty_color, 1)

        legend_y = y_offset + 30
        cv2.putText(pred_img, "Keypoints Legend:", (10, legend_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        legend_y += 20
        cv2.line(pred_img, (10, legend_y), (30, legend_y), 
                 (0, 255, 0), 2)  # 绿色线代表骨骼
        cv2.putText(pred_img, "Skeleton", (35, legend_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        legend_y += 20
        cv2.circle(pred_img, (20, legend_y), 5, (0, 255, 0), -1)
        cv2.putText(pred_img, "Angle Point", (35, legend_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return np.hstack((frame_resized, input_img, pred_img))


def display_progress(combined_img, current_frame, total_frames):
    progress = current_frame / total_frames
    progress_bar_width = combined_img.shape[1] - 20
    progress_x = int(progress * progress_bar_width)
    
    # 使用深色背景使进度条更清晰
    cv2.rectangle(combined_img, 
                 (10, combined_img.shape[0] - 35),
                 (combined_img.shape[1] - 10, combined_img.shape[0] - 15),
                 (0, 0, 0), -1)  # 黑色背景
    
    # 进度条
    cv2.rectangle(combined_img, 
                 (10, combined_img.shape[0] - 30),
                 (progress_x, combined_img.shape[0] - 20),
                 (0, 255, 0), -1)

    # 进度文本
    cv2.putText(combined_img, 
               f"Processing: {int(progress * 100)}%",
               (10, combined_img.shape[0] - 40),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main():
    video_path = "video/deal.mp4"
    opWrapper = initialize_openpose()
    pose_model = load_pose_model()
    pose_kf = PoseKalmanFilter(num_keypoints=13)
    temporal_recognizer = TemporalActionRecognizer()

    cap = cv2.VideoCapture(video_path)
    
    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            combined_img = process_frame(frame, opWrapper, pose_model, 
                                      pose_kf, temporal_recognizer)

            # 添加进度条
            display_progress(combined_img, current_frame, total_frames)

            cv2.imshow("OpenPose with Action Recognition", combined_img)
            current_frame += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # 空格键暂停
                cv2.waitKey(0)

    except Exception as e:
        print(f"Error during video processing: {e}")
        
    finally:
        print("Cleaning up...")
        cap.release()
        cv2.destroyAllWindows()
        print("Done!")

if __name__ == "__main__":
    main()
