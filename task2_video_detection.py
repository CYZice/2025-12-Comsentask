from ultralytics import YOLO
import cv2
import numpy as np
import time


def task2_video_detection():
    print("=== 任务2：视频目标检测 ===")
    model_path = "./runs/train/extended/weights/best.pt"
    # 加载预训练模型
    model = YOLO(model_path)

    # 选择检测模式
    print("请选择检测模式:")
    print("1. 摄像头实时检测")
    print("2. 视频文件检测")
    print("3. 使用示例视频")

    choice = input("请输入选择 (1/2/3): ").strip()

    if choice == "1":
        # 摄像头检测
        print("启动摄像头实时检测...")
        detect_from_camera(model)
    elif choice == "2":
        # 视频文件检测
        video_path = input("请输入视频文件路径: ").strip()
        detect_from_video(model, video_path)
    elif choice == "3":
        # 使用网络视频
        print("使用示例视频进行检测...")
        video_url = "https://ultralytics.com/images/decelera_landscape_min.mov"
        detect_from_video(model, video_url)
    else:
        print("无效选择，使用默认摄像头检测")
        detect_from_camera(model)


def detect_from_camera(model, camera_id=0):
    """从摄像头进行实时检测"""
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"无法打开摄像头 {camera_id}")
        return

    # 获取摄像头参数
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"摄像头参数: {width}x{height}, {fps}fps")
    print("按 'q' 键退出检测")

    # 用于计算FPS
    prev_time = time.time()
    fps_counter = 0
    fps_display = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头画面")
            break

        # 进行检测
        results = model(frame, conf=0.5, iou=0.45)

        # 绘制检测结果
        annotated_frame = results[0].plot()

        # 计算并显示FPS
        current_time = time.time()
        fps_counter += 1
        if current_time - prev_time >= 1.0:
            fps_display = fps_counter
            fps_counter = 0
            prev_time = current_time

        # 在画面上显示FPS
        cv2.putText(
            annotated_frame,
            f"FPS: {fps_display}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        # 显示结果
        cv2.imshow("YOLOv8 Real-time Detection", annotated_frame)

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera detection completed")


def detect_from_video(model, video_path):
    """从视频文件进行检测"""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Cannot open video file: {video_path}")
        return

    # 获取视频参数
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video parameters: {width}x{height}, {fps}fps, {total_frames} frames")

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path = "task2_video_output.mp4"
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()

    print("Starting video processing...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 进行检测
        results = model(frame, conf=0.5, iou=0.45)

        # 绘制检测结果
        annotated_frame = results[0].plot()

        # 写入输出视频
        out.write(annotated_frame)

        frame_count += 1

        # 显示进度
        if frame_count % 30 == 0:
            elapsed_time = time.time() - start_time
            progress = (frame_count / total_frames) * 100
            fps_processing = frame_count / elapsed_time
            print(
                f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames) "
                f"Processing speed: {fps_processing:.1f}fps"
            )

    cap.release()
    out.release()

    total_time = time.time() - start_time
    print(f"\nVideo processing completed!")
    print(f"Total frames: {frame_count}")
    print(f"Processing time: {total_time:.1f} seconds")
    print(f"Average processing speed: {frame_count/total_time:.1f}fps")
    print(f"Output video saved as: {output_path}")


if __name__ == "__main__":
    task2_video_detection()
