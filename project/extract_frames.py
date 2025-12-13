import cv2
import os
import glob
import sys
from pathlib import Path

# 添加项目根目录到 sys.path 以支持绝对导入
# 使用 .resolve() 确保处理软链接和相对路径的准确性


def extract_frames(video_folder: str, output_folder: str, interval: int = 10) -> None:
    """
    从视频中提取帧并保存为图片。

    Args:
        video_folder (str): 视频文件夹路径
        output_folder (str): 输出图片文件夹路径
        interval (int): 每隔多少帧提取一张图片，默认为10
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取所有视频文件 (支持 .mp4, .avi, .mov)
    video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_folder, ext)))

    if not video_files:
        print(f"[警告] 在 {video_folder} 中未找到视频文件！")
        return

    img_counter = 0

    for video_path in video_files:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[错误] 无法打开视频文件: {video_path}")
            continue

        frame_count = 0

        print(f"[信息] 正在处理: {video_path}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 每隔 interval 帧保存一次
            if frame_count % interval == 0:
                # 生成文件名，例如 000001.jpg
                img_name = f"{img_counter:06d}.jpg"
                save_path = os.path.join(output_folder, img_name)
                cv2.imwrite(save_path, frame)
                print(f"[信息] 已保存: {img_name}")
                img_counter += 1

            frame_count += 1

        cap.release()

    print(f"[完成] 处理完成！共生成 {img_counter} 张图片。")


def main():
    """主函数"""
    # --- 配置 ---
    video_dir = (
        "D:/Microsoft VS Code/PYTHON/comsen-task-1/project/videos"  # 视频存放路径
    )
    output_dir = (
        "D:/Microsoft VS Code/PYTHON/comsen-task-1/project/raw_images"  # 图片输出路径
    )
    frame_interval = 30  # 每隔多少帧抽一张 (根据视频长短和需求调整)

    extract_frames(video_dir, output_dir, frame_interval)


if __name__ == "__main__":
    main()
