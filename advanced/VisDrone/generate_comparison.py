import os
import random
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
import sys

# 设置中文字体，防止乱码
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 用来正常显示中文标签
plt.rcParams["axes.unicode_minus"] = False  # 用来正常显示负号


def generate_comparison():
    # 路径设置
    base_dir = Path(__file__).resolve().parent
    test_images_dir = base_dir / "small_datasets/test/images"
    output_dir = base_dir / "runs/detect/comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 模型路径
    model_baseline_path = base_dir / "runs/train/baseline2/weights/best.pt"
    model_best_path = base_dir / "best.pt"

    # 检查模型是否存在
    if not model_baseline_path.exists():
        print(f"Error: Baseline model not found at {model_baseline_path}")
        # 尝试下载 yolov8n.pt 如果不存在
        print("Attempting to download yolov8n.pt...")
        try:
            YOLO("yolov8n.pt")  # 这会自动下载
        except Exception as e:
            print(f"Failed to download yolov8n.pt: {e}")
            return

    if not model_best_path.exists():
        print(f"Error: Best model not found at {model_best_path}")
        return

    # 加载模型
    print("Loading models...")
    try:
        model_baseline = YOLO(model_baseline_path)
        model_best = YOLO(model_best_path)
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # 获取测试图片
    if not test_images_dir.exists():
        print(f"Error: Test images directory not found at {test_images_dir}")
        return

    image_files = list(test_images_dir.glob("*.jpg"))
    if not image_files:
        print(f"Error: No images found in {test_images_dir}")
        return

    # 随机选择一张图片
    selected_image = random.choice(image_files)
    print(f"Selected image: {selected_image.name}")

    # 推理
    print("Running inference...")
    # conf=0.25 是默认置信度，可以根据需要调整
    results_baseline = model_baseline(selected_image, conf=0.25)
    results_best = model_best(selected_image, conf=0.25)

    # 获取结果图片 (numpy array)
    # plot() 返回 BGR 格式的 numpy 数组
    img_baseline = results_baseline[0].plot()
    img_best = results_best[0].plot()

    # 转换颜色空间 BGR -> RGB 以便 matplotlib 正确显示
    img_baseline_rgb = cv2.cvtColor(img_baseline, cv2.COLOR_BGR2RGB)
    img_best_rgb = cv2.cvtColor(img_best, cv2.COLOR_BGR2RGB)

    # 创建对比图
    plt.figure(figsize=(16, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(img_baseline_rgb)
    plt.title("Baseline (YOLOv8n)", fontsize=16)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img_best_rgb)
    plt.title("Ours (Best)", fontsize=16)
    plt.axis("off")

    plt.tight_layout()

    output_path = output_dir / f"comparison_{selected_image.stem}.png"
    plt.savefig(output_path)
    print(f"Comparison image saved to {output_path}")

    # 同时也保存单独的图片，方便查看
    cv2.imwrite(str(output_dir / f"baseline_{selected_image.name}"), img_baseline)
    cv2.imwrite(str(output_dir / f"best_{selected_image.name}"), img_best)

    print("Done!")


if __name__ == "__main__":
    generate_comparison()
