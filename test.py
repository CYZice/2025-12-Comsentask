from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt


# 任务1：使用预训练模型进行图像检测
def task1_image_detection():
    print("=== 任务1：使用预训练模型进行图像检测 ===")

    # 加载预训练模型（模型会自动下载）
    model = YOLO("yolov8n.pt")  # 首次运行时会自动下载模型文件

    # 对本地图片进行预测
    results = model.predict(
        "bus.jpg",
        save=True,
        show=False,  # 不自动显示，我们用matplotlib显示
        conf=0.5,  # 置信度阈值
        iou=0.45,  # IOU阈值
    )

    # 显示检测结果
    result = results[0]

    # 使用matplotlib显示原图和检测结果
    plt.figure(figsize=(12, 6))

    # 原图
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(result.orig_img, cv2.COLOR_BGR2RGB))
    plt.title("原图")
    plt.axis("off")

    # 检测结果图
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB))
    plt.title("检测结果")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("task1_detection_result.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 打印检测到的物体信息
    print(f"检测到 {len(result.boxes)} 个物体:")
    for box in result.boxes:
        class_id = int(box.cls)
        class_name = result.names[class_id]
        confidence = float(box.conf)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(
            f"  {class_name}: {confidence:.2f} (位置: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}])"
        )

    print(f"\n检测结果已保存到 runs/detect/predict/ 目录")
    print(f"结果对比图已保存为 task1_detection_result.png")

    return model


if __name__ == "__main__":
    # 执行任务1
    model = task1_image_detection()
