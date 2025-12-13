from ultralytics import YOLO
from ultralytics.utils import SETTINGS
import os
import yaml
import shutil


def task3_custom_training():
    print("=== 任务3：自定义数据集训练 ===")

    # 训练模型
    train_model()

    # # 评估模型
    # evaluate_model()

    # # 测试模型
    # test_model()


def train_model():
    SETTINGS.update({"datasets_dir": "./datasets"})
    """训练模型"""
    print("\n2. 开始训练模型...")

    # === 增强1：检查关键文件是否存在 ===
    model_path = "./yolov8x_epoch10/coco_test/weights/best.pt"
    data_config_path = "./coco.yaml"

    if not os.path.exists(model_path):
        print(f"错误：预训练模型文件不存在于 {model_path}")
        return None
    if not os.path.exists(data_config_path):
        print(f"错误：数据集配置文件不存在于 {data_config_path}")
        return None

    # === 原有代码 ===
    try:
        model = YOLO(model_path)
        results = model.train(
            data=data_config_path,
            epochs=50,
            imgsz=640,
            batch=16,
            name="coco_test",
            project="yolov8x_epoch50",
            device=0,
            exist_ok=False,
            save=True,
            save_period=10,
            patience=50,
            amp=True,
            resume=False,
        )
        print("\n训练完成!")
        return results
    except Exception as e:
        print(f"训练失败: {e}")
        print("请确保:")
        # 可以给出更具体的排查方向
        print("1. 数据集配置文件中的路径是否正确。")
        print("2. 图片文件和标注文件是否一一对应且完好无损[1](@ref)。")
        print("3. GPU驱动和CUDA环境是否正常。")
        return None


def evaluate_model():
    """评估模型"""
    print("\n3. 评估模型...")

    # 加载训练好的模型
    try:
        model = YOLO("runs/detect/custom_yolo/weights/best.pt")

        # 验证模型
        metrics = model.val()

        print("评估结果:")
        print(f"mAP50: {metrics.box.map50:.3f}")
        print(f"mAP50-95: {metrics.box.map:.3f}")

    except Exception as e:
        print(f"评估失败: {e}")
        print("请确保模型训练成功")


def test_model():
    """测试模型"""
    print("\n4. 测试模型...")

    try:
        # 加载训练好的模型
        model = YOLO("runs/detect/custom_yolo/weights/best.pt")

        # 测试图片
        if os.path.exists("bus.jpg"):
            print("使用 bus.jpg 进行测试")
            results = model("bus.jpg", save=True)

            result = results[0]
            print(f"检测到 {len(result.boxes)} 个物体")
            for box in result.boxes:
                class_id = int(box.cls)
                class_name = result.names[class_id]
                confidence = float(box.conf)
                print(f"  {class_name}: {confidence:.2f}")
        else:
            print("未找到测试图片 bus.jpg")

    except Exception as e:
        print(f"测试失败: {e}")
        print("请确保模型训练成功")


if __name__ == "__main__":
    task3_custom_training()
