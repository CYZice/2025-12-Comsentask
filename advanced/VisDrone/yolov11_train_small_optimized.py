import sys
from pathlib import Path
import os
import warnings
import multiprocessing

# 添加项目根目录到 sys.path 以支持绝对导入
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Windows环境配置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

from ultralytics import YOLO


def train_model():
    """
    主函数：使用小数据集训练YOLO模型

    Returns:
        bool: 如果训练成功返回 True，否则返回 False
    """
    # 使用YOLOv8模型（作为YOLOv11的替代）
    model = YOLO("runs/train/baseline/weights/best.pt")  # 使用YOLOv8n模型

    # 使用小数据集的data.yaml路径
    data_yaml_path = "small_datasets/data.yaml"

    # 训练配置
    results = model.train(
        data=data_yaml_path,  # 数据集配置文件路径
        epochs=100,  # 训练轮数
        imgsz=640,  # 图像尺寸
        batch=8,  # 批次大小，如果显存不够可以改小 (8 或 4)
        device=0,  # 使用第一张显卡
        # === Windows 关键配置 ===
        workers=0,  # 设置为0，避免多进程死锁/卡顿
        amp=True,  # 混合精度训练，显存小的话设为 True (默认就是True)
        project="runs/train",  # 项目目录
        name="baseline",  # 实验名称
        save=True,  # 保存训练结果
    )

    # 记下最后的 mAP50 值，比如是 0.15
    print("[信息] 训练完成!")
    return True


if __name__ == "__main__":
    # 必须放在这里面执行
    multiprocessing.freeze_support()
    train_model()
