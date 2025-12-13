#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import subprocess
import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any

# 添加项目根目录到 sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ==================== 工具函数 ====================


def check_ultralytics_installation() -> bool:
    try:
        import ultralytics

        print(f"[信息] Ultralytics版本: {ultralytics.__version__}")
        return True
    except ImportError:
        return False


def install_ultralytics() -> bool:
    print("[信息] 正在安装Ultralytics...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "ultralytics"], check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"[错误] 安装失败: {e}")
        return False


def validate_dataset_structure(dataset_path: str) -> bool:
    """验证刚才生成的 images/labels 结构是否存在"""
    # 检查标准YOLO结构
    required_dirs = [
        os.path.join(dataset_path, "train", "images"),
        os.path.join(dataset_path, "train", "labels"),
        os.path.join(dataset_path, "val", "images"),
        os.path.join(dataset_path, "val", "labels"),
    ]

    missing_dirs = [d for d in required_dirs if not os.path.exists(d)]

    if missing_dirs:
        print(f"[错误] 数据集结构不完整，缺失目录:\n" + "\n".join(missing_dirs))
        return False

    # 检查是否为空
    for d in required_dirs:
        if not os.listdir(d):
            print(f"[警告] 目录看似为空: {d}")

    print("[信息] 数据集结构验证通过 (Standard YOLOv8 Format)")
    return True


def create_training_config(data_yaml_path: str, **kwargs) -> Dict[str, Any]:
    # 基础配置
    config = {
        "data": data_yaml_path,
        "model": "yolov8n.pt",  # 推荐从 nano 模型开始
        "epochs": 200,  # 训练轮次
        "imgsz": 640,  # 图片大小
        "batch": 16,  # 显存如果不够（爆显存），请把这个改小，比如 8 或 4
        "workers": 16,  # Windows下建议不要设置太大，容易报错
        "device": 0,  # 这是这一步的关键！指定使用 GPU (0号卡)
        "project": "runs/train",  # 结果保存路径
        "name": "extended",  # 实验名称
        "exist_ok": True,  # 允许覆盖
        "patience": 50,  # 早停机制
        "verbose": True,
        "save": True,  # 保存训练结果
    }
    config.update(kwargs)
    return config


def train_yolo_model(config: Dict[str, Any]) -> bool:
    try:
        from ultralytics import YOLO

        # 强制检查 GPU
        import torch

        if not torch.cuda.is_available():
            print("\n[严重警告] 你的 PyTorch 无法检测到 GPU！训练将非常慢。")
            print("请检查之前的 CUDA 安装步骤。\n")
        else:
            print(f"[信息] 使用设备: {torch.cuda.get_device_name(0)}")

        print(f"[开始] 加载模型 {config['model']}...")
        model = YOLO(config["model"])

        print(f"[开始] 启动训练...")
        results = model.train(**config)

        print(f"\n[成功] 训练完成！结果保存在: {results.save_dir}")
        return True

    except Exception as e:
        print(f"\n[错误] 训练过程中断: {e}")
        # 如果是 Windows 常见的 DataLoader worker 错误
        if "num_workers" in str(e):
            print("[建议] 尝试将 workers 设置为 0 再试一次。")
        return False


# ==================== 主逻辑 ====================


def main():
    print("=" * 60)
    print("YOLOv8 训练启动脚本")
    print("=" * 60)

    # 1. 检查库
    if not check_ultralytics_installation():
        if not install_ultralytics():
            return

    # --- 关键配置：对齐 split_dataset.py 生成的路径 ---
    # 假设该脚本在 project 下，数据在 project/dataset_yolo 下
    # 如果你的文件夹结构不同，请手动修改 dataset_root 变量

    # 自动定位到上一步生成的 'dataset_yolo' 文件夹
    current_dir = Path(__file__).resolve().parent
    # 尝试几种可能的路径（兼容你的 VS Code 目录结构）
    possible_paths = [
        os.path.join(current_dir, "dataset_yolo"),  # 脚本就在 project 目录下
        os.path.join(
            current_dir, "project", "dataset_yolo"
        ),  # 脚本在 comsen-task-1 目录下
        "D:/Microsoft VS Code/PYTHON/comsen-task-1/project/dataset_yolo",  # 绝对路径兜底
    ]

    dataset_root = None
    for p in possible_paths:
        if os.path.exists(p):
            dataset_root = p
            break

    if not dataset_root:
        print(f"[错误] 找不到数据集目录 'dataset_yolo'。")
        print(f"请检查上一步 split_dataset.py 是否成功运行，或手动修改本脚本中的路径。")
        return

    print(f"[路径] 数据集目录: {dataset_root}")

    # 2. 定位 YAML 文件
    yaml_path = os.path.join(dataset_root, "dataset.yaml")
    if not os.path.exists(yaml_path):
        print(f"[错误] 找不到配置文件: {yaml_path}")
        return

    # 3. 验证结构
    if not validate_dataset_structure(dataset_root):
        return

    # 4. 读取类别数（自动适配单类或多类）
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            y_data = yaml.safe_load(f)
            nc = y_data.get("nc", 1)
            print(f"[配置] 检测到类别数量 (nc): {nc}")
    except:
        nc = 1  # 默认兜底

    # 5. 生成配置并开始
    train_config = create_training_config(
        data_yaml_path=yaml_path,
        batch=8,  # 显存小的话改小这个
        workers=16,  # Windows下设小一点比较稳
    )

    train_yolo_model(train_config)


if __name__ == "__main__":
    main()
