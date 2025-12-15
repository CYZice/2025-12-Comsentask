import sys
from pathlib import Path
import os
import shutil
import random

# 添加项目根目录到 sys.path 以支持绝对导入
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def reorganize_dataset():
    """
    将valid和test数据集重新组合成三个较小的数据集

    Returns:
        bool: 如果重组成功返回 True，否则返回 False
    """
    # 设置随机种子以确保结果可重现
    random.seed(42)

    # 定义路径
    visdrone_path = Path("VISDRONE-2")
    valid_path = visdrone_path / "valid"
    test_path = visdrone_path / "test"

    # 新数据集路径
    new_dataset_path = visdrone_path / "small_datasets"
    train_path = new_dataset_path / "train"
    val_path = new_dataset_path / "val"
    test_path_new = new_dataset_path / "test"

    # 创建新目录结构
    for path in [train_path, val_path, test_path_new]:
        (path / "images").mkdir(parents=True, exist_ok=True)
        (path / "labels").mkdir(parents=True, exist_ok=True)

    # 获取所有valid和test的图片和标签文件
    valid_images = list((valid_path / "images").glob("*.jpg"))
    valid_labels = list((valid_path / "labels").glob("*.txt"))
    test_images = list((test_path / "images").glob("*.jpg"))
    test_labels = list((test_path / "labels").glob("*.txt"))

    # 合并所有文件
    all_images = valid_images + test_images
    all_labels = valid_labels + test_labels

    print(f"[信息] 找到 {len(all_images)} 张图片和 {len(all_labels)} 个标签文件")

    # 随机打乱文件列表
    combined = list(zip(all_images, all_labels))
    random.shuffle(combined)
    all_images, all_labels = zip(*combined)

    # 计算分割点
    total_count = len(all_images)
    train_count = int(total_count * 0.6)  # 60% 用于训练
    val_count = int(total_count * 0.2)  # 20% 用于验证
    test_count = total_count - train_count - val_count  # 剩余20%用于测试

    print(
        f"[信息] 数据集分割: 训练集={train_count}, 验证集={val_count}, 测试集={test_count}"
    )

    # 分割文件
    train_images = all_images[:train_count]
    train_labels = all_labels[:train_count]

    val_images = all_images[train_count : train_count + val_count]
    val_labels = all_labels[train_count : train_count + val_count]

    test_images = all_images[train_count + val_count :]
    test_labels = all_labels[train_count + val_count :]

    # 复制文件到新目录
    def copy_files(images, labels, dest_path, dataset_name):
        """复制图片和标签文件到目标目录"""
        print(f"[信息] 正在复制 {dataset_name} 数据集...")
        for i, (img, label) in enumerate(zip(images, labels)):
            # 复制图片
            shutil.copy2(img, dest_path / "images" / img.name)
            # 复制标签
            shutil.copy2(label, dest_path / "labels" / label.name)

            # 显示进度
            if (i + 1) % 100 == 0 or i == len(images) - 1:
                print(f"[进度] {dataset_name}: 已复制 {i+1}/{len(images)} 个文件")

    # 复制各个数据集
    copy_files(train_images, train_labels, train_path, "训练集")
    copy_files(val_images, val_labels, val_path, "验证集")
    copy_files(test_images, test_labels, test_path_new, "测试集")

    print("[信息] 数据集重组完成!")
    return True


def create_new_data_yaml():
    """
    创建新的data.yaml文件以适应重组后的数据集

    Returns:
        bool: 如果创建成功返回 True，否则返回 False
    """
    # 定义路径
    visdrone_path = Path("VISDRONE-2")
    new_dataset_path = visdrone_path / "small_datasets"

    # 读取原始data.yaml文件
    original_yaml_path = visdrone_path / "data.yaml"
    with open(original_yaml_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 创建新的data.yaml文件
    new_yaml_path = new_dataset_path / "data.yaml"
    with open(new_yaml_path, "w", encoding="utf-8") as f:
        # 更新路径
        f.write("train: ./train/images\n")
        f.write("val: ./val/images\n")
        f.write("test: ./test/images\n\n")

        # 保留类别信息
        for line in lines[3:]:
            f.write(line)

    print(f"[信息] 已创建新的data.yaml文件: {new_yaml_path}")
    return True


if __name__ == "__main__":
    print("[信息] 开始重组数据集...")

    # 重组数据集
    if reorganize_dataset():
        print("[信息] 数据集重组成功!")
    else:
        print("[错误] 数据集重组失败!")
        sys.exit(1)

    # 创建新的data.yaml文件
    if create_new_data_yaml():
        print("[信息] 新的data.yaml文件创建成功!")
    else:
        print("[错误] 新的data.yaml文件创建失败!")
        sys.exit(1)

    print("[信息] 所有操作完成!")
