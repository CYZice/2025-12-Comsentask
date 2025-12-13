import os
import random
import shutil
import glob
import sys
from pathlib import Path
import yaml  # 确保导入yaml

# 添加项目根目录到 sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def split_dataset(
    source_dir: str,
    train_dir: str,
    val_dir: str,
    train_ratio: float = 0.8,
    shuffle: bool = True,
    seed: int = 42,
) -> None:
    """
    将数据集分割为训练集和验证集 (YOLO标准格式：images/labels分离)。
    结构如下:
    train/
        images/
        labels/
    val/
        images/
        labels/
    """
    if not os.path.exists(source_dir):
        print(f"[错误] 源目录不存在: {source_dir}")
        return

    # 获取所有图片文件
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(source_dir, ext)))

    if not image_files:
        print(f"[错误] 在 {source_dir} 中未找到图片文件！")
        return

    print(f"[信息] 找到 {len(image_files)} 张图片")

    # 随机打乱
    if shuffle:
        random.seed(seed)
        random.shuffle(image_files)

    # 计算分割点
    split_point = int(len(image_files) * train_ratio)
    train_files = image_files[:split_point]
    val_files = image_files[split_point:]

    print(f"[信息] 训练集: {len(train_files)} 张，验证集: {len(val_files)} 张")

    # --- 定义子目录 ---
    train_img_dir = os.path.join(train_dir, "images")
    train_lbl_dir = os.path.join(train_dir, "labels")
    val_img_dir = os.path.join(val_dir, "images")
    val_lbl_dir = os.path.join(val_dir, "labels")

    # 创建输出目录结构
    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    # 辅助函数：复制文件
    def copy_files(files_list, img_dest, lbl_dest, tag):
        print(f"[信息] 正在处理 {tag}...")
        for i, img_path in enumerate(files_list, 1):
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            txt_path = os.path.join(source_dir, f"{base_name}.txt")

            # 1. 复制图片 -> images/
            shutil.copy2(img_path, img_dest)

            # 2. 复制标注 -> labels/
            if os.path.exists(txt_path):
                shutil.copy2(txt_path, lbl_dest)

            # 显示进度
            if i % 100 == 0:
                print(f"[进度] {tag}: {i}/{len(files_list)}")

    # 执行复制
    copy_files(train_files, train_img_dir, train_lbl_dir, "训练集")
    copy_files(val_files, val_img_dir, val_lbl_dir, "验证集")

    # 复制类别文件 (可选，通常放在根目录备查)
    classes_file = "classes.txt"
    src_classes = os.path.join(source_dir, classes_file)
    if os.path.exists(src_classes):
        # 复制到 dataset 根目录一份，方便查看
        dataset_root = os.path.dirname(train_dir)
        shutil.copy2(src_classes, dataset_root)
        print(f"[信息] 已复制 classes.txt 到数据集根目录")

    print(f"[完成] 数据集分割完成！已分离 images 和 labels 文件夹。")


def create_yaml_config(
    train_dir: str, val_dir: str, classes: list[str], output_path: str = "dataset.yaml"
) -> None:
    """
    创建YOLO训练配置文件。
    注意：在YOLO格式中，path指向 images 文件夹即可，模型会自动查找同级 labels 文件夹。
    """
    # 构造指向 images 的绝对路径
    train_img_path = os.path.abspath(os.path.join(train_dir, "images"))
    val_img_path = os.path.abspath(os.path.join(val_dir, "images"))

    config = {
        # path 是数据集根目录（可选，如果下面 train/val 写绝对路径，这里可以不写）
        "path": os.path.abspath(os.path.dirname(train_dir)),
        "train": train_img_path,  # 指向训练集图片目录
        "val": val_img_path,  # 指向验证集图片目录
        "nc": len(classes),
        "names": {
            i: name for i, name in enumerate(classes)
        },  # 建议使用字典格式 {0: 'cat', ...}
    }

    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(
            config, f, allow_unicode=True, default_flow_style=False, sort_keys=False
        )

    print(f"[信息] 已创建配置文件: {output_path}")
    print(f"      Train 指向: {train_img_path}")
    print(f"      Val   指向: {val_img_path}")


def main():
    """主函数"""
    # --- 配置 ---
    # 请确保这些路径是你实际想要的路径
    source_dir = "D:/Microsoft VS Code/PYTHON/comsen-task-1/project/augmented_data"

    # 建议输出路径不要包含 deep 嵌套，或者直接确保该文件夹是空的
    output_root = "D:/Microsoft VS Code/PYTHON/comsen-task-1/project/dataset_yolo"
    train_dir = os.path.join(output_root, "train")
    val_dir = os.path.join(output_root, "val")

    train_ratio = 0.8  # 训练集比例

    # 检查源目录
    if not os.path.exists(source_dir):
        print(f"[错误] 源目录不存在: {source_dir}")
        print("[提示] 请先运行数据增强脚本生成增强数据")
        return

    # 为了防止路径混淆，建议先清空或新建一个干净的输出目录
    if os.path.exists(output_root):
        print(f"[警告] 输出目录已存在: {output_root}, 即将进行合并或覆盖...")

    # 执行分割
    split_dataset(source_dir, train_dir, val_dir, train_ratio)

    # 读取类别信息
    classes_file = os.path.join(source_dir, "classes.txt")
    if os.path.exists(classes_file):
        with open(classes_file, "r", encoding="utf-8") as f:
            classes = f.read().strip().splitlines()

        # 创建YOLO配置文件 (保存到 output_root 下)
        yaml_path = os.path.join(output_root, "dataset.yaml")
        create_yaml_config(train_dir, val_dir, classes, output_path=yaml_path)

    print("\n[下一步] 使用以下命令开始训练:")
    print(
        f"yolo train data='{os.path.join(output_root, 'dataset.yaml')}' model=yolov8n.pt epochs=100 imgsz=640"
    )


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"缺少必要的库: {e}")
        print("请运行: pip install pyyaml")
