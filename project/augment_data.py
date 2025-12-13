import albumentations as A
import cv2
import os
import glob
import random
import sys
from pathlib import Path

# 添加项目根目录到 sys.path 以支持绝对导入
# 使用 .resolve() 确保处理软链接和相对路径的准确性
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# --- 配置 ---
input_dir = "D:/Microsoft VS Code/PYTHON/comsen-task-1/project/labeled_data"  # 标注好的原图和txt所在目录
output_dir = "D:/Microsoft VS Code/PYTHON/comsen-task-1/project/augmented_data"  # 增强后数据存放目录
classes_file = (
    "D:/Microsoft VS Code/PYTHON/comsen-task-1/project/classes.txt"  # 你的类别文件
)
aug_times = 5  # 每张原图生成多少张增强图

# 单类别检测设置 - 强制所有类别ID为0
SINGLE_CLASS_MODE = True  # 启用单类别模式
TARGET_CLASS_ID = 0  # 所有物体都视为这个类别ID

# 增量模式设置
INCREMENTAL_MODE = True  # 启用增量模式，避免覆盖已有数据
SKIP_EXISTING = True  # 如果检测到已有增强数据，跳过该图片


def create_augmentation_pipeline() -> A.Compose:
    """
    创建数据增强流水线。

    Returns:
        A.Compose: 配置好的增强流水线
    """
    # 定义增强流水线 (Pipeline)
    # 注意：bbox_params 是关键，确保坐标会自动转换
    transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),  # 水平翻转
            A.RandomBrightnessContrast(p=0.2),  # 随机亮度对比度
            A.Rotate(limit=15, p=0.5),  # 随机旋转 +/- 15度
            A.GaussNoise(p=0.2),  # 高斯噪点
            A.Blur(blur_limit=3, p=0.1),  # 模糊
            A.RandomShadow(p=0.1),  # 随机阴影
            A.ColorJitter(p=0.2),  # 颜色抖动
            # 也可以加 Cutout 或 Mosaic，但基础增强上面这些够用了
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )

    return transform


def read_classes(input_path: str) -> list[str]:
    """
    读取类别文件 - 单类别模式只返回一个类别。

    Args:
        input_path (str): 类别文件路径

    Returns:
        list[str]: 类别列表（单类别模式下只包含一个类别）
    """
    if SINGLE_CLASS_MODE:
        # 单类别模式：只返回一个类别
        return ["target"]

    # 多类别模式：正常读取类别文件
    classes_file_path = os.path.join(input_path, classes_file)
    if not os.path.exists(classes_file_path):
        print(f"[警告] 未找到类别文件: {classes_file_path}")
        return []

    with open(classes_file_path, "r", encoding="utf-8") as f:
        classes = f.read().strip().splitlines()

    return classes


def parse_yolo_annotation(
    txt_path: str, img_height: int, img_width: int
) -> tuple[list, list]:
    """
    解析YOLO格式的标注文件 - 单类别模式下强制所有类别ID为0。

    Args:
        txt_path (str): 标注文件路径
        img_height (int): 图片高度
        img_width (int): 图片宽度

    Returns:
        tuple[list, list]: 边界框列表和类别ID列表
    """
    bboxes = []
    category_ids = []

    if not os.path.exists(txt_path):
        return bboxes, category_ids

    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 解析 YOLO 格式: class_id x_center y_center width height
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        if SINGLE_CLASS_MODE:
            # 单类别模式：强制使用目标类别ID
            class_id = TARGET_CLASS_ID
        else:
            # 多类别模式：使用原始类别ID
            class_id = int(parts[0])

        # YOLO格式是相对坐标，需要保持为相对坐标
        bbox = [float(x) for x in parts[1:5]]  # [x_center, y_center, width, height]
        bboxes.append(bbox)
        category_ids.append(class_id)

    return bboxes, category_ids


def save_augmented_data(
    output_path: str,
    base_name: str,
    image: any,
    bboxes: list,
    category_ids: list,
    suffix: str = "",
) -> None:
    """
    保存增强后的数据和标注。

    Args:
        output_path (str): 输出目录路径
        base_name (str): 基础文件名
        image (any): 增强后的图片
        bboxes (list): 增强后的边界框
        category_ids (list): 类别ID列表
        suffix (str): 文件名后缀
    """
    # 保存增强后的图片
    img_name = f"{base_name}{suffix}.jpg"
    cv2.imwrite(os.path.join(output_path, img_name), image)

    # 保存增强后的标签
    txt_name = f"{base_name}{suffix}.txt"
    with open(os.path.join(output_path, txt_name), "w", encoding="utf-8") as f:
        for bbox, cat_id in zip(bboxes, category_ids):
            # 限制坐标在 0-1 之间 (旋转可能会导致轻微越界)
            clipper_bbox = [min(max(x, 0.0), 1.0) for x in bbox]
            line = f"{cat_id} {' '.join(map(str, clipper_bbox))}\n"
            f.write(line)


def get_next_file_number(output_path: str) -> int:
    """
    获取下一个可用的文件编号。

    Args:
        output_path (str): 输出目录路径

    Returns:
        int: 下一个可用的文件编号
    """
    if not os.path.exists(output_path):
        return 0

    # 获取所有jpg文件
    existing_files = glob.glob(os.path.join(output_path, "*.jpg"))

    max_number = -1
    for file_path in existing_files:
        filename = os.path.basename(file_path)
        # 提取文件名中的数字部分（去掉扩展名）
        name_without_ext = os.path.splitext(filename)[0]

        # 处理不同的命名格式：000000_orig, 000000_aug_0, 000001_orig 等
        if "_" in name_without_ext:
            number_part = name_without_ext.split("_")[0]
        else:
            number_part = name_without_ext

        try:
            file_number = int(number_part)
            max_number = max(max_number, file_number)
        except ValueError:
            continue

    return max_number + 1 if max_number >= 0 else 0


def augment_dataset() -> None:
    """
    执行数据增强的主函数 - 支持单类别模式。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[信息] 创建输出目录: {output_dir}")

    # 读取类别
    classes = read_classes(input_dir)
    if classes:
        # 复制 classes.txt 到输出目录
        with open(os.path.join(output_dir, classes_file), "w", encoding="utf-8") as f:
            f.write("\n".join(classes))
        if SINGLE_CLASS_MODE:
            print(f"[信息] 单类别模式：类别为 '{classes[0]}' (ID: {TARGET_CLASS_ID})")
        else:
            print(f"[信息] 找到 {len(classes)} 个类别: {', '.join(classes)}")

    # 获取所有图片
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))

    if not image_paths:
        print(f"[错误] 在 {input_dir} 中未找到图片文件！")
        print("[提示] 请确保已使用LabelImg完成标注，图片和标注文件在同一目录")
        return

    # 获取下一个可用的文件编号
    next_file_number = get_next_file_number(output_dir)
    print(
        f"[信息] 找到 {len(image_paths)} 张图片，下一个文件编号从 {next_file_number:06d} 开始"
    )

    if INCREMENTAL_MODE and next_file_number > 0:
        print(
            f"[增量模式] 检测到已有增强数据，将从编号 {next_file_number:06d} 开始继续"
        )

    # 创建增强流水线
    transform = create_augmentation_pipeline()

    success_count = 0
    error_count = 0
    current_file_number = next_file_number

    for img_path in image_paths:
        try:
            # 读取图片
            image = cv2.imread(img_path)
            if image is None:
                print(f"[警告] 无法读取图片: {img_path}")
                continue

            h, w, _ = image.shape

            # 读取对应的 txt 标签
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            txt_path = os.path.join(input_dir, f"{base_name}.txt")

            bboxes, category_ids = parse_yolo_annotation(txt_path, h, w)

            # 生成新的文件编号
            new_base_name = f"{current_file_number:06d}"
            current_file_number += 1

            # 检查是否有框 (没框的图跳过或仅做图片增强)
            if not bboxes:
                print(f"[警告] {base_name} 没有找到标注框，跳过增强")
                continue

            # 在单类别模式下，显示类别ID转换信息
            if SINGLE_CLASS_MODE and category_ids:
                original_ids = set(category_ids)
                if len(original_ids) > 1 or TARGET_CLASS_ID not in original_ids:
                    print(
                        f"[信息] {base_name}: 原始类别ID {list(original_ids)} -> 统一为 {TARGET_CLASS_ID}"
                    )

            # 先保存一张原图进去
            save_augmented_data(
                output_dir, new_base_name, image, bboxes, category_ids, "_orig"
            )

            # 生成 aug_times 张增强图
            for i in range(aug_times):
                try:
                    # 执行变换
                    transformed = transform(
                        image=image, bboxes=bboxes, class_labels=category_ids
                    )
                    transformed_image = transformed["image"]
                    transformed_bboxes = transformed["bboxes"]
                    transformed_class_labels = transformed["class_labels"]

                    # 保存增强后的数据
                    suffix = f"_aug_{i}"
                    save_augmented_data(
                        output_dir,
                        new_base_name,
                        transformed_image,
                        transformed_bboxes,
                        transformed_class_labels,
                        suffix,
                    )

                    success_count += 1

                except Exception as e:
                    print(f"[警告] {new_base_name} 第{i}次增强失败: {e}")
                    error_count += 1

        except Exception as e:
            print(f"[错误] 处理 {img_path} 时出错: {e}")
            error_count += 1

    print(f"[完成] 数据增强完成！")
    if SINGLE_CLASS_MODE:
        print(f"[统计] 单类别模式：类别ID统一为 {TARGET_CLASS_ID}")
    print(f"[统计] 成功增强: {success_count} 张，错误: {error_count} 张")
    print(f"[输出] 增强数据保存在: {output_dir}")


def main():
    """主函数"""
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"[错误] 输入目录不存在: {input_dir}")
        print("[提示] 请先使用LabelImg完成图片标注")
        return

    # 检查输出目录是否已有数据
    if os.path.exists(output_dir) and os.listdir(output_dir):
        print(f"[警告] 输出目录 {output_dir} 已存在文件")
        print(f"当前增量模式状态: {'开启' if INCREMENTAL_MODE else '关闭'}")

        if INCREMENTAL_MODE:
            print("[提示] 增量模式已开启，新文件将自动接续在现有文件之后")
        else:
            print("[警告] 增量模式已关闭，可能会覆盖现有文件")
            response = input("是否继续？(y/N): ").strip().lower()
            if response != "y":
                print("[取消] 操作已取消")
                return

    augment_dataset()


if __name__ == "__main__":
    main()
