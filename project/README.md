# 目标检测数据集制作工具

这是一个基于YOLO格式的目标检测数据集制作工具，支持从视频中提取帧、使用LabelImg进行标注，以及使用Albumentations进行数据增强。

## 功能特点

- 🎬 **视频抽帧**: 自动从视频中提取帧，支持多种视频格式
- 🏷️ **标注支持**: 与LabelImg无缝集成，支持YOLO格式
- 🔄 **数据增强**: 使用Albumentations进行丰富的数据增强
- 📊 **完整流程**: 从原始视频到训练就绪的数据集

## 项目结构

```
project/
├── videos/           # 放置原始视频文件
├── raw_images/     # 视频抽帧后的图片
├── labeled_data/    # 使用LabelImg标注后的数据
├── augmented_data/  # 数据增强后的最终数据集
├── train/           # 训练集（手动分割）
├── val/              # 验证集（手动分割）
├── extract_frames.py # 视频抽帧脚本
├── augment_data.py  # 数据增强脚本
└── README.md        # 本文件
```

## 使用流程

### 第一步：视频抽帧

1. 将你的视频文件放入 `videos/` 目录
2. 运行抽帧脚本：
   ```bash
   python extract_frames.py
   ```
- 自动扫描 `videos/` 目录下的所有视频文件
- 按指定间隔（默认每秒1帧）提取图片
- 抽帧后的图片将保存在 `raw_images/` 目录

**配置参数**（在脚本中修改）：
- `frame_interval`: 抽帧间隔，默认为15帧
- 支持的视频格式：.mp4, .avi, .mov, .mkv

### 第二步：图片标注

1. 安装LabelImg：
   ```bash
   pip install labelImg
   ```

2. 启动LabelImg：
   ```bash
   labelImg
   ```

3. 设置路径：
   - **Open Dir**: 选择 `raw_images/` 文件夹
   - **Change Save Dir**: 选择 `labeled_data/` 文件夹

4. 关键设置：
   - 点击左侧 **PascalVOC** 按钮，切换为 **YOLO** 格式

5. 标注快捷键：
   - `W`: 创建标注框
   - `D`: 下一张图片
   - `A`: 上一张图片
   - `Del`: 删除选中框

6. 标注完成后，`labeled_data/` 目录将包含：
   - 图片文件（.jpg）
   - 对应的标注文件（.txt）
   - 类别文件（classes.txt）

### 第三步：数据增强（可选）
```bash
python augment_data.py
```
- 支持单类别模式（所有类别ID自动转换为0）
- 提供多种增强方式：翻转、旋转、亮度调整等
- 增强后的数据保存在 `augmented_data/` 目录

### 第四步：数据集分割

```bash
python split_dataset.py
```
- 按比例分割训练集和验证集（默认8:2）
- 自动生成符合YOLO格式的目录结构
- 复制类别文件到训练目录

### 第五步：单类别训练

```bash
python train_single_class.py
```
- 自动验证数据集结构和标注文件
- 使用YOLOv8预训练模型进行迁移学习
- 支持单类别目标检测训练

### 快速开始

```bash
python quick_start.py
```
- 一键完成所有步骤
- 支持分步执行或完整流程

### 第四步：数据集分割

数据增强完成后，你需要手动将 `augmented_data/` 中的数据分割为训练集和验证集：

1. 将大约80%的数据移动到 `train/` 目录
2. 将剩余20%的数据移动到 `val/` 目录

或者使用脚本自动分割（可选）：
```bash
# 创建数据集分割脚本
python create_dataset_split.py
```

## 依赖安装

```bash
# 安装所有依赖
pip install -r requirements.txt
```

主要依赖：
- opencv-python: 视频处理和图像操作
- albumentations: 数据增强
- numpy: 数值计算

## 使用示例

### 示例1：制作杯子检测数据集

```bash
# 1. 将拍摄的视频放入videos目录
# 2. 视频抽帧
python extract_frames.py

# 3. 使用LabelImg标注（手动操作）
labelImg

# 4. 数据增强
python augment_data.py

# 5. 分割数据集（手动或使用脚本）
```

### 示例2：自定义增强参数

编辑 `augment_data.py` 文件：
```python
# 调整增强强度
aug_times = 10  # 每张图片生成10张增强图片

# 自定义增强流水线
transform = A.Compose([
    A.HorizontalFlip(p=0.8),  # 80%概率水平翻转
    A.RandomBrightnessContrast(p=0.5),  # 50%概率调整亮度对比度
    # 添加更多增强操作...
])
```

## 注意事项

1. **视频质量**: 建议使用清晰、光线充足的视频
2. **多角度拍摄**: 为获得更好的检测效果，从不同角度和距离拍摄目标
3. **标注质量**: 确保标注框准确包围目标物体
4. **数据平衡**: 尽量保持各类别的数据量平衡
5. **增强适度**: 过度的增强可能会影响模型性能

## 单类别目标检测训练指南

### 重要原则

1. **标注文件要求**：
   - 所有标注文件的类别ID必须为0
   - 错误做法：`99 0.45 0.67 0.1 0.2`（类别ID为99）
   - 正确做法：`0 0.45 0.67 0.1 0.2`（类别ID为0）

2. **配置文件设置**：
   - 类别数量必须设置为1：`nc: 1`
   - 类别名称列表只包含一个名称：`names: [0: 'your_class_name']`

3. **训练策略**：
   - 使用预训练模型（如yolov8n.pt）进行迁移学习
   - 保留主干网络权重，只更新输出层

### 配置文件示例（data.yaml）

```yaml
# 数据集路径
path: ./train
train: images/train
val: images/val

# 单类别设置
nc: 1  # 类别数量必须为1
names:
  0: 'target'  # 类别名称可以自定义
```

### 训练命令

```bash
# 使用本项目的训练脚本（推荐）
python train_single_class.py

# 或使用YOLOv8命令行
yolo detect train data=data.yaml model=yolov8n.pt epochs=100 imgsz=640
```

### 训练过程说明

1. **模型加载**：加载预训练的YOLOv8n模型（原本支持80类）
2. **网络调整**：自动调整输出层，只输出1个类别
3. **权重保留**：保留主干网络的特征提取能力
4. **重点训练**：主要训练新的输出层权重

### 验证训练结果

训练完成后，可以在以下目录找到结果：
- `runs/detect/single_class_training/` - 训练结果
- `best.pt` - 最佳模型权重
- `results.csv` - 训练过程数据

## 扩展功能

你可以根据需要扩展以下功能：
- 添加更多的数据增强操作
- 支持其他标注格式（如COCO、VOC）
- 自动数据集分割
- 数据集质量检查
- 标注可视化工具

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 联系方式

如有问题，请在GitHub上提交Issue。