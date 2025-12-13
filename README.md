# YOLO目标检测项目

## 项目简介
这是一个YOLO目标检测学习和实践项目，包含了从环境配置到模型训练的完整流程。

## 项目结构
```
project/
├── reports/              # 报告文档
│   ├── learning_notes.md # 学习笔记
│   ├── practice_report.md # 实践报告
│   ├── images/          # 报告用图片
│   └── videos/          # 报告用视频
├── codes/               # 代码文件
│   ├── task2_video_detection.py  # 视频检测
│   ├── task3_custom_training.py  # 自定义训练
│   └── utils/           # 工具函数
├── datasets/            # 数据集配置
│   ├── coco.yaml       # COCO数据集配置
│   └── custom_dataset.yaml # 自定义数据集配置
├── results/             # 实验结果
│   ├── images/         # 检测结果图片
│   ├── videos/         # 检测视频结果
│   └── models/         # 训练模型
├── screenshots/         # 环境配置截图
└── docs/               # 其他文档
```

## 使用说明

### 环境配置
```bash
pip install -r requirements.txt
```

### 运行检测
```bash
python codes/task2_video_detection.py
python codes/task3_custom_training.py
```

## 注意事项
- 数据集文件较大，已添加到.gitignore中
- 训练模型文件不上传到GitHub
- 检测结果保存在results目录中
