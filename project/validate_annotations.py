#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO标注文件验证工具
用于验证标注文件是否符合单类别目标检测要求
"""

import sys
from pathlib import Path

# 添加项目根目录到 sys.path 以支持绝对导入
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import os
import glob
from typing import List, Tuple, Dict, Optional


def validate_annotation_file(file_path: str) -> Tuple[bool, List[str], Dict[str, int]]:
    """
    验证单个标注文件。
    
    Args:
        file_path (str): 标注文件路径。
        
    Returns:
        Tuple[bool, List[str], Dict[str, int]]: (是否有效, 错误列表, 统计信息)
    """
    errors = []
    stats = {
        'total_lines': 0,
        'valid_lines': 0,
        'class_ids': {},
        'invalid_format_lines': 0,
        'empty_lines': 0
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        stats['total_lines'] = len(lines)
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            if not line:
                stats['empty_lines'] += 1
                continue
            
            parts = line.split()
            
            # 检查格式
            if len(parts) < 5:
                errors.append(f"{os.path.basename(file_path)}:{line_num} - 格式错误（需要至少5个值，当前{len(parts)}个）")
                stats['invalid_format_lines'] += 1
                continue
            
            try:
                class_id = int(parts[0])
                
                # 检查类别ID
                if class_id < 0:
                    errors.append(f"{os.path.basename(file_path)}:{line_num} - 类别ID不能为负数: {class_id}")
                    continue
                
                # 记录类别ID统计
                stats['class_ids'][str(class_id)] = stats['class_ids'].get(str(class_id), 0) + 1
                
                # 检查坐标值
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # 检查坐标范围（YOLO格式应该在0-1之间）
                if not (0 <= x_center <= 1):
                    errors.append(f"{os.path.basename(file_path)}:{line_num} - x_center超出范围: {x_center}")
                    continue
                
                if not (0 <= y_center <= 1):
                    errors.append(f"{os.path.basename(file_path)}:{line_num} - y_center超出范围: {y_center}")
                    continue
                
                if not (0 <= width <= 1):
                    errors.append(f"{os.path.basename(file_path)}:{line_num} - width超出范围: {width}")
                    continue
                
                if not (0 <= height <= 1):
                    errors.append(f"{os.path.basename(file_path)}:{line_num} - height超出范围: {height}")
                    continue
                
                stats['valid_lines'] += 1
                
            except ValueError as e:
                errors.append(f"{os.path.basename(file_path)}:{line_num} - 数值转换错误: {e}")
                stats['invalid_format_lines'] += 1
                continue
            
    except Exception as e:
        errors.append(f"{os.path.basename(file_path)} - 文件读取错误: {e}")
        return False, errors, stats
    
    return len(errors) == 0, errors, stats


def validate_dataset_directory(dataset_path: str, expected_class_id: Optional[int] = 0) -> Tuple[bool, List[str], Dict[str, any]]:
    """
    验证整个数据集的标注文件。
    
    Args:
        dataset_path (str): 数据集路径。
        expected_class_id (Optional[int]): 期望的类别ID，默认为0（单类别）。
        
    Returns:
        Tuple[bool, List[str], Dict[str, any]]: (是否有效, 错误列表, 汇总统计)
    """
    errors = []
    summary = {
        'total_files': 0,
        'valid_files': 0,
        'invalid_files': 0,
        'total_objects': 0,
        'class_distribution': {},
        'files_with_wrong_class': [],
        'empty_files': []
    }
    
    # 查找所有标注文件
    label_dirs = ['labels/train', 'labels/val']
    all_label_files = []
    
    for label_dir in label_dirs:
        label_path = os.path.join(dataset_path, label_dir)
        if os.path.exists(label_path):
            txt_files = glob.glob(os.path.join(label_path, "*.txt"))
            all_label_files.extend(txt_files)
    
    if not all_label_files:
        errors.append(f"在 {dataset_path} 中未找到标注文件")
        return False, errors, summary
    
    summary['total_files'] = len(all_label_files)
    
    print(f"[信息] 找到 {len(all_label_files)} 个标注文件，开始验证...")
    
    for i, file_path in enumerate(all_label_files, 1):
        if i % 100 == 0:
            print(f"[进度] 已验证 {i}/{len(all_label_files)} 个文件")
        
        is_valid, file_errors, stats = validate_annotation_file(file_path)
        
        if is_valid:
            summary['valid_files'] += 1
        else:
            summary['invalid_files'] += 1
            errors.extend(file_errors)
        
        # 更新总体统计
        summary['total_objects'] += stats['valid_lines']
        
        # 更新类别分布
        for class_id, count in stats['class_ids'].items():
            summary['class_distribution'][class_id] = summary['class_distribution'].get(class_id, 0) + count
        
        # 检查是否有错误的类别ID
        if expected_class_id is not None:
            file_class_ids = set(stats['class_ids'].keys())
            if file_class_ids and str(expected_class_id) not in file_class_ids:
                summary['files_with_wrong_class'].append(os.path.basename(file_path))
        
        # 检查空文件
        if stats['total_lines'] == 0 or (stats['total_lines'] == stats['empty_lines']):
            summary['empty_files'].append(os.path.basename(file_path))
    
    return len(errors) == 0, errors, summary


def print_validation_report(is_valid: bool, errors: List[str], summary: Dict[str, any]) -> None:
    """
    打印验证报告。
    
    Args:
        is_valid (bool): 验证结果。
        errors (List[str]): 错误列表。
        summary (Dict[str, any]): 汇总统计。
    """
    print("\n" + "="*60)
    print("YOLO标注文件验证报告")
    print("="*60)
    
    # 总体结果
    if is_valid:
        print("✅ [结果] 验证通过")
    else:
        print("❌ [结果] 验证失败")
    
    # 文件统计
    print(f"\n📊 [文件统计]")
    print(f"   总文件数: {summary['total_files']}")
    print(f"   有效文件: {summary['valid_files']}")
    print(f"   无效文件: {summary['invalid_files']}")
    
    # 对象统计
    print(f"\n🎯 [对象统计]")
    print(f"   总对象数: {summary['total_objects']}")
    
    # 类别分布
    if summary['class_distribution']:
        print(f"\n🏷️  [类别分布]")
        for class_id, count in sorted(summary['class_distribution'].items(), key=lambda x: int(x[0])):
            print(f"   类别ID {class_id}: {count} 个对象")
    
    # 特殊文件
    if summary['files_with_wrong_class']:
        print(f"\n⚠️  [类别错误文件] ({len(summary['files_with_wrong_class'])}个)")
        for filename in summary['files_with_wrong_class'][:5]:  # 只显示前5个
            print(f"   {filename}")
        if len(summary['files_with_wrong_class']) > 5:
            print(f"   ... 还有 {len(summary['files_with_wrong_class']) - 5} 个文件")
    
    if summary['empty_files']:
        print(f"\n📄 [空标注文件] ({len(summary['empty_files'])}个)")
        for filename in summary['empty_files'][:5]:  # 只显示前5个
            print(f"   {filename}")
        if len(summary['empty_files']) > 5:
            print(f"   ... 还有 {len(summary['empty_files']) - 5} 个文件")
    
    # 错误详情
    if errors:
        print(f"\n❗ [错误详情] ({len(errors)}个错误)")
        for error in errors[:10]:  # 只显示前10个错误
            print(f"   {error}")
        if len(errors) > 10:
            print(f"   ... 还有 {len(errors) - 10} 个错误")
    
    # 建议
    print(f"\n💡 [建议]")
    if summary['class_distribution'] and len(summary['class_distribution']) > 1:
        print("   ⚠️  检测到多个类别ID，单类别训练要求所有类别ID为0")
        print("   📋 建议运行: python validate_annotations.py --fix-class-ids")
    
    if summary['invalid_files'] > 0:
        print("   ⚠️  存在格式错误的标注文件")
        print("   📋 请检查标注文件格式是否符合YOLO标准")
    
    if summary['empty_files']:
        print("   ⚠️  存在空标注文件")
        print("   📋 请确认这些图片是否真的没有任何目标对象")
    
    if is_valid:
        print("   ✅ 标注文件验证通过，可以进行训练")
    
    print("\n" + "="*60)


def main():
    """
    主函数。
    """
    print("YOLO标注文件验证工具")
    print("="*60)
    
    # 默认数据集路径
    dataset_path = "train"
    expected_class_id = 0  # 默认期望类别ID为0（单类别）
    
    if not os.path.exists(dataset_path):
        print(f"[错误] 数据集路径不存在: {dataset_path}")
        print("[提示] 请先运行 split_dataset.py 创建数据集")
        return
    
    print(f"[信息] 验证数据集: {dataset_path}")
    print(f"[信息] 期望类别ID: {expected_class_id}")
    
    # 验证数据集
    is_valid, errors, summary = validate_dataset_directory(dataset_path, expected_class_id)
    
    # 打印报告
    print_validation_report(is_valid, errors, summary)
    
    # 返回码
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()