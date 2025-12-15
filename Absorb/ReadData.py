#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
读取Absorb文件夹数据的程序
功能：
1. 从"数据/Absorb"文件夹读取数据
2. 每个元素的名字作为文件夹，文件夹内部是对应的数据
3. 存储为一个字典，键为元素名字、值为处理好的数据
4. 每个元素的数据处理：
   - 只关注txt文件
   - 文件名形如"SAg_E22.1N1000000Sn32Layers.txt"，提取Layers前面的数字作为衰减片总层数
   - 每个txt文件：第一列是能量探测器的index（正比于能量）、第二列是事件数
   - 提取数据：总事件数（求和）、高斯峰位置（调用高斯峰拟合）、高斯峰面积（使用峰高和峰宽按照高斯分布计算）
5. 每个文件夹有多个这样的文件，都需要这样处理，并按照衰减片层数分类
6. 整理数据：把数据存成二维数组，第一个分量是衰减片层数、第二个是总事件数、第三个是高斯峰面积、第四个是高斯峰位置
"""

import os
import sys
import numpy as np
import glob
import re

# 添加当前目录到路径，以便导入高斯峰拟合模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/MSL")

# 导入高斯峰拟合中的函数
try:
    from 高斯峰拟合 import (
        read_data,
        calculate_total_events,
        calculate_index_statistics,
        iterative_gaussian_fit
    )
    GAUSSIAN_FIT_AVAILABLE = True
    print("成功导入高斯峰拟合模块")
except ImportError as e:
    print(f"警告: 无法导入高斯峰拟合模块 - {e}")
    print("将使用本地实现的简化版本")
    GAUSSIAN_FIT_AVAILABLE = False


def get_absorb_folder_path():
    """
    获取Absorb文件夹的路径
    
    返回:
    absorb_path: Absorb文件夹的完整路径
    """
    # 获取当前脚本的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建Absorb文件夹路径
    # 当前目录: D:\同步文件\课程作业\2025秋\近物实验\X射线标识谱\数据\数据处理\Absorb
    # Absorb数据目录: D:\同步文件\课程作业\2025秋\近物实验\X射线标识谱\数据\Absorb
    # 需要向上两级到数据目录，然后进入Absorb
    data_dir = os.path.dirname(os.path.dirname(current_dir))
    absorb_path = os.path.join(data_dir, "Absorb")
    
    return absorb_path


def extract_layers_from_filename(filename):
    """
    从文件名中提取衰减片层数
    
    参数:
    filename: 文件名，例如 "SAg_E22.1N1000000Sn32Layers.txt"
    
    返回:
    layers: 衰减片层数（整数），如果提取失败则返回None
    """
    # 使用正则表达式匹配Sn后面的数字（直到Layers）
    # 模式：Sn后跟0个或多个数字，然后是Layers
    pattern = r'Sn(\d+)Layers'
    match = re.search(pattern, filename)
    
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    else:
        return None


def extract_element_name_from_folder(folder_path):
    """
    从文件夹路径中提取元素名
    
    参数:
    folder_path: 文件夹路径，例如 "数据/Absorb/Ag"
    
    返回:
    element_name: 元素名，例如 "Ag"
    """
    # 获取文件夹名（路径的最后一部分）
    folder_name = os.path.basename(folder_path)
    return folder_name


def find_txt_files_in_folder(folder_path):
    """
    在文件夹中查找所有.txt文件
    
    参数:
    folder_path: 文件夹路径
    
    返回:
    txt_files: 所有.txt文件的完整路径列表
    """
    # 使用glob查找所有.txt文件
    pattern = os.path.join(folder_path, "*.txt")
    txt_files = glob.glob(pattern)
    
    # 过滤掉可能的非数据文件（如err.out, log.out等）
    filtered_files = []
    for file_path in txt_files:
        file_name = os.path.basename(file_path)
        # 只保留以S开头（表示元素）的.txt文件
        if file_name.startswith('S'):
            filtered_files.append(file_path)
    
    return sorted(filtered_files)


def calculate_gaussian_area(peak_height, peak_std):
    """
    计算高斯峰面积
    
    高斯分布公式: f(x) = A * exp(-(x-μ)²/(2σ²))
    其中A是峰高，σ是标准差
    
    高斯分布的积分（面积） = A * σ * √(2π)
    
    参数:
    peak_height: 峰高
    peak_std: 标准差
    
    返回:
    area: 高斯峰面积
    """
    return peak_height * peak_std * np.sqrt(2 * np.pi)


def process_single_txt_file(file_path):
    """
    处理单个txt文件
    
    参数:
    file_path: txt文件路径
    
    返回:
    result_dict: 包含处理结果的字典，包括：
        - layers: 衰减片层数
        - total_events: 总事件数
        - peak_position: 高斯峰位置
        - peak_std: 高斯峰标准差
        - peak_height: 高斯峰高度
        - peak_area: 高斯峰面积
        - file_name: 文件名
    """
    # 从文件名提取层数
    file_name = os.path.basename(file_path)
    layers = extract_layers_from_filename(file_name)
    
    if layers is None:
        print(f"  警告: 无法从文件名提取层数: {file_name}")
        return None
    
    print(f"  处理文件: {file_name} (层数: {layers})")
    
    # 读取数据
    try:
        if GAUSSIAN_FIT_AVAILABLE:
            energy_index, event_counts = read_data(file_path)
        else:
            # 使用numpy读取数据
            data = np.loadtxt(file_path)
            
            # 确保数据有两列
            if data.ndim == 1:
                # 如果只有一列数据
                energy_index = np.arange(1, len(data) + 1)
                event_counts = data
            else:
                energy_index = data[:, 0]
                event_counts = data[:, 1]
    except Exception as e:
        print(f"    读取失败: {e}")
        return None
    
    # 计算总事件数
    if GAUSSIAN_FIT_AVAILABLE:
        total_events = calculate_total_events(event_counts)
    else:
        total_events = int(np.sum(event_counts))
    
    # 计算高斯峰参数
    if GAUSSIAN_FIT_AVAILABLE:
        # 使用高斯峰拟合模块
        mean_index, var_index = calculate_index_statistics(energy_index, event_counts)
        initial_std = np.sqrt(var_index)
        
        # 执行迭代高斯拟合
        peak_position, peak_std, peak_height, _ = iterative_gaussian_fit(
            energy_index, event_counts, initial_mean=mean_index, initial_std=initial_std
        )
    else:
        # 使用简化版本
        # 计算加权均值和方差
        total_events_sum = np.sum(event_counts)
        if total_events_sum == 0:
            mean_index = 0.0
            var_index = 0.0
        else:
            mean_index = np.sum(energy_index * event_counts) / total_events_sum
            var_index = np.sum(event_counts * (energy_index - mean_index) ** 2) / total_events_sum
        
        initial_std = np.sqrt(var_index)
        
        # 简化迭代高斯拟合
        current_mean = mean_index
        current_std = initial_std
        max_iterations = 50
        tolerance = 1e-6
        
        for iteration in range(max_iterations):
            # 选择3σ以内的数据
            lower_bound = current_mean - 3 * current_std
            upper_bound = current_mean + 3 * current_std
            
            mask = (energy_index >= lower_bound) & (energy_index <= upper_bound)
            
            if np.sum(mask) < 10:
                break
            
            selected_indices = energy_index[mask]
            selected_counts = event_counts[mask]
            
            total_selected_counts = np.sum(selected_counts)
            if total_selected_counts > 0:
                new_mean = np.sum(selected_indices * selected_counts) / total_selected_counts
                new_var = np.sum(selected_counts * (selected_indices - new_mean) ** 2) / total_selected_counts
                new_std = np.sqrt(new_var)
            else:
                new_mean = current_mean
                new_std = current_std
            
            # 检查收敛
            mean_change = abs(new_mean - current_mean)
            std_change = abs(new_std - current_std)
            
            current_mean = new_mean
            current_std = new_std
            
            if mean_change < tolerance and std_change < tolerance:
                break
        
        peak_position = current_mean
        peak_std = current_std
        peak_height = total_events / (peak_std * np.sqrt(2 * np.pi))
    
    # 计算高斯峰面积
    peak_area = calculate_gaussian_area(peak_height, peak_std)
    
    print(f"    总事件数: {total_events:,.0f}")
    print(f"    峰位置: {peak_position:.4f}")
    print(f"    峰标准差: {peak_std:.4f}")
    print(f"    峰高度: {peak_height:.2f}")
    print(f"    峰面积: {peak_area:.2f}")
    
    return {
        'layers': layers,
        'total_events': total_events,
        'peak_position': peak_position,
        'peak_std': peak_std,
        'peak_height': peak_height,
        'peak_area': peak_area,
        'file_name': file_name
    }


def process_element_folder(element_folder_path):
    """
    处理单个元素文件夹
    
    参数:
    element_folder_path: 元素文件夹路径
    
    返回:
    element_data: 该元素的数据，按层数分类的列表
    """
    # 提取元素名
    element_name = extract_element_name_from_folder(element_folder_path)
    print(f"处理元素: {element_name}")
    
    # 查找所有txt文件
    txt_files = find_txt_files_in_folder(element_folder_path)
    
    if not txt_files:
        print(f"  警告: 在文件夹中未找到任何txt文件: {element_folder_path}")
        return []
    
    print(f"  找到 {len(txt_files)} 个txt文件")
    
    # 处理每个文件
    all_results = []
    for file_path in txt_files:
        result = process_single_txt_file(file_path)
        if result is not None:
            all_results.append(result)
    
    # 按层数排序
    all_results.sort(key=lambda x: x['layers'])
    
    return all_results


def organize_data_by_element(all_element_data):
    """
    按元素整理数据
    
    参数:
    all_element_data: 所有元素的数据字典，键为元素名，值为该元素的数据列表
    
    返回:
    organized_data: 整理后的数据字典，键为元素名，值为二维数组
                    二维数组格式: [[层数, 总事件数, 高斯峰面积, 高斯峰位置], ...]
    """
    organized_data = {}
    
    for element_name, element_results in all_element_data.items():
        if not element_results:
            continue
        
        # 创建二维数组
        data_array = []
        for result in element_results:
            # 按照要求：第一个分量是衰减片层数、第二个是总事件数、第三个是高斯峰面积、第四个是高斯峰位置
            row = [
                result['layers'],
                result['total_events'],
                result['peak_area'],
                result['peak_position']
            ]
            data_array.append(row)
        
        organized_data[element_name] = data_array
    
    return organized_data


def print_organized_data(organized_data):
    """
    打印整理后的数据
    
    参数:
    organized_data: 整理后的数据字典
    """
    print("\n" + "=" * 80)
    print("整理后的数据（按元素分类）:")
    print("=" * 80)
    
    for element_name, data_array in organized_data.items():
        print(f"\n元素: {element_name}")
        print("-" * 60)
        print("层数\t总事件数\t峰面积\t\t峰位置")
        print("-" * 60)
        
        for row in data_array:
            layers, total_events, peak_area, peak_position = row
            print(f"{layers:4d}\t{total_events:10,.0f}\t{peak_area:10.2f}\t{peak_position:10.4f}")


def save_data_to_file(organized_data, output_file):
    """
    将数据保存到文件
    
    参数:
    organized_data: 整理后的数据字典
    output_file: 输出文件路径
    """
    import json
    
    # 将数据转换为可序列化的格式
    serializable_data = {}
    for element_name, data_array in organized_data.items():
        serializable_data[element_name] = data_array
    
    # 保存为JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n数据已保存到: {output_file}")


def main():
    """
    主函数
    """
    print("=" * 60)
    print("Absorb数据处理程序")
    print("=" * 60)
    
    # 1. 获取Absorb文件夹路径
    print("1. 获取Absorb文件夹路径...")
    absorb_path = get_absorb_folder_path()
    print(f"   Absorb路径: {absorb_path}")
    
    # 检查路径是否存在
    if not os.path.exists(absorb_path):
        print(f"错误: Absorb文件夹不存在 - {absorb_path}")
        print("请确保Absorb文件夹位于正确的位置")
        return
    
    # 2. 查找所有元素文件夹
    print("\n2. 查找Absorb文件夹中的元素文件夹...")
    element_folders = []
    
    for item in os.listdir(absorb_path):
        item_path = os.path.join(absorb_path, item)
        if os.path.isdir(item_path):
            element_folders.append(item_path)
    
    if not element_folders:
        print("  未找到任何元素文件夹")
        return
    
    print(f"  找到 {len(element_folders)} 个元素文件夹")
    for i, folder_path in enumerate(element_folders[:10]):  # 只显示前10个
        element_name = extract_element_name_from_folder(folder_path)
        print(f"  {i+1}. {element_name}")
    if len(element_folders) > 10:
        print(f"  ... 还有 {len(element_folders) - 10} 个文件夹")
    
    # 3. 处理每个元素文件夹
    print("\n3. 处理每个元素文件夹...")
    all_element_data = {}
    
    for i, folder_path in enumerate(element_folders):
        element_name = extract_element_name_from_folder(folder_path)
        print(f"\n[{i+1}/{len(element_folders)}] ", end="")
        
        element_results = process_element_folder(folder_path)
        if element_results:
            all_element_data[element_name] = element_results
    
    if not all_element_data:
        print("没有成功处理任何元素文件夹")
        return
    
    # 4. 整理数据
    print("\n4. 整理数据...")
    organized_data = organize_data_by_element(all_element_data)
    
    # 5. 打印整理后的数据
    print_organized_data(organized_data)
    
    # 6. 保存数据到文件
    print("\n5. 保存数据到文件...")
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "absorb_processed_data.json")
    save_data_to_file(organized_data, output_file)
    
    # 7. 返回数据供下一步使用
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    print("\n数据已准备好，等待下一步指示。")
    print(f"数据已保存到: {output_file}")
    print(f"数据结构: 字典，键为元素名，值为二维数组")
    print("二维数组格式: [层数, 总事件数, 高斯峰面积, 高斯峰位置]")
    
    return organized_data


if __name__ == "__main__":
    main()
