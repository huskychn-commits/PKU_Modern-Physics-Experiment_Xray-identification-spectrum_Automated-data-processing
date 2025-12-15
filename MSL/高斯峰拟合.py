#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高斯峰拟合脚本
功能：
1. 从文件路径提取元素名
2. 读取数据文件（第一列：能量index；第二列：事件数）
3. 计算总事件数
4. 绘制事件数的直方图
"""

import os
import numpy as np
import re
import sys

# 尝试导入matplotlib，如果失败则提供友好的错误信息
try:
    import matplotlib
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入matplotlib - {e}")
    print("提示: 请确保matplotlib已正确安装，或使用base环境运行脚本")
    print("      base环境已测试可用，py310环境可能存在兼容性问题")
    MATPLOTLIB_AVAILABLE = False
except Exception as e:
    print(f"警告: matplotlib导入时出现错误 - {e}")
    print("提示: 这可能是环境配置问题，建议使用base环境")
    MATPLOTLIB_AVAILABLE = False

def extract_element_name(file_path):
    """
    从文件路径中提取元素名
    
    参数:
    file_path: 文件路径，例如 'D:\\课程作业\\2025秋\\近物实验\\X射线标识谱\\数据\\MSL\\MSL_N4000000_Fe.txt'
    
    返回:
    元素名，例如 'Fe'
    """
    # 获取文件名（不含路径）
    file_name = os.path.basename(file_path)
    
    # 移除扩展名
    name_without_ext = os.path.splitext(file_name)[0]
    
    # 按_分割，取最后一个部分作为元素名
    # 例如：MSL_N4000000_Fe -> Fe
    parts = name_without_ext.split('_')
    if len(parts) > 0:
        return parts[-1]
    else:
        return "Unknown"

def read_data(file_path):
    """
    读取数据文件
    
    参数:
    file_path: 数据文件路径
    
    返回:
    energy_index: 能量index数组
    event_counts: 事件数数组
    """
    # 使用numpy读取数据
    data = np.loadtxt(file_path)
    
    # 确保数据有两列
    if data.ndim == 1:
        # 如果只有一列数据（不应该发生）
        energy_index = np.arange(1, len(data) + 1)
        event_counts = data
    else:
        energy_index = data[:, 0]
        event_counts = data[:, 1]
    
    return energy_index, event_counts

def calculate_total_events(event_counts):
    """
    计算总事件数
    
    参数:
    event_counts: 事件数数组
    
    返回:
    总事件数（整数）
    """
    # 使用int()确保返回整数
    return int(np.sum(event_counts))

def calculate_index_statistics(energy_index, event_counts):
    """
    计算index的加权统计量（均值和方差）
    认为index是一个线性的物理量，数据是频数
    
    参数:
    energy_index: 能量index数组
    event_counts: 事件数数组（作为权重）
    
    返回:
    mean_index: 加权均值
    var_index: 加权方差
    """
    # 总事件数
    total_events = np.sum(event_counts)
    
    if total_events == 0:
        return 0.0, 0.0
    
    # 加权均值: μ = Σ(x_i * w_i) / Σw_i
    mean_index = np.sum(energy_index * event_counts) / total_events
    
    # 加权方差: σ² = Σ(w_i * (x_i - μ)²) / Σw_i
    var_index = np.sum(event_counts * (energy_index - mean_index) ** 2) / total_events
    
    return mean_index, var_index

def iterative_gaussian_fit(energy_index, event_counts, initial_mean=None, initial_std=None, max_iterations=50, tolerance=1e-6):
    """
    迭代高斯拟合函数，只使用3σ以内的数据
    
    参数:
    energy_index: 能量index数组
    event_counts: 事件数数组
    initial_mean: 初始均值估计（如果为None，则使用加权均值）
    initial_std: 初始标准差估计（如果为None，则使用加权标准差）
    max_iterations: 最大迭代次数
    tolerance: 收敛容忍度
    
    返回:
    optimized_mean: 优化后的均值
    optimized_std: 优化后的标准差
    optimized_height: 优化后的峰高
    loss_history: 损失函数历史
    """
    # 初始参数估计
    if initial_mean is None or initial_std is None:
        initial_mean, initial_var = calculate_index_statistics(energy_index, event_counts)
        initial_std = np.sqrt(initial_var)
    
    current_mean = initial_mean
    current_std = initial_std
    
    loss_history = []
    
    print(f"   开始迭代拟合: 初始均值={current_mean:.4f}, 初始标准差={current_std:.4f}")
    
    for iteration in range(max_iterations):
        # 1. 选择3σ以内的数据
        lower_bound = current_mean - 3 * current_std
        upper_bound = current_mean + 3 * current_std
        
        # 创建掩码选择3σ以内的数据点
        mask = (energy_index >= lower_bound) & (energy_index <= upper_bound)
        
        if np.sum(mask) < 10:  # 如果数据点太少，停止迭代
            print(f"   警告: 第{iteration+1}次迭代，3σ内数据点不足({np.sum(mask)}个)")
            break
        
        selected_indices = energy_index[mask]
        selected_counts = event_counts[mask]
        
        # 2. 计算当前参数下的高斯分布预测值
        # 高斯分布公式: f(x) = A * exp(-(x-μ)²/(2σ²))
        # 其中A是归一化常数（峰高）
        total_selected_counts = np.sum(selected_counts)
        predicted = (total_selected_counts / (current_std * np.sqrt(2 * np.pi))) * \
                   np.exp(-0.5 * ((selected_indices - current_mean) / current_std) ** 2)
        
        # 3. 计算损失函数（差平方和）
        loss = np.sum((selected_counts - predicted) ** 2)
        loss_history.append(loss)
        
        # 4. 使用选中的数据重新计算均值和标准差
        if total_selected_counts > 0:
            new_mean = np.sum(selected_indices * selected_counts) / total_selected_counts
            new_var = np.sum(selected_counts * (selected_indices - new_mean) ** 2) / total_selected_counts
            new_std = np.sqrt(new_var)
        else:
            new_mean = current_mean
            new_std = current_std
        
        # 5. 检查收敛
        mean_change = abs(new_mean - current_mean)
        std_change = abs(new_std - current_std)
        
        if iteration > 0:
            loss_change = abs(loss_history[-1] - loss_history[-2]) / loss_history[-2]
        else:
            loss_change = float('inf')
        
        # 更新参数
        current_mean = new_mean
        current_std = new_std
        
        # 打印迭代信息
        if (iteration + 1) % 10 == 0 or iteration == 0 or iteration == max_iterations - 1:
            print(f"   迭代 {iteration+1}: μ={current_mean:.4f}, σ={current_std:.4f}, "
                  f"损失={loss:.2f}, 3σ内数据点={np.sum(mask)}个")
        
        # 检查收敛条件
        if mean_change < tolerance and std_change < tolerance and (iteration > 0 and loss_change < tolerance):
            print(f"   在第{iteration+1}次迭代收敛")
            break
    
    # 计算最终的峰高：A = N / (σ * √(2π))
    total_events = np.sum(event_counts)
    optimized_height = total_events / (current_std * np.sqrt(2 * np.pi))
    
    print(f"   迭代拟合完成: 最终均值={current_mean:.4f}, 最终标准差={current_std:.4f}, 峰高={optimized_height:.2f}")
    
    return current_mean, current_std, optimized_height, loss_history

def plot_histogram(energy_index, event_counts, element_name, total_events, mean_index, var_index, 
                   optimized_mean=None, optimized_std=None):
    """
    绘制直方图和高斯分布线
    
    参数:
    energy_index: 能量index数组
    event_counts: 事件数数组
    element_name: 元素名
    total_events: 总事件数
    mean_index: index的加权均值
    var_index: index的加权方差
    optimized_mean: 迭代优化后的均值（如果为None则不绘制）
    optimized_std: 迭代优化后的标准差（如果为None则不绘制）
    """
    if not MATPLOTLIB_AVAILABLE:
        print("   matplotlib不可用，跳过绘图")
        print("   建议: 使用base环境运行脚本以获得完整功能")
        print("         base环境命令: conda activate base")
        return
    
    try:
        # 设置中文字体
        try:
            # 尝试使用系统字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
        except:
            pass
        
        plt.figure(figsize=(12, 6))
        
        # 绘制直方图（无边框，更简洁）
        plt.bar(energy_index, event_counts, width=1.0, alpha=0.7, color='steelblue', edgecolor=None, label='实验数据')
        
        # 计算直接统计的高斯分布
        std_index = np.sqrt(var_index)
        
        # 创建更密集的x轴点用于绘制平滑的高斯曲线
        x_fine = np.linspace(min(energy_index) - 20, max(energy_index) + 20, 1000)
        
        # 直接计算的高斯分布公式: f(x) = A * exp(-(x-μ)²/(2σ²))
        gaussian_direct = (total_events / (std_index * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_fine - mean_index) / std_index) ** 2)
        
        # 绘制直接计算的高斯分布线（灰色，细线条，透明度50%）
        plt.plot(x_fine, gaussian_direct, color='gray', linewidth=1.5, alpha=0.5, 
                label=f'直接计算的高斯分布\nμ={mean_index:.2f}, σ={std_index:.2f}')
        
        # 如果提供了优化后的参数，绘制迭代拟合的高斯分布（红线）
        if optimized_mean is not None and optimized_std is not None:
            # 计算迭代拟合的高斯分布
            gaussian_optimized = (total_events / (optimized_std * np.sqrt(2 * np.pi))) * \
                                np.exp(-0.5 * ((x_fine - optimized_mean) / optimized_std) ** 2)
            
            # 绘制迭代拟合的高斯分布线（红色，细线条，透明度70%）
            plt.plot(x_fine, gaussian_optimized, color='red', linewidth=2.0, alpha=0.7, 
                    label=f'迭代拟合的高斯分布\nμ={optimized_mean:.2f}, σ={optimized_std:.2f}')
            
            # 在图上标记3σ范围
            lower_bound = optimized_mean - 3 * optimized_std
            upper_bound = optimized_mean + 3 * optimized_std
            plt.axvline(x=lower_bound, color='orange', linestyle='--', alpha=0.5, linewidth=1.0, 
                       label=f'3σ范围: [{lower_bound:.1f}, {upper_bound:.1f}]')
            plt.axvline(x=upper_bound, color='orange', linestyle='--', alpha=0.5, linewidth=1.0)
            
            # 更新统计信息文本
            stats_text = f'总事件数: {total_events:,}\n直接计算:\n  均值 μ: {mean_index:.2f}\n  标准差 σ: {std_index:.2f}\n迭代拟合:\n  均值 μ: {optimized_mean:.2f}\n  标准差 σ: {optimized_std:.2f}'
        else:
            stats_text = f'总事件数: {total_events:,}\n均值 μ: {mean_index:.2f}\n标准差 σ: {std_index:.2f}\n方差: {var_index:.2f}'
        
        # 设置图表标题和标签
        plt.title(f'{element_name} 元素 - 事件数分布与高斯拟合 (总事件数: {total_events:,})', fontsize=14, fontweight='bold')
        plt.xlabel('能量 Index', fontsize=12)
        plt.ylabel('事件数', fontsize=12)
        
        # 添加网格
        plt.grid(True, alpha=0.3, linestyle='--')
        
        # 自动调整x轴范围
        plt.xlim(min(energy_index) - 5, max(energy_index) + 5)
        
        # 添加图例
        plt.legend(fontsize=10, loc='upper right')
        
        # 添加统计信息文本
        plt.text(0.02, 0.98, stats_text, 
                 transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 调整布局
        plt.tight_layout()
        
        # 显示图表
        plt.show()
        print("   绘图完成! (包含高斯分布线)")
    except Exception as e:
        print(f"   绘图失败: {e}")
        print("   提示: 这可能是matplotlib配置问题")
        print("         建议使用base环境: conda activate base")

def process_file(file_path):
    """
    处理单个文件
    
    参数:
    file_path: 数据文件路径
    """
    print(f"\n处理文件: {file_path}")
    print("-" * 40)
    
    # 1. 从文件路径提取元素名
    element_name = extract_element_name(file_path)
    print(f"1. 从文件路径提取元素名: {element_name}")
    
    # 2. 读取数据
    print(f"2. 读取数据文件")
    try:
        energy_index, event_counts = read_data(file_path)
        print(f"   读取成功: {len(energy_index)} 个数据点")
        
        # 显示前5个数据点
        print(f"   前5个数据点:")
        for i in range(min(5, len(energy_index))):
            print(f"     Index: {energy_index[i]:.0f}, 事件数: {event_counts[i]:.0f}")
    except Exception as e:
        print(f"   读取失败: {e}")
        return None, None, None, None
    
    # 3. 计算总事件数
    total_events = calculate_total_events(event_counts)
    print(f"3. 计算总事件数: {total_events:,.0f}")
    
    return energy_index, event_counts, element_name, total_events

def main():
    """
    主函数
    """
    print("=" * 60)
    print("高斯峰拟合脚本 - 基础功能")
    print("=" * 60)
    
    # 显示当前环境信息
    print(f"Python可执行文件: {sys.executable}")
    print(f"Matplotlib可用: {MATPLOTLIB_AVAILABLE}")
    
    # 检查是否有命令行参数
    if len(sys.argv) > 1:
        # 使用命令行参数指定的文件
        file_paths = sys.argv[1:]
    else:
        # 使用默认文件路径（相对路径）
        # 脚本位于: 数据/数据处理/MSL/高斯峰拟合.py
        # 数据文件位于: 数据/MSL/
        # 相对路径: ../../MSL/
        file_path = '../../MSL/MSL_N4000000_Fe.txt'
        file_paths = [file_path]
    
    all_results = []
    
    for file_path in file_paths:
        print(f"\n处理文件: {file_path}")
        print("-" * 40)
        
        # 1. 从文件路径提取元素名
        element_name = extract_element_name(file_path)
        print(f"1. 从文件路径提取元素名: {element_name}")
        
        # 2. 读取数据
        print(f"2. 读取数据文件")
        try:
            energy_index, event_counts = read_data(file_path)
            print(f"   读取成功: {len(energy_index)} 个数据点")
            
            # 显示前5个数据点
            print(f"   前5个数据点:")
            for i in range(min(5, len(energy_index))):
                print(f"     Index: {energy_index[i]:.0f}, 事件数: {event_counts[i]:.0f}")
        except Exception as e:
            print(f"   读取失败: {e}")
            # 尝试使用正确的相对路径
            print("   尝试使用正确的相对路径...")
            # 从示例路径提取文件名
            file_name = os.path.basename(file_path)
            # 脚本位于: 数据/数据处理/MSL/高斯峰拟合.py
            # 数据文件位于: 数据/MSL/
            # 相对路径: ../../MSL/
            relative_path = f'../../MSL/{file_name}'
            try:
                energy_index, event_counts = read_data(relative_path)
                print(f"   使用相对路径读取成功: {len(energy_index)} 个数据点")
            except Exception as e2:
                print(f"   相对路径也失败: {e2}")
                continue
        
        # 3. 计算总事件数
        total_events = calculate_total_events(event_counts)
        print(f"3. 计算总事件数: {total_events:,.0f}")
        
        # 4. 计算index的统计量（加权均值和方差）
        mean_index, var_index = calculate_index_statistics(energy_index, event_counts)
        std_index = np.sqrt(var_index)  # 标准差
        print(f"4. Index统计量:")
        print(f"   加权均值: {mean_index:.4f}")
        print(f"   加权方差: {var_index:.4f}")
        print(f"   标准差: {std_index:.4f}")
        
        # 5. 执行迭代高斯拟合（只使用3σ以内的数据）
        print(f"5. 执行迭代高斯拟合...")
        optimized_mean, optimized_std, optimized_height, loss_history = iterative_gaussian_fit(
            energy_index, event_counts, initial_mean=mean_index, initial_std=std_index
        )
        
        # 计算优化后的方差
        optimized_var = optimized_std ** 2
        
        all_results.append((file_path, element_name, total_events, mean_index, var_index, optimized_mean, optimized_var, optimized_height))
        
        # 6. 绘制直方图（包含直接计算和迭代拟合的高斯分布）
        print(f"6. 绘制 {element_name} 元素的直方图...")
        plot_histogram(energy_index, event_counts, element_name, total_events, mean_index, var_index, 
                      optimized_mean, optimized_std)
    
    # 显示所有处理结果摘要
    if all_results:
        print("\n" + "=" * 60)
        print("处理结果摘要:")
        print("=" * 60)
        for result in all_results:
            if len(result) == 8:  # 包含优化后的参数和峰高
                file_path, element_name, total_events, mean_index, var_index, optimized_mean, optimized_var, optimized_height = result
                optimized_std = np.sqrt(optimized_var)
                # 计算直接计算的峰高：A = N / (σ * √(2π))
                direct_height = total_events / (np.sqrt(var_index) * np.sqrt(2 * np.pi))
                print(f"文件: {os.path.basename(file_path)}")
                print(f"元素: {element_name}")
                print(f"总事件数: {total_events:,.0f}")
                print(f"直接计算 - 均值: {mean_index:.4f}, 方差: {var_index:.4f}, 标准差: {np.sqrt(var_index):.4f}, 峰高: {direct_height:.2f}")
                print(f"迭代拟合 - 均值: {optimized_mean:.4f}, 方差: {optimized_var:.4f}, 标准差: {optimized_std:.4f}, 峰高: {optimized_height:.2f}")
                print(f"参数变化 - Δμ: {optimized_mean - mean_index:.4f}, Δσ: {optimized_std - np.sqrt(var_index):.4f}, Δ峰高: {optimized_height - direct_height:.2f}")
            else:
                file_path, element_name, total_events, mean_index, var_index = result
                print(f"文件: {os.path.basename(file_path)}")
                print(f"元素: {element_name}")
                print(f"总事件数: {total_events:,.0f}")
                print(f"Index加权均值: {mean_index:.4f}")
                print(f"Index加权方差: {var_index:.4f}")
                print(f"Index标准差: {np.sqrt(var_index):.4f}")
            print("-" * 40)
    
    print("\n" + "=" * 60)
    print("完成！高斯峰的参数拟合完毕。")
    print("=" * 60)
    
    # 使用说明
    if len(sys.argv) == 1:
        print("\n使用说明:")
        print("可以通过命令行参数指定文件: python 高斯峰拟合.py 文件路径")
        print("示例: python 高斯峰拟合.py ..\\MSL\\MSL_N4000000_Ag.txt")

if __name__ == "__main__":
    main()
