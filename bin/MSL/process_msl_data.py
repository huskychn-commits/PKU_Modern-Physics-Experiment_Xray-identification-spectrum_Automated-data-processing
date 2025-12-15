#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理MSL文件夹数据的程序
功能：
1. 通过读自己的路径，生成MSL文件夹的路径
2. 找到MSL文件夹中所有.txt文件
3. 读取每个文件并提取元素名
4. 调用高斯峰拟合功能进行峰值拟合
5. 输出表格：元素名、拟合得到的峰位置和峰宽
"""

import os
import sys
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt

# 添加当前目录到路径，以便导入高斯峰拟合模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入高斯峰拟合中的函数
try:
    from 高斯峰拟合 import (
        extract_element_name,
        read_data,
        calculate_total_events,
        calculate_index_statistics,
        iterative_gaussian_fit
    )
    GAUSSIAN_FIT_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入高斯峰拟合模块 - {e}")
    print("将使用本地实现的简化版本")
    GAUSSIAN_FIT_AVAILABLE = False

# 尝试导入xray_element_tool以获取原子序数
try:
    from xray_element_tool import ElementDatabase
    ELEMENT_DB_AVAILABLE = True
except ImportError:
    print("警告: 无法导入xray_element_tool模块，将使用默认原子序数映射")
    ELEMENT_DB_AVAILABLE = False


def get_msl_folder_path():
    """
    通过读自己的路径，生成MSL文件夹的路径
    
    返回:
    msl_path: MSL文件夹的完整路径
    """
    # 获取当前脚本的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建MSL文件夹路径
    # 当前目录: D:\课程作业\2025秋\近物实验\X射线标识谱\数据\数据处理\bin\MSL
    # MSL数据目录: D:\课程作业\2025秋\近物实验\X射线标识谱\数据\MSL
    # 需要向上三级到数据目录，然后进入MSL
    data_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    msl_path = os.path.join(data_dir, "MSL")
    
    return msl_path


def find_txt_files(msl_path):
    """
    找到MSL文件夹中所有.txt文件
    
    参数:
    msl_path: MSL文件夹路径
    
    返回:
    txt_files: 所有.txt文件的完整路径列表
    """
    # 使用glob查找所有.txt文件
    pattern = os.path.join(msl_path, "*.txt")
    txt_files = glob.glob(pattern)
    
    # 过滤掉可能的非数据文件（如err.out, log.out等）
    filtered_files = []
    for file_path in txt_files:
        file_name = os.path.basename(file_path)
        # 只保留以MSL_开头的.txt文件
        if file_name.startswith("MSL_"):
            filtered_files.append(file_path)
    
    return sorted(filtered_files)


def extract_element_name_from_path(file_path):
    """
    从文件路径中提取元素名（简化版本，如果无法导入高斯峰拟合模块则使用）
    
    参数:
    file_path: 文件路径
    
    返回:
    元素名
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


def calculate_index_statistics_simple(energy_index, event_counts):
    """
    计算index的加权统计量（均值和方差）- 简化版本
    
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


def iterative_gaussian_fit_simple(energy_index, event_counts, initial_mean=None, initial_std=None, max_iterations=50, tolerance=1e-6):
    """
    迭代高斯拟合函数，只使用3σ以内的数据 - 简化版本
    
    参数:
    energy_index: 能量index数组
    event_counts: 事件数数组
    initial_mean: 初始均值估计
    initial_std: 初始标准差估计
    max_iterations: 最大迭代次数
    tolerance: 收敛容忍度
    
    返回:
    optimized_mean: 优化后的均值
    optimized_std: 优化后的标准差
    """
    # 初始参数估计
    if initial_mean is None or initial_std is None:
        initial_mean, initial_var = calculate_index_statistics_simple(energy_index, event_counts)
        initial_std = np.sqrt(initial_var)
    
    current_mean = initial_mean
    current_std = initial_std
    
    for iteration in range(max_iterations):
        # 1. 选择3σ以内的数据
        lower_bound = current_mean - 3 * current_std
        upper_bound = current_mean + 3 * current_std
        
        # 创建掩码选择3σ以内的数据点
        mask = (energy_index >= lower_bound) & (energy_index <= upper_bound)
        
        if np.sum(mask) < 10:  # 如果数据点太少，停止迭代
            break
        
        selected_indices = energy_index[mask]
        selected_counts = event_counts[mask]
        
        # 2. 使用选中的数据重新计算均值和标准差
        total_selected_counts = np.sum(selected_counts)
        if total_selected_counts > 0:
            new_mean = np.sum(selected_indices * selected_counts) / total_selected_counts
            new_var = np.sum(selected_counts * (selected_indices - new_mean) ** 2) / total_selected_counts
            new_std = np.sqrt(new_var)
        else:
            new_mean = current_mean
            new_std = current_std
        
        # 3. 检查收敛
        mean_change = abs(new_mean - current_mean)
        std_change = abs(new_std - current_std)
        
        # 更新参数
        current_mean = new_mean
        current_std = new_std
        
        # 检查收敛条件
        if mean_change < tolerance and std_change < tolerance:
            break
    
    return current_mean, current_std


def process_single_file(file_path, use_gaussian_module=True):
    """
    处理单个文件，进行峰值拟合
    
    参数:
    file_path: 数据文件路径
    use_gaussian_module: 是否使用高斯峰拟合模块
    
    返回:
    result: 包含元素名、峰位置、峰宽的字典
    """
    print(f"处理文件: {os.path.basename(file_path)}")
    
    # 提取元素名
    if use_gaussian_module and GAUSSIAN_FIT_AVAILABLE:
        element_name = extract_element_name(file_path)
    else:
        element_name = extract_element_name_from_path(file_path)
    
    # 读取数据
    try:
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
        print(f"  读取失败: {e}")
        return None
    
    # 计算总事件数
    total_events = np.sum(event_counts)
    
    # 计算初始统计量
    if use_gaussian_module and GAUSSIAN_FIT_AVAILABLE:
        mean_index, var_index = calculate_index_statistics(energy_index, event_counts)
        initial_std = np.sqrt(var_index)
        
        # 执行迭代高斯拟合（现在返回4个值：均值、标准差、峰高、损失历史）
        optimized_mean, optimized_std, optimized_height, _ = iterative_gaussian_fit(
            energy_index, event_counts, initial_mean=mean_index, initial_std=initial_std
        )
    else:
        mean_index, var_index = calculate_index_statistics_simple(energy_index, event_counts)
        initial_std = np.sqrt(var_index)
        
        # 执行迭代高斯拟合（简化版本）
        optimized_mean, optimized_std = iterative_gaussian_fit_simple(
            energy_index, event_counts, initial_mean=mean_index, initial_std=initial_std
        )
        # 计算峰高：A = N / (σ * √(2π))
        optimized_height = total_events / (optimized_std * np.sqrt(2 * np.pi))
    
    # 计算峰高（用于显示，但不保存到CSV）
    peak_height = total_events / (optimized_std * np.sqrt(2 * np.pi))
    
    print(f"  拟合峰高: {peak_height:.2f}")
    
    # 峰宽通常用FWHM（半高全宽）表示，FWHM = 2.355 * σ
    fwhm = 2.355 * optimized_std
    
    print(f"  元素: {element_name}")
    print(f"  总事件数: {total_events:,.0f}")
    print(f"  拟合峰位置: {optimized_mean:.4f}")
    print(f"  拟合标准差: {optimized_std:.4f}")
    print(f"  拟合FWHM: {fwhm:.4f}")
    print("-" * 40)
    
    return {
        'element': element_name,
        'peak_position': optimized_mean,
        'peak_std': optimized_std,
        'peak_fwhm': fwhm,
        'total_events': total_events,
        'file_name': os.path.basename(file_path)
    }


def get_atomic_number(element_symbol):
    """
    获取元素的原子序数
    
    参数:
    element_symbol: 元素符号（如'Fe', 'Cu'等）
    
    返回:
    atomic_number: 原子序数，如果找不到则返回None
    """
    if ELEMENT_DB_AVAILABLE:
        try:
            db = ElementDatabase()
            element_info = db.get_element_info(element_symbol)
            if element_info:
                return element_info['atomic_number']
        except:
            pass
    
    # 如果无法从数据库获取，使用硬编码的映射
    element_to_z = {
        'Ag': 47, 'Co': 27, 'Cu': 29, 'Fe': 26, 'Mo': 42,
        'Ni': 28, 'Se': 34, 'Sr': 38, 'Ti': 22, 'Zn': 30, 'Zr': 40
    }
    
    return element_to_z.get(element_symbol)


def test_plot(results):
    """
    绘制peak_position随原子序数变化的图像
    
    参数:
    results: 处理结果列表，每个元素是包含'element', 'peak_position', 'peak_std'的字典
    """
    print("\n6. 绘制peak_position随原子序数变化的图像...")
    
    # 准备数据
    atomic_numbers = []
    peak_positions = []
    peak_stds = []
    element_symbols = []
    
    for result in results:
        element_symbol = result['element']
        atomic_number = get_atomic_number(element_symbol)
        
        if atomic_number is not None:
            atomic_numbers.append(atomic_number)
            peak_positions.append(result['peak_position'])
            peak_stds.append(result['peak_std'])
            element_symbols.append(element_symbol)
        else:
            print(f"  警告: 无法获取元素 {element_symbol} 的原子序数，跳过")
    
    if not atomic_numbers:
        print("  错误: 没有有效的原子序数数据，无法绘图")
        return
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 设置中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass
    
    # 绘制误差横线（E+σ和E-σ）- 使用灰色
    for z, peak, std in zip(atomic_numbers, peak_positions, peak_stds):
        # 绘制横线表示E±σ范围
        plt.hlines(y=peak, xmin=z-0.3, xmax=z+0.3, color='red', linewidth=1.5, alpha=0.7)
        plt.hlines(y=peak+std, xmin=z-0.2, xmax=z+0.2, color='gray', linewidth=1, alpha=0.6, linestyle='--')
        plt.hlines(y=peak-std, xmin=z-0.2, xmax=z+0.2, color='gray', linewidth=1, alpha=0.6, linestyle='--')
        
        # 绘制垂直线连接E±σ
        plt.vlines(x=z, ymin=peak-std, ymax=peak+std, color='gray', linewidth=1, alpha=0.6)
    
    # 添加元素符号标签
    for z, peak, symbol in zip(atomic_numbers, peak_positions, element_symbols):
        plt.text(z, peak + 5, symbol, fontsize=10, ha='center', va='bottom', alpha=0.8)
    
    # 设置坐标轴标签
    plt.xlabel('原子序数 (Z)', fontsize=14)
    plt.ylabel('峰位置 (能量Index)', fontsize=14)
    
    # 设置标题
    plt.title('MSL数据：峰位置随原子序数变化', fontsize=16, fontweight='bold')
    
    # 添加网格
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', label='峰位置',
               linewidth=1.5, alpha=0.7),
        Line2D([0], [0], color='gray', linestyle='--', label='E±σ范围',
               linewidth=1, alpha=0.6)
    ]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=12)
    
    # 设置坐标轴范围
    plt.xlim(min(atomic_numbers) - 2, max(atomic_numbers) + 2)
    plt.ylim(min(peak_positions) - 20, max(peak_positions) + 30)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    output_image = os.path.join(os.path.dirname(os.path.abspath(__file__)), "msl_peak_vs_atomic_number.png")
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"  图像已保存到: {output_image}")
    
    # 显示图像
    print("  正在显示图像...")
    plt.show()


def moseley_plot(results):
    """
    绘制莫塞莱定律图像：√E vs Z，并进行线性拟合
    
    根据莫塞莱定律：E ∝ (Z-σ)²，所以√E ∝ (Z-σ)
    绘制√E vs Z的散点图，进行线性拟合：√E = aZ + b
    Z轴截距（当√E=0时）为：Z = -b/a = σ
    
    参数:
    results: 处理结果列表，每个元素是包含'element', 'peak_position', 'peak_std'的字典
    """
    print("\n7. 绘制莫塞莱定律图像：√E vs Z...")
    
    # 准备数据
    atomic_numbers = []
    peak_positions = []
    peak_stds = []
    element_symbols = []
    
    for result in results:
        element_symbol = result['element']
        atomic_number = get_atomic_number(element_symbol)
        
        if atomic_number is not None:
            atomic_numbers.append(atomic_number)
            peak_positions.append(result['peak_position'])
            peak_stds.append(result['peak_std'])
            element_symbols.append(element_symbol)
        else:
            print(f"  警告: 无法获取元素 {element_symbol} 的原子序数，跳过")
    
    if not atomic_numbers:
        print("  错误: 没有有效的原子序数数据，无法绘图")
        return
    
    # 计算√E（注意：这里E是峰位置，作为相对能量）
    sqrt_E = np.sqrt(peak_positions)
    
    # 计算√E的误差传播：Δ(√E) = (1/(2√E)) * ΔE
    sqrt_E_errors = [std / (2 * np.sqrt(E)) if E > 0 else 0 for E, std in zip(peak_positions, peak_stds)]
    
    # 线性拟合：√E = aZ + b
    Z_array = np.array(atomic_numbers)
    sqrt_E_array = np.array(sqrt_E)
    
    # 使用numpy的polyfit进行线性拟合（1次多项式）
    coefficients = np.polyfit(Z_array, sqrt_E_array, 1)
    a, b = coefficients  # a是斜率，b是截距
    
    # 计算拟合线
    Z_fit = np.linspace(min(Z_array) - 2, max(Z_array) + 2, 100)
    sqrt_E_fit = a * Z_fit + b
    
    # 计算屏蔽常数σ：当√E=0时，Z = -b/a = σ
    sigma = -b / a
    
    # 计算拟合优度R²
    residuals = sqrt_E_array - (a * Z_array + b)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((sqrt_E_array - np.mean(sqrt_E_array))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    print(f"  线性拟合结果: √E = {a:.4f}Z + {b:.4f}")
    print(f"  拟合优度 1 - R² = {1-r_squared:.2e}")
    print(f"  屏蔽常数 σ = {sigma:.4f}")
    print(f"  莫塞莱定律: E ∝ (Z - {sigma:.2f})²")
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 设置中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass
    
    # 绘制数据点（按照E-Z关系的样式：红色横线标记峰位置，灰色误差棒）
    # 首先绘制误差横线（E+σ和E-σ）- 使用灰色
    for z, sqrt_e, sqrt_err in zip(Z_array, sqrt_E_array, sqrt_E_errors):
        # 绘制横线表示√E±σ范围
        plt.hlines(y=sqrt_e, xmin=z-0.3, xmax=z+0.3, color='red', linewidth=1.5, alpha=0.7)
        plt.hlines(y=sqrt_e+sqrt_err, xmin=z-0.2, xmax=z+0.2, color='gray', linewidth=1, alpha=0.6, linestyle='--')
        plt.hlines(y=sqrt_e-sqrt_err, xmin=z-0.2, xmax=z+0.2, color='gray', linewidth=1, alpha=0.6, linestyle='--')
        
        # 绘制垂直线连接√E±σ
        plt.vlines(x=z, ymin=sqrt_e-sqrt_err, ymax=sqrt_e+sqrt_err, color='gray', linewidth=1, alpha=0.6)
    
    # 绘制拟合线（绿色虚线）
    plt.plot(Z_fit, sqrt_E_fit, 'g--', linewidth=1.5, alpha=0.5, label=f'线性拟合: √E = {a:.3f}Z + {b:.3f}', zorder=4)
    
    # 标记Z轴截距（σ点）
    plt.axvline(x=sigma, color='blue', linestyle=':', linewidth=2, alpha=0.7, 
                label=f'屏蔽常数 σ = {sigma:.2f}')
    
    # 添加元素符号标签
    for z, sqrt_e, symbol in zip(Z_array, sqrt_E_array, element_symbols):
        plt.text(z, sqrt_e + 0.2, symbol, fontsize=10, ha='center', va='bottom', alpha=0.8)
    
    # 设置坐标轴标签
    plt.xlabel('原子序数 (Z)', fontsize=14)
    plt.ylabel('√E (√能量Index)', fontsize=14)
    
    # 设置标题
    plt.title('莫塞莱定律：√E vs Z 线性拟合', fontsize=16, fontweight='bold')
    
    # 添加网格
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', label='√E 位置',
               linewidth=1.5, alpha=0.7),
        Line2D([0], [0], color='gray', linestyle='--', label='√E±σ范围',
               linewidth=1, alpha=0.6),
        Line2D([0], [0], color='green', linestyle='--', label=f'线性拟合: √E = {a:.3f}Z + {b:.3f}',
               linewidth=1.5, alpha=0.5),
    ]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=11)
    
    # 设置坐标轴范围
    plt.xlim(min(Z_array) - 2, max(Z_array) + 2)
    plt.ylim(min(sqrt_E_array) - 1, max(sqrt_E_array) + 1)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    output_image = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moseley_law_fit.png")
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"  图像已保存到: {output_image}")
    
    # 显示图像
    print("  正在显示图像...")
    plt.show()
    
    return sigma, r_squared


def generate_figures(output_dir=None, image_name=None):
    """
    生成所有图片并保存到指定目录
    
    参数:
    output_dir: 输出目录，如果为None则保存到脚本所在目录
    image_name: 图片名称，如果为None则生成所有图片
    """
    print("=" * 60)
    print("MSL数据处理程序 - 生成图片")
    print("=" * 60)
    
    # 1. 获取MSL文件夹路径
    print("1. 获取MSL文件夹路径...")
    msl_path = get_msl_folder_path()
    print(f"   MSL路径: {msl_path}")
    
    # 检查路径是否存在
    if not os.path.exists(msl_path):
        print(f"错误: MSL文件夹不存在 - {msl_path}")
        print("请确保MSL文件夹位于正确的位置")
        return []
    
    # 2. 查找所有.txt文件
    print("\n2. 查找MSL文件夹中的.txt文件...")
    txt_files = find_txt_files(msl_path)
    
    if not txt_files:
        print("  未找到任何.txt文件")
        return []
    
    print(f"  找到 {len(txt_files)} 个.txt文件")
    
    # 3. 处理每个文件
    print("\n3. 处理文件并进行峰值拟合...")
    results = []
    
    for i, file_path in enumerate(txt_files):
        print(f"\n[{i+1}/{len(txt_files)}] ", end="")
        result = process_single_file(file_path, use_gaussian_module=True)
        if result:
            results.append(result)
    
    if not results:
        print("没有成功处理任何文件")
        return []
    
    # 4. 绘制图像并保存
    generated_images = []
    
    # 根据image_name参数决定生成哪个图片
    if image_name is None or "谱线与原子序数关系图" in image_name:
        print("\n4. 绘制peak_position随原子序数变化的图像...")
        atomic_numbers = []
        peak_positions = []
        peak_stds = []
        element_symbols = []
        
        for result in results:
            element_symbol = result['element']
            atomic_number = get_atomic_number(element_symbol)
            
            if atomic_number is not None:
                atomic_numbers.append(atomic_number)
                peak_positions.append(result['peak_position'])
                peak_stds.append(result['peak_std'])
                element_symbols.append(element_symbol)
            else:
                print(f"  警告: 无法获取元素 {element_symbol} 的原子序数，跳过")
        
        if atomic_numbers:
            # 创建图形
            plt.figure(figsize=(12, 8))
            
            # 设置中文字体
            try:
                plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
            except:
                pass
            
            # 绘制误差横线
            for z, peak, std in zip(atomic_numbers, peak_positions, peak_stds):
                plt.hlines(y=peak, xmin=z-0.3, xmax=z+0.3, color='red', linewidth=1.5, alpha=0.7)
                plt.hlines(y=peak+std, xmin=z-0.2, xmax=z+0.2, color='gray', linewidth=1, alpha=0.6, linestyle='--')
                plt.hlines(y=peak-std, xmin=z-0.2, xmax=z+0.2, color='gray', linewidth=1, alpha=0.6, linestyle='--')
                plt.vlines(x=z, ymin=peak-std, ymax=peak+std, color='gray', linewidth=1, alpha=0.6)
            
            # 添加元素符号标签
            for z, peak, symbol in zip(atomic_numbers, peak_positions, element_symbols):
                plt.text(z, peak + 5, symbol, fontsize=10, ha='center', va='bottom', alpha=0.8)
            
            # 设置坐标轴标签
            plt.xlabel('原子序数 (Z)', fontsize=14)
            plt.ylabel('峰位置 (能量Index)', fontsize=14)
            plt.title('MSL数据：峰位置随原子序数变化', fontsize=16, fontweight='bold')
            plt.grid(True, alpha=0.3, linestyle='--')
            
            # 添加图例
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='red', label='峰位置', linewidth=1.5, alpha=0.7),
                Line2D([0], [0], color='gray', linestyle='--', label='E±σ范围', linewidth=1, alpha=0.6)
            ]
            plt.legend(handles=legend_elements, loc='upper left', fontsize=12)
            
            # 设置坐标轴范围
            plt.xlim(min(atomic_numbers) - 2, max(atomic_numbers) + 2)
            plt.ylim(min(peak_positions) - 20, max(peak_positions) + 30)
            plt.tight_layout()
            
            # 保存图像
            if output_dir:
                if image_name:
                    output_image = os.path.join(output_dir, f"{image_name}.png")
                else:
                    output_image = os.path.join(output_dir, "msl_peak_vs_atomic_number.png")
            else:
                if image_name:
                    output_image = f"{image_name}.png"
                else:
                    output_image = "msl_peak_vs_atomic_number.png"
            plt.savefig(output_image, dpi=300, bbox_inches='tight')
            generated_images.append(output_image)
            print(f"  图像已保存到: {output_image}")
            plt.close()
    
    if image_name is None or "莫塞莱定律" in image_name:
        print("\n5. 绘制莫塞莱定律图像：√E vs Z...")
        atomic_numbers = []
        peak_positions = []
        peak_stds = []
        element_symbols = []
        
        for result in results:
            element_symbol = result['element']
            atomic_number = get_atomic_number(element_symbol)
            
            if atomic_number is not None:
                atomic_numbers.append(atomic_number)
                peak_positions.append(result['peak_position'])
                peak_stds.append(result['peak_std'])
                element_symbols.append(element_symbol)
        
        if atomic_numbers:
            # 计算√E
            sqrt_E = np.sqrt(peak_positions)
            sqrt_E_errors = [std / (2 * np.sqrt(E)) if E > 0 else 0 for E, std in zip(peak_positions, peak_stds)]
            
            # 线性拟合
            Z_array = np.array(atomic_numbers)
            sqrt_E_array = np.array(sqrt_E)
            coefficients = np.polyfit(Z_array, sqrt_E_array, 1)
            a, b = coefficients
            sigma = -b / a
            
            # 创建图形
            plt.figure(figsize=(12, 8))
            
            # 设置中文字体
            try:
                plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
            except:
                pass
            
            # 绘制数据点
            for z, sqrt_e, sqrt_err in zip(Z_array, sqrt_E_array, sqrt_E_errors):
                plt.hlines(y=sqrt_e, xmin=z-0.3, xmax=z+0.3, color='red', linewidth=1.5, alpha=0.7)
                plt.hlines(y=sqrt_e+sqrt_err, xmin=z-0.2, xmax=z+0.2, color='gray', linewidth=1, alpha=0.6, linestyle='--')
                plt.hlines(y=sqrt_e-sqrt_err, xmin=z-0.2, xmax=z+0.2, color='gray', linewidth=1, alpha=0.6, linestyle='--')
                plt.vlines(x=z, ymin=sqrt_e-sqrt_err, ymax=sqrt_e+sqrt_err, color='gray', linewidth=1, alpha=0.6)
            
            # 绘制拟合线
            Z_fit = np.linspace(min(Z_array) - 2, max(Z_array) + 2, 100)
            sqrt_E_fit = a * Z_fit + b
            plt.plot(Z_fit, sqrt_E_fit, 'g--', linewidth=1.5, alpha=0.5, label=f'线性拟合: √E = {a:.3f}Z + {b:.3f}', zorder=4)
            
            # 标记Z轴截距
            plt.axvline(x=sigma, color='blue', linestyle=':', linewidth=2, alpha=0.7, label=f'屏蔽常数 σ = {sigma:.2f}')
            
            # 添加元素符号标签
            for z, sqrt_e, symbol in zip(Z_array, sqrt_E_array, element_symbols):
                plt.text(z, sqrt_e + 0.2, symbol, fontsize=10, ha='center', va='bottom', alpha=0.8)
            
            # 设置坐标轴标签
            plt.xlabel('原子序数 (Z)', fontsize=14)
            plt.ylabel('√E (√能量Index)', fontsize=14)
            plt.title('莫塞莱定律：√E vs Z 线性拟合', fontsize=16, fontweight='bold')
            plt.grid(True, alpha=0.3, linestyle='--')
            
            # 添加图例
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='red', label='√E 位置', linewidth=1.5, alpha=0.7),
                Line2D([0], [0], color='gray', linestyle='--', label='√E±σ范围', linewidth=1, alpha=0.6),
                Line2D([0], [0], color='green', linestyle='--', label=f'线性拟合: √E = {a:.3f}Z + {b:.3f}', linewidth=1.5, alpha=0.5),
            ]
            plt.legend(handles=legend_elements, loc='upper left', fontsize=11)
            
            # 设置坐标轴范围
            plt.xlim(min(Z_array) - 2, max(Z_array) + 2)
            plt.ylim(min(sqrt_E_array) - 1, max(sqrt_E_array) + 1)
            plt.tight_layout()
            
            # 保存图像
            if output_dir:
                if image_name:
                    output_image = os.path.join(output_dir, f"{image_name}.png")
                else:
                    output_image = os.path.join(output_dir, "moseley_law_fit.png")
            else:
                if image_name:
                    output_image = f"{image_name}.png"
                else:
                    output_image = "moseley_law_fit.png"
            plt.savefig(output_image, dpi=300, bbox_inches='tight')
            generated_images.append(output_image)
            print(f"  图像已保存到: {output_image}")
            plt.close()
    
    print("\n" + "=" * 60)
    print(f"生成完成！共生成 {len(generated_images)} 个图片")
    print("=" * 60)
    
    return generated_images


def main():
    """
    主函数
    """
    print("=" * 60)
    print("MSL数据处理程序 - 主函数")
    print("=" * 60)
    
    # 1. 获取MSL文件夹路径
    print("1. 获取MSL文件夹路径...")
    msl_path = get_msl_folder_path()
    print(f"   MSL路径: {msl_path}")
    
    # 检查路径是否存在
    if not os.path.exists(msl_path):
        print(f"错误: MSL文件夹不存在 - {msl_path}")
        print("请确保MSL文件夹位于正确的位置")
        return
    
    # 2. 查找所有.txt文件
    print("\n2. 查找MSL文件夹中的.txt文件...")
    txt_files = find_txt_files(msl_path)
    
    if not txt_files:
        print("  未找到任何.txt文件")
        return
    
    print(f"  找到 {len(txt_files)} 个.txt文件")
    
    # 3. 处理每个文件
    print("\n3. 处理文件并进行峰值拟合...")
    results = []
    
    for i, file_path in enumerate(txt_files):
        print(f"\n[{i+1}/{len(txt_files)}] ", end="")
        result = process_single_file(file_path, use_gaussian_module=True)
        if result:
            results.append(result)
    
    if not results:
        print("没有成功处理任何文件")
        return
    
    # 4. 保存结果到CSV文件（保存到脚本所在目录，而不是Figures目录）
    print("\n4. 保存结果到CSV文件...")
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "msl_peak_results.csv")
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 重新排序列
    df = df[['element', 'peak_position', 'peak_std', 'peak_fwhm', 'total_events', 'file_name']]
    
    # 保存到CSV
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"  结果已保存到: {output_file}")
    
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
