#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MSL数据线性拟合程序
功能：通过最小二乘法拟合参数b，使得process_msl_data.py中的峰位置(index)乘以b后
      对应到plot_kalpha_vs_z.py中的数据库Kα线能量(keV)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 添加当前目录到路径，以便导入xray_element_tool模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from xray_element_tool import ElementDatabase
    ELEMENT_DB_AVAILABLE = True
except ImportError:
    print("错误: 无法导入xray_element_tool模块")
    print("请确保xray_element_tool.py在同一目录下")
    ELEMENT_DB_AVAILABLE = False


def setup_chinese_font():
    """设置matplotlib中文字体支持"""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        return True
    except:
        return False


def load_msl_peak_data():
    """
    加载MSL峰位置数据
    
    返回:
    DataFrame: 包含元素、峰位置等数据的DataFrame
    """
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "msl_peak_results.csv")
    
    if not os.path.exists(csv_path):
        print(f"错误: 找不到MSL峰位置数据文件 {csv_path}")
        print("请先运行process_msl_data.py生成msl_peak_results.csv文件")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        print(f"成功加载MSL峰位置数据，共 {len(df)} 个元素")
        return df
    except Exception as e:
        print(f"加载MSL峰位置数据失败: {e}")
        return None


def get_database_kalpha_energy(element_symbol):
    """
    获取元素的数据库Kα线能量（查表值）
    
    参数:
    element_symbol: 元素符号（如'Fe', 'Cu'等）
    
    返回:
    float: 数据库Kα线能量（keV），如果找不到返回None
    """
    if not ELEMENT_DB_AVAILABLE:
        print(f"警告: 无法使用元素数据库，将使用硬编码的数据库值")
        # 硬编码的数据库Kα线能量（keV），来自xray_element_tool.py
        db_energies = {
            'Ag': 22.163, 'Co': 6.930, 'Cu': 8.047, 'Fe': 6.404,
            'Mo': 17.479, 'Ni': 7.478, 'Se': 11.222, 'Sr': 14.165,
            'Ti': 4.508, 'Zn': 8.638, 'Zr': 15.775
        }
        return db_energies.get(element_symbol)
    
    try:
        db = ElementDatabase()
        element_info = db.get_element_info(element_symbol)
        if element_info:
            k_alpha_db = element_info['properties'].get('k_alpha_exp')
            return k_alpha_db
    except Exception as e:
        print(f"获取元素 {element_symbol} 的数据库Kα线能量失败: {e}")
    
    return None


def get_atomic_numbers(elements):
    """
    获取元素的原子序数数组
    
    参数:
    elements: 元素符号数组
    
    返回:
    list: 原子序数数组
    """
    atomic_numbers = []
    
    if not ELEMENT_DB_AVAILABLE:
        # 硬编码的原子序数
        atomic_number_map = {
            'Ag': 47, 'Co': 27, 'Cu': 29, 'Fe': 26,
            'Mo': 42, 'Ni': 28, 'Se': 34, 'Sr': 38,
            'Ti': 22, 'Zn': 30, 'Zr': 40
        }
        for element in elements:
            atomic_numbers.append(atomic_number_map.get(element, 0))
        return atomic_numbers
    
    try:
        db = ElementDatabase()
        for element in elements:
            element_info = db.get_element_info(element)
            if element_info:
                atomic_numbers.append(element_info['atomic_number'])
            else:
                atomic_numbers.append(0)
    except Exception as e:
        print(f"获取原子序数失败: {e}")
        # 如果失败，使用硬编码值
        atomic_number_map = {
            'Ag': 47, 'Co': 27, 'Cu': 29, 'Fe': 26,
            'Mo': 42, 'Ni': 28, 'Se': 34, 'Sr': 38,
            'Ti': 22, 'Zn': 30, 'Zr': 40
        }
        for element in elements:
            atomic_numbers.append(atomic_number_map.get(element, 0))
    
    return atomic_numbers


def prepare_fitting_data(df):
    """
    准备拟合数据
    
    参数:
    df: 包含MSL峰位置数据的DataFrame
    
    返回:
    tuple: (indices, energies, elements, valid_mask)
      indices: 峰位置数组
      energies: 数据库Kα线能量数组（keV）
      elements: 元素符号数组
      valid_mask: 有效数据掩码（既有峰位置又有数据库能量的数据）
    """
    data = []
    
    for _, row in df.iterrows():
        element = row['element']
        index = row['peak_position']
        energy = get_database_kalpha_energy(element)
        
        if energy is not None:
            data.append({
                'element': element,
                'index': index,
                'energy': energy
            })
            print(f"  元素 {element}: 峰位置 = {index:.4f}, 数据库能量 = {energy:.4f} keV")
        else:
            print(f"  警告: 元素 {element} 没有数据库Kα线能量数据，跳过")
    
    if not data:
        print("错误: 没有有效的拟合数据")
        return None, None, None, None
    
    # 按照能量从小到大排序
    data.sort(key=lambda x: x['energy'])
    
    # 提取排序后的数据
    indices = [item['index'] for item in data]
    energies = [item['energy'] for item in data]
    elements = [item['element'] for item in data]
    
    return np.array(indices), np.array(energies), elements, None


def loss_function(b, indices, energies):
    """
    损失函数：最小二乘损失
    
    参数:
    b: 拟合参数（标量）
    indices: 峰位置数组
    energies: 数据库能量数组
    
    返回:
    float: 损失值
    """
    # 计算预测能量
    predicted = indices * b
    # 计算残差平方和
    loss = np.sum((predicted - energies) ** 2)
    return loss


def fit_linear_model(indices, energies):
    """
    拟合线性模型：E = b * index
    
    参数:
    indices: 峰位置数组
    energies: 数据库能量数组
    
    返回:
    dict: 包含拟合结果的字典
    """
    print("\n开始线性拟合...")
    
    # 使用最小二乘法直接计算b
    # 对于模型 E = b * index，最小二乘解为：
    # b = Σ(index_i * E_i) / Σ(index_i^2)
    numerator = np.sum(indices * energies)
    denominator = np.sum(indices ** 2)
    
    if denominator == 0:
        print("错误: 分母为零，无法计算最小二乘解")
        return None
    
    b_ls = numerator / denominator
    print(f"最小二乘法直接计算: b = {b_ls:.6f}")
    
    # 使用最小二乘解作为最终结果
    b_final = b_ls
    print(f"最终拟合参数: b = {b_final:.6f}")
    
    # 计算拟合优度指标
    predicted = indices * b_final
    residuals = predicted - energies
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((energies - np.mean(energies)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # 计算平均绝对误差和均方根误差
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    
    print(f"拟合优度 R² = {r_squared:.6f}")
    print(f"平均绝对误差 MAE = {mae:.6f} keV")
    print(f"均方根误差 RMSE = {rmse:.6f} keV")
    
    return {
        'b': b_final,
        'b_ls': b_ls,
        'loss': ss_res,
        'r_squared': r_squared,
        'mae': mae,
        'rmse': rmse,
        'predicted': predicted,
        'residuals': residuals
    }


def plot_comparison(indices, energies, elements, fit_result):
    """
    绘制对比图：将MSL数据转换后的能量与数据库能量对比
    
    参数:
    indices: 峰位置数组
    energies: 数据库能量数组（keV）
    elements: 元素符号数组
    fit_result: 拟合结果字典
    """
    if fit_result is None:
        print("错误: 没有拟合结果，无法绘图")
        return
    
    b = fit_result['b']
    predicted = fit_result['predicted']
    
    # 获取原子序数
    atomic_numbers = get_atomic_numbers(elements)
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 设置中文字体
    chinese_font_available = setup_chinese_font()
    
    # 第一个子图：能量对比散点图（横轴为原子序数）
    ax1.scatter(atomic_numbers, energies, color='blue', s=80, 
                label='数据库能量 (keV)', zorder=5, alpha=0.8)
    ax1.scatter(atomic_numbers, predicted, color='red', s=80, marker='s',
                label=f'MSL数据 × {b:.4f} (keV)', zorder=5, alpha=0.8)
    
    # 添加连接线
    for i in range(len(elements)):
        ax1.plot([atomic_numbers[i], atomic_numbers[i]], [energies[i], predicted[i]], 
                'gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # 添加数据标签（元素符号）
    for i in range(len(elements)):
        # 数据库能量点标签
        ax1.text(atomic_numbers[i], energies[i], f' {elements[i]}', 
                fontsize=9, ha='left', va='bottom', alpha=0.8)
        # MSL转换能量点标签
        ax1.text(atomic_numbers[i], predicted[i], f' {elements[i]}', 
                fontsize=9, ha='right', va='top', alpha=0.8)
    
    # 设置x轴标签
    ax1.set_xlabel('原子序数 Z', fontsize=12)
    ax1.set_ylabel('能量 (keV)', fontsize=12)
    
    # 设置标题
    ax1.set_title('MSL数据转换能量 vs 数据库Kα线能量 (按原子序数)', fontsize=14, fontweight='bold')
    
    # 添加网格和图例
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper left', fontsize=11)
    
    # 第二个子图：残差图（横轴为原子序数）
    residuals = fit_result['residuals']
    ax2.scatter(atomic_numbers, residuals, color='green', s=80, zorder=5, alpha=0.8)
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # 添加数据标签（元素符号）
    for i in range(len(elements)):
        ax2.text(atomic_numbers[i], residuals[i], f' {elements[i]}', 
                fontsize=9, ha='center', va='bottom' if residuals[i] >= 0 else 'top', alpha=0.8)
    
    # 设置坐标轴标签
    ax2.set_xlabel('原子序数 Z', fontsize=12)
    ax2.set_ylabel('残差 (keV)', fontsize=12)
    
    # 设置标题
    ax2.set_title('拟合残差 (数据库值 - 预测值) (按原子序数)', fontsize=14, fontweight='bold')
    
    # 添加网格
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # 调整布局
    plt.tight_layout()
    
    # 显示图像（不保存）
    print("正在显示对比图...")
    plt.show()




def main():
    """主函数"""
    print("=" * 60)
    print("MSL数据线性拟合程序")
    print("功能: 拟合参数b，使得 index × b ≈ 数据库Kα线能量")
    print("=" * 60)
    
    # 1. 加载MSL峰位置数据
    print("\n1. 加载MSL峰位置数据...")
    df = load_msl_peak_data()
    if df is None:
        return
    
    # 2. 准备拟合数据
    print("\n2. 准备拟合数据...")
    indices, energies, elements, valid_mask = prepare_fitting_data(df)
    if indices is None:
        return
    
    print(f"\n有效数据统计:")
    print(f"  总元素数: {len(df)}")
    print(f"  有效元素数: {len(indices)}")
    print(f"  无效元素数: {len(df) - len(indices)}")
    
    # 3. 执行线性拟合
    fit_result = fit_linear_model(indices, energies)
    if fit_result is None:
        return
    
    # 4. 打印详细拟合数据
    print("\n" + "=" * 60)
    print("详细拟合数据:")
    print("-" * 80)
    print(f"{'元素':<6} {'峰位置(index)':<15} {'数据库能量(keV)':<15} {'预测能量(keV)':<15} {'残差(keV)':<15}")
    print("-" * 80)
    
    for i in range(len(elements)):
        element = elements[i]
        index = indices[i]
        energy = energies[i]
        predicted = fit_result['predicted'][i]
        residual = fit_result['residuals'][i]
        print(f"{element:<8} {index:<18.4f} {energy:<18.4f} {predicted:<18.4f} {residual:<18.4f}")
    
    print("-" * 80)
    
    # 5. 打印最终拟合结果
    print("\n" + "=" * 60)
    print("最终拟合结果:")
    print(f"  拟合参数 b = {fit_result['b']:.6f}")
    print(f"  损失函数值 = {fit_result['loss']:.6f}")
    print(f"  拟合优度 R² = {fit_result['r_squared']:.6f}")
    print(f"  平均绝对误差 MAE = {fit_result['mae']:.6f} keV")
    print(f"  均方根误差 RMSE = {fit_result['rmse']:.6f} keV")
    
    # 6. 打印对比表：MSL转换能量 vs 数据库能量
    print("\n" + "=" * 60)
    print("能量对比表:")
    print("-" * 100)
    print(f"{'元素':<6} {'峰位置(index)':<15} {'MSL转换能量(keV)':<20} {'数据库能量(keV)':<20} {'相对偏差(%)':<15}")
    print("-" * 100)
    
    for i in range(len(elements)):
        element = elements[i]
        index = indices[i]
        msl_converted_energy = index * fit_result['b']  # MSL转换能量 = index × b
        db_energy = energies[i]  # 数据库能量
        relative_deviation = ((msl_converted_energy - db_energy) / db_energy) * 100  # 相对偏差百分比
        print(f"{element:<8} {index:<18.4f} {msl_converted_energy:<23.4f} {db_energy:<23.4f} {relative_deviation:<18.4f}")
    
    print("-" * 100)
    
    # 计算平均相对偏差
    relative_deviations = []
    for i in range(len(elements)):
        msl_converted_energy = indices[i] * fit_result['b']
        db_energy = energies[i]
        relative_deviation = ((msl_converted_energy - db_energy) / db_energy) * 100
        relative_deviations.append(relative_deviation)
    
    avg_relative_deviation = np.mean(np.abs(relative_deviations))
    print(f"平均相对偏差: {avg_relative_deviation:.4f}%")
    
    # 7. 绘制对比图（只显示不保存）
    print("\n7. 绘制对比图...")
    plot_comparison(indices, energies, elements, fit_result)
    
    print("\n" + "=" * 60)
    print("程序执行完毕！")
    print("=" * 60)


if __name__ == "__main__":
    main()
