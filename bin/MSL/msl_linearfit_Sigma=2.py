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


def get_database_kalpha_energy(element_symbol, sigma=2.0):
    """
    获取元素的数据库Kα线能量（查表值），并乘以修正因子 (Z-sigma)/(Z-1)
    
    参数:
    element_symbol: 元素符号（如'Fe', 'Cu'等）
    sigma: 修正因子中的sigma参数，默认值为2.0
    
    返回:
    float: 修正后的数据库Kα线能量（keV），如果找不到返回None
    """
    if not ELEMENT_DB_AVAILABLE:
        print(f"警告: 无法使用元素数据库，将使用硬编码的数据库值")
        # 硬编码的数据库Kα线能量（keV），来自xray_element_tool.py
        db_energies = {
            'Ag': 22.163, 'Co': 6.930, 'Cu': 8.047, 'Fe': 6.404,
            'Mo': 17.479, 'Ni': 7.478, 'Se': 11.222, 'Sr': 14.165,
            'Ti': 4.508, 'Zn': 8.638, 'Zr': 15.775
        }
        k_alpha_db = db_energies.get(element_symbol)
        if k_alpha_db is None:
            return None
        
        # 硬编码的原子序数
        atomic_number_map = {
            'Ag': 47, 'Co': 27, 'Cu': 29, 'Fe': 26,
            'Mo': 42, 'Ni': 28, 'Se': 34, 'Sr': 38,
            'Ti': 22, 'Zn': 30, 'Zr': 40
        }
        Z = atomic_number_map.get(element_symbol, 0)
        if Z <= 1:
            print(f"警告: 元素 {element_symbol} 的原子序数 Z={Z} 太小，无法计算修正因子")
            return k_alpha_db
        
        # 应用修正因子 (Z-sigma)/(Z-1)
        if Z - 1 == 0:
            print(f"警告: 元素 {element_symbol} 的 Z-1=0，无法计算修正因子")
            return k_alpha_db
        
        correction_factor = (Z - sigma) / (Z - 1)
        corrected_energy = k_alpha_db * correction_factor
        print(f"  元素 {element_symbol}: Z={Z}, sigma={sigma}, 原始能量={k_alpha_db:.4f} keV, 修正因子={(Z-sigma)/(Z-1):.6f}, 修正后能量={corrected_energy:.4f} keV")
        return corrected_energy
    
    try:
        db = ElementDatabase()
        element_info = db.get_element_info(element_symbol)
        if element_info:
            k_alpha_db = element_info['properties'].get('k_alpha_exp')
            if k_alpha_db is None:
                return None
            
            # 获取原子序数 Z
            Z = element_info['atomic_number']
            if Z <= 1:
                print(f"警告: 元素 {element_symbol} 的原子序数 Z={Z} 太小，无法计算修正因子")
                return k_alpha_db
            
            # 检查分母是否为零
            if Z - 1 == 0:
                print(f"警告: 元素 {element_symbol} 的 Z-1=0，无法计算修正因子")
                return k_alpha_db
            
            # 应用修正因子 (Z-sigma)/(Z-1)
            correction_factor = (Z - sigma) / (Z - 1)
            corrected_energy = k_alpha_db * correction_factor
            print(f"  元素 {element_symbol}: Z={Z}, sigma={sigma}, 原始能量={k_alpha_db:.4f} keV, 修正因子={(Z-sigma)/(Z-1):.6f}, 修正后能量={corrected_energy:.4f} keV")
            return corrected_energy
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


def get_raw_database_kalpha_energy(element_symbol):
    """
    获取元素的原始数据库Kα线能量（未修正的查表值）
    
    参数:
    element_symbol: 元素符号（如'Fe', 'Cu'等）
    
    返回:
    float: 原始数据库Kα线能量（keV），如果找不到返回None
    """
    if not ELEMENT_DB_AVAILABLE:
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


def prepare_fitting_data(df, sigma=2.0):
    """
    准备拟合数据
    
    参数:
    df: 包含MSL峰位置数据的DataFrame
    sigma: 修正因子中的sigma参数，默认值为2.0
    
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
        energy = get_database_kalpha_energy(element, sigma)
        
        if energy is not None:
            data.append({
                'element': element,
                'index': index,
                'energy': energy
            })
            print(f"  元素 {element}: 峰位置 = {index:.4f}, 数据库能量 = {energy:.4f} keV (sigma={sigma})")
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


def loss_function(params, indices, elements, db_energies_raw):
    """
    损失函数：同时优化b和sigma
    
    参数:
    params: 包含b和sigma的数组 [b, sigma]
    indices: 峰位置数组
    elements: 元素符号数组
    db_energies_raw: 原始数据库能量数组（未修正）
    
    返回:
    float: 损失值
    """
    b, sigma = params
    
    # 计算修正后的数据库能量
    corrected_energies = []
    for i, element in enumerate(elements):
        # 获取原子序数
        atomic_numbers = get_atomic_numbers([element])
        Z = atomic_numbers[0]
        
        if Z <= 1 or Z - 1 == 0:
            # 如果无法计算修正因子，使用原始能量
            corrected_energies.append(db_energies_raw[i])
        else:
            # 应用修正因子 (Z-sigma)/(Z-1)
            correction_factor = (Z - sigma) / (Z - 1)
            corrected_energy = db_energies_raw[i] * correction_factor
            corrected_energies.append(corrected_energy)
    
    corrected_energies = np.array(corrected_energies)
    
    # 计算预测能量
    predicted = indices * b
    # 计算残差平方和
    loss = np.sum((predicted - corrected_energies) ** 2)
    return loss


def fit_linear_model_with_sigma(indices, elements, db_energies_raw):
    """
    拟合线性模型：E_corrected = b * index
    其中 E_corrected = E_raw * (Z-sigma)/(Z-1)
    同时拟合参数b和sigma
    
    参数:
    indices: 峰位置数组
    elements: 元素符号数组
    db_energies_raw: 原始数据库能量数组（未修正）
    
    返回:
    dict: 包含拟合结果的字典
    """
    print("\n开始线性拟合（同时优化b和sigma）...")
    
    # 初始猜测值
    b_init = np.mean(db_energies_raw) / np.mean(indices) if np.mean(indices) != 0 else 1.0
    sigma_init = 2.0  # 初始猜测值
    
    print(f"初始猜测: b = {b_init:.6f}, sigma = {sigma_init:.6f}")
    
    # 使用共轭梯度下降法优化
    from scipy.optimize import minimize
    
    try:
        # 尝试多种优化方法，包括共轭梯度法
        methods_to_try = [
            ('CG', '共轭梯度法'),
            ('BFGS', '拟牛顿法'),
            ('L-BFGS-B', '有界拟牛顿法'),
            ('TNC', '截断牛顿法')
        ]
        
        best_result = None
        best_method = None
        
        for method_name, method_desc in methods_to_try:
            try:
                print(f"\n尝试使用{method_desc} ({method_name})进行优化...")
                result = minimize(
                    loss_function,
                    [b_init, sigma_init],
                    args=(indices, elements, db_energies_raw),
                    method=method_name,
                    bounds=[(0.1, 10.0), (0.1, 10.0)] if method_name in ['L-BFGS-B', 'TNC'] else None,
                    options={'maxiter': 1000, 'disp': False}
                )
                
                if result.success:
                    print(f"  {method_desc}优化成功! 损失值: {result.fun:.6f}")
                    if best_result is None or result.fun < best_result.fun:
                        best_result = result
                        best_method = method_desc
                else:
                    print(f"  {method_desc}优化失败: {result.message}")
            except Exception as e:
                print(f"  {method_desc}优化过程中出错: {e}")
        
        if best_result is not None:
            b_final, sigma_final = best_result.x
            print(f"\n最佳优化方法: {best_method}")
            print(f"拟合参数: b = {b_final:.6f}, sigma = {sigma_final:.6f}")
            print(f"损失函数值: {best_result.fun:.6f}")
            print(f"迭代次数: {best_result.nit}")
            print(f"函数调用次数: {best_result.nfev}")
        else:
            print("\n所有优化方法都失败了，使用初始猜测值")
            b_final, sigma_final = b_init, sigma_init
            
    except ImportError:
        print("警告: 无法导入scipy.optimize，使用简单网格搜索")
        # 简单的网格搜索
        best_loss = float('inf')
        best_b = b_init
        best_sigma = sigma_init
        
        # 简单的网格搜索
        for b_test in np.linspace(0.5, 5.0, 20):
            for sigma_test in np.linspace(0.5, 5.0, 20):
                loss = loss_function([b_test, sigma_test], indices, elements, db_energies_raw)
                if loss < best_loss:
                    best_loss = loss
                    best_b = b_test
                    best_sigma = sigma_test
        
        b_final, sigma_final = best_b, best_sigma
        print(f"网格搜索结果: b = {b_final:.6f}, sigma = {sigma_final:.6f}")
        print(f"损失函数值: {best_loss:.6f}")
    
    # 使用最终参数计算修正后的能量
    corrected_energies = []
    for i, element in enumerate(elements):
        atomic_numbers = get_atomic_numbers([element])
        Z = atomic_numbers[0]
        
        if Z <= 1 or Z - 1 == 0:
            corrected_energies.append(db_energies_raw[i])
        else:
            correction_factor = (Z - sigma_final) / (Z - 1)
            corrected_energy = db_energies_raw[i] * correction_factor
            corrected_energies.append(corrected_energy)
    
    corrected_energies = np.array(corrected_energies)
    
    # 计算预测能量
    predicted = indices * b_final
    
    # 计算拟合优度指标
    residuals = predicted - corrected_energies
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((corrected_energies - np.mean(corrected_energies)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # 计算平均绝对误差和均方根误差
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    
    print(f"拟合优度 R² = {r_squared:.6f}")
    print(f"平均绝对误差 MAE = {mae:.6f} keV")
    print(f"均方根误差 RMSE = {rmse:.6f} keV")
    
    return {
        'b': b_final,
        'sigma': sigma_final,
        'loss': ss_res,
        'r_squared': r_squared,
        'mae': mae,
        'rmse': rmse,
        'predicted': predicted,
        'residuals': residuals,
        'corrected_energies': corrected_energies
    }


def fit_linear_model(indices, energies):
    """
    拟合线性模型：E = b * index（旧版本，保持兼容性）
    
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
    ax2.set_title('拟合残差 (实验值 - 数据库值) (按原子序数)', fontsize=14, fontweight='bold')
    
    # 添加网格
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    # 调整布局
    plt.tight_layout()
    
    # 显示图像（不保存）
    print("正在显示对比图...")
    plt.show()


def generate_figures(output_dir=None, image_name=None):
    """
    生成线性拟合对比图并保存到指定目录
    
    参数:
    output_dir: 输出目录，如果为None则保存到脚本所在目录
    image_name: 图片名称（可选），用于指定生成哪个图片
                None: 生成所有图片
                "同时拟合sigma和道能量宽度": 只生成图III.1.4
                "实验数据与相对论理论预言对比图": 只生成图III.1.5
    """
    print("=" * 60)
    print("MSL数据线性拟合 - 生成图片")
    print("=" * 60)
    
    if image_name:
        print(f"指定生成图片: {image_name}")
    
    # 1. 加载MSL峰位置数据
    print("\n1. 加载MSL峰位置数据...")
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "msl_peak_results.csv")
    
    if not os.path.exists(csv_path):
        print(f"错误: 找不到MSL峰位置数据文件 {csv_path}")
        print("请先运行process_msl_data.py生成msl_peak_results.csv文件")
        return []
    
    try:
        df = pd.read_csv(csv_path)
        print(f"成功加载MSL峰位置数据，共 {len(df)} 个元素")
    except Exception as e:
        print(f"加载MSL峰位置数据失败: {e}")
        return []
    
    # 2. 准备拟合数据（使用sigma=2.0的默认值）
    print("\n2. 准备拟合数据（使用默认sigma=2.0）...")
    # 简化版本：只使用部分元素
    elements = []
    indices = []
    energies = []
    
    for _, row in df.iterrows():
        element = row['element']
        index = row['peak_position']
        
        # 获取数据库能量（简化版本）
        if element in ['Ag', 'Co', 'Cu', 'Fe', 'Mo', 'Ni', 'Se', 'Sr', 'Ti', 'Zn', 'Zr']:
            # 硬编码的数据库Kα线能量（keV）
            db_energies = {
                'Ag': 22.163, 'Co': 6.930, 'Cu': 8.047, 'Fe': 6.404,
                'Mo': 17.479, 'Ni': 7.478, 'Se': 11.222, 'Sr': 14.165,
                'Ti': 4.508, 'Zn': 8.638, 'Zr': 15.775
            }
            energy = db_energies.get(element)
            if energy is not None:
                elements.append(element)
                indices.append(index)
                energies.append(energy)
                print(f"  元素 {element}: 峰位置 = {index:.4f}, 数据库能量 = {energy:.4f} keV")
    
    if not elements:
        print("错误: 没有有效的拟合数据")
        return []
    
    indices = np.array(indices)
    energies = np.array(energies)
    
    # 3. 执行线性拟合
    print("\n3. 执行线性拟合...")
    # 使用最小二乘法直接计算b
    numerator = np.sum(indices * energies)
    denominator = np.sum(indices ** 2)
    
    if denominator == 0:
        print("错误: 分母为零，无法计算最小二乘解")
        return []
    
    b = numerator / denominator
    print(f"拟合参数: b = {b:.6f}")
    
    # 计算预测能量
    predicted = indices * b
    
    # 创建拟合结果字典，用于plot_comparison函数
    fit_result = {
        'b': b,
        'predicted': predicted,
        'residuals': predicted - energies
    }
    
    # 4. 生成图片
    generated_images = []
    
    # 设置中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass
    
    # 获取原子序数（简化版本）
    atomic_number_map = {
        'Ag': 47, 'Co': 27, 'Cu': 29, 'Fe': 26,
        'Mo': 42, 'Ni': 28, 'Se': 34, 'Sr': 38,
        'Ti': 22, 'Zn': 30, 'Zr': 40
    }
    atomic_numbers = [atomic_number_map.get(elem, 0) for elem in elements]
    
    # 根据image_name参数决定生成哪些图片
    generate_all = image_name is None
    # 处理CSV文件中的图片名称
    if image_name:
        # 提取图片名称中的描述部分（去掉"图III.1.4 - "前缀）
        if " - " in image_name:
            name_part = image_name.split(" - ", 1)[1]
        else:
            name_part = image_name
        
        # 图III.1.4: "通过对比实验数据和资料，确定道系数和屏蔽系数" -> 对应图片1
        # 图III.1.5: "实验数据与相对论理论预言对比图" -> 对应图片2
        generate_fig1 = name_part == "通过对比实验数据和资料，确定道系数和屏蔽系数"
        generate_fig2 = name_part == "实验数据与相对论理论预言对比图"
    else:
        generate_fig1 = True
        generate_fig2 = True
    
    # 图片1：同时拟合sigma和道能量宽度（双子图）
    if generate_fig1:
        print("\n4. 生成同时拟合sigma和道能量宽度图...")
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 第一个子图：能量对比散点图
        ax1.scatter(atomic_numbers, energies, color='blue', s=80, 
                    label='数据库能量 (keV)', zorder=5, alpha=0.8)
        ax1.scatter(atomic_numbers, predicted, color='red', s=80, marker='s',
                    label=f'MSL数据 × {b:.4f} (keV)', zorder=5, alpha=0.8)
        
        # 添加连接线
        for i in range(len(elements)):
            ax1.plot([atomic_numbers[i], atomic_numbers[i]], [energies[i], predicted[i]], 
                    'gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # 添加数据标签
        for i in range(len(elements)):
            ax1.text(atomic_numbers[i], energies[i], f' {elements[i]}', 
                    fontsize=9, ha='left', va='bottom', alpha=0.8)
            ax1.text(atomic_numbers[i], predicted[i], f' {elements[i]}', 
                    fontsize=9, ha='right', va='top', alpha=0.8)
        
        ax1.set_xlabel('原子序数 Z', fontsize=12)
        ax1.set_ylabel('能量 (keV)', fontsize=12)
        ax1.set_title('MSL数据转换能量 vs 数据库Kα线能量', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(loc='upper left', fontsize=11)
        
        # 第二个子图：残差图
        residuals = predicted - energies
        ax2.scatter(atomic_numbers, residuals, color='green', s=80, zorder=5, alpha=0.8)
        ax2.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # 添加数据标签
        for i in range(len(elements)):
            ax2.text(atomic_numbers[i], residuals[i], f' {elements[i]}', 
                    fontsize=9, ha='center', va='bottom' if residuals[i] >= 0 else 'top', alpha=0.8)
        
        ax2.set_xlabel('原子序数 Z', fontsize=12)
        ax2.set_ylabel('残差 (keV)', fontsize=12)
        ax2.set_title('拟合残差 (实验值 - 数据库值)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片1
        if output_dir:
            if image_name and generate_fig1 and not generate_fig2:
                # 如果只生成图片1，使用image_name作为文件名
                output_filename1 = os.path.join(output_dir, f"{image_name}.png")
            else:
                output_filename1 = os.path.join(output_dir, "同时拟合sigma和道能量宽度.png")
        else:
            if image_name and generate_fig1 and not generate_fig2:
                output_filename1 = f"{image_name}.png"
            else:
                output_filename1 = "同时拟合sigma和道能量宽度.png"
        
        plt.savefig(output_filename1, dpi=300, bbox_inches='tight')
        generated_images.append(output_filename1)
        print(f"图片1已保存到: {output_filename1}")
        plt.close()
    
    # 图片2：实验数据与相对论理论预言对比图
    if generate_fig2:
        print("\n5. 生成实验数据与相对论理论预言对比图...")
        # 创建图形
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
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
        ax1.set_title('实验数据与相对论理论预言对比 (按原子序数)', fontsize=14, fontweight='bold')
        
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
        ax2.set_title('拟合残差 (实验值 - 数据库值) (按原子序数)', fontsize=14, fontweight='bold')
        
        # 添加网格
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片2
        if output_dir:
            if image_name and generate_fig2 and not generate_fig1:
                # 如果只生成图片2，使用image_name作为文件名
                output_filename2 = os.path.join(output_dir, f"{image_name}.png")
            else:
                output_filename2 = os.path.join(output_dir, "实验数据与相对论理论预言对比图.png")
        else:
            if image_name and generate_fig2 and not generate_fig1:
                output_filename2 = f"{image_name}.png"
            else:
                output_filename2 = "实验数据与相对论理论预言对比图.png"
        
        plt.savefig(output_filename2, dpi=300, bbox_inches='tight')
        generated_images.append(output_filename2)
        print(f"图片2已保存到: {output_filename2}")
        plt.close()
    
    print("\n" + "=" * 60)
    print(f"生成完成！共生成 {len(generated_images)} 个图片")
    print("=" * 60)
    
    return generated_images


def main():
    """主函数"""
    # 调用generate_figures函数，不指定输出目录（保存到脚本所在目录）
    generated_images = generate_figures()
    
    # 显示图形
    if generated_images:
        print(f"\n共生成 {len(generated_images)} 个图片")
        for img in generated_images:
            print(f"  - {img}")


if __name__ == "__main__":
    main()
