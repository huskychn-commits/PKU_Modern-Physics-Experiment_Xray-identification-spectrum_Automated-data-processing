#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制MSL转换能量与数据库Kα线能量的对比图
功能：
1. 调用msl_linearfit_Sigma=2.py获取b值
2. 将MSL峰位置（index）转换为keV单位的实际能量
3. 绘制数据库数据和MSL转换数据的对比图
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入必要的模块
try:
    from xray_element_tool import ElementDatabase
    ELEMENT_DB_AVAILABLE = True
except ImportError:
    print("错误: 无法导入xray_element_tool模块")
    ELEMENT_DB_AVAILABLE = False

# 设置中文字体支持
def setup_chinese_font():
    """设置matplotlib中文字体支持"""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        return True
    except:
        return False


def get_b_value_from_fitting():
    """
    获取b值
    
    返回:
    float: b值
    """
    # 使用之前运行得到的b值（避免运行拟合脚本）
    print("使用缓存的b值（避免运行拟合脚本）")
    return 0.064675  # 使用之前运行得到的默认值


def load_msl_data():
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


def collect_database_data():
    """
    收集数据库Kα线能量数据
    
    返回:
    tuple: (Z_list, exp_list, nonrel_list, rel_list, element_symbols)
      Z_list: 原子序数列表
      exp_list: 实验Kα线能量列表（keV）
      nonrel_list: 非相对论理论Kα线能量列表（keV）
      rel_list: 相对论理论Kα线能量列表（keV）
      element_symbols: 元素符号列表
    """
    if not ELEMENT_DB_AVAILABLE:
        print("错误: 元素数据库不可用")
        return None, None, None, None, None
    
    try:
        db = ElementDatabase()
        
        Z_list = []
        exp_list = []
        nonrel_list = []
        rel_list = []
        element_symbols = []
        
        # 获取所有元素
        elements = db.list_all_elements()
        
        for elem in elements:
            Z = elem['atomic_number']
            symbol = elem['symbol']
            
            # 获取元素完整信息
            element_info = db.get_element_info(symbol)
            if not element_info:
                continue
                
            # 获取实验Kα线能量（如果存在）
            k_alpha_exp = element_info['properties'].get('k_alpha_exp')
            
            # 获取非相对论理论Kα线能量
            k_alpha_nonrel = db.get_k_alpha_energy(symbol)
            
            # 获取相对论理论Kα线能量
            k_alpha_rel = db.get_k_alpha_energy_relativistic(symbol)
            
            # 只收集所有数据都有效的元素
            if k_alpha_nonrel is not None and k_alpha_rel is not None:
                Z_list.append(Z)
                exp_list.append(k_alpha_exp if k_alpha_exp is not None else None)
                nonrel_list.append(k_alpha_nonrel)
                rel_list.append(k_alpha_rel)
                element_symbols.append(symbol)
        
        return Z_list, exp_list, nonrel_list, rel_list, element_symbols
        
    except Exception as e:
        print(f"收集数据库数据时出错: {e}")
        return None, None, None, None, None


def plot_comparison(b_value, sigma=2.0):
    """
    绘制MSL转换能量与数据库Kα线能量的对比图
    
    参数:
    b_value: MSL index到keV的转换系数
    sigma: 修正因子中的sigma参数，默认值为2.0
    """
    print(f"\n使用b值进行能量转换: b = {b_value:.6f}")
    print(f"使用修正因子中的sigma参数: sigma = {sigma:.6f}")
    
    # 加载MSL数据
    print("加载MSL数据...")
    msl_df = load_msl_data()
    if msl_df is None:
        return
    
    # 收集数据库数据
    print("收集数据库数据...")
    db_Z, db_exp, db_nonrel, db_rel, db_symbols = collect_database_data()
    if db_Z is None:
        return
    
    # 准备MSL数据
    msl_Z = []
    msl_energies = []
    msl_symbols = []
    msl_errors = []
    
    for _, row in msl_df.iterrows():
        element = row['element']
        index = row['peak_position']
        std = row['peak_std']
        
        # 获取原子序数
        Z = get_atomic_number(element)
        if Z is None:
            print(f"警告: 无法获取元素 {element} 的原子序数，跳过")
            continue
        
        # 转换能量：E = index × b
        energy = index * b_value
        error = std * b_value  # 误差传播
        
        msl_Z.append(Z)
        msl_energies.append(energy)
        msl_symbols.append(element)
        msl_errors.append(error)
    
    if not msl_Z:
        print("错误: 没有有效的MSL数据")
        return
    
    # 应用修正因子到数据库数据
    print("\n应用修正因子到数据库数据...")
    db_exp_corrected = []
    db_nonrel_corrected = []
    db_rel_corrected = []
    
    for i, Z in enumerate(db_Z):
        # 计算修正因子 (Z-sigma)/(Z-1)
        if Z > 1 and Z - 1 != 0:
            correction_factor = (Z - sigma) / (Z - 1)
        else:
            correction_factor = 1.0
            print(f"警告: 元素 {db_symbols[i]} (Z={Z}) 的Z-1=0，无法计算修正因子，使用1.0")
        
        # 应用修正因子
        if db_exp[i] is not None:
            db_exp_corrected.append(db_exp[i] * correction_factor)
        else:
            db_exp_corrected.append(None)
        
        db_nonrel_corrected.append(db_nonrel[i] * correction_factor)
        db_rel_corrected.append(db_rel[i] * correction_factor)
        
        print(f"  元素 {db_symbols[i]}: Z={Z}, 修正因子={correction_factor:.6f}")
    
    # 打印修正前后的对比
    print("\n数据库数据修正前后对比:")
    print("-" * 120)
    print(f"{'元素':<6} {'原子序数':<10} {'原始理论值':<15} {'修正后理论值':<15} {'原始实验值':<15} {'修正后实验值':<15}")
    print("-" * 120)
    
    for i in range(len(db_Z)):
        element = db_symbols[i]
        Z = db_Z[i]
        rel_original = db_rel[i]
        rel_corrected = db_rel_corrected[i]
        exp_original = db_exp[i] if db_exp[i] is not None else "N/A"
        exp_corrected = db_exp_corrected[i] if db_exp_corrected[i] is not None else "N/A"
        
        print(f"{element:<8} {Z:<12} {rel_original:<18.4f} {rel_corrected:<18.4f} {str(exp_original):<18} {str(exp_corrected):<18}")
    
    print("-" * 120)
    
    # 创建图形（单个图，不需要残差图）
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 设置中文字体
    chinese_font_available = setup_chinese_font()
    
    print("\n绘制数据库数据与MSL转换数据对比图...")
    
    # 绘制数据库理论线（使用修正后的数据）
    db_Z_array = np.array(db_Z)
    db_rel_corrected_array = np.array(db_rel_corrected)
    
    # 对原子序数进行排序以便绘制平滑曲线
    sorted_indices = np.argsort(db_Z_array)
    db_Z_sorted = db_Z_array[sorted_indices]
    db_rel_corrected_sorted = db_rel_corrected_array[sorted_indices]
    
    # 绘制修正后的相对论理论线（蓝色，线宽减半，不透明度60%）
    ax.plot(db_Z_sorted, db_rel_corrected_sorted, 'b-', linewidth=1, 
             label='数据库相对论理论线（修正后）', zorder=2, alpha=0.6)
    
    # 绘制数据库实验数据点（使用修正后的数据，绿色横线格式）
    exp_Z = []
    exp_E_corrected = []
    for i, (Z, exp_corrected) in enumerate(zip(db_Z, db_exp_corrected)):
        if exp_corrected is not None:
            exp_Z.append(Z)
            exp_E_corrected.append(exp_corrected)
    
    if exp_Z:
        # 使用绿色横线表示修正后的数据库实验数据点
        for Z, energy in zip(exp_Z, exp_E_corrected):
            # 绘制绿色横线表示修正后的数据库实验数据点
            ax.hlines(y=energy, xmin=Z-0.3, xmax=Z+0.3, color='green', 
                      linewidth=1.5, alpha=0.7, zorder=5)
        
        # 注意：这里不添加误差线，因为数据库实验数据通常没有误差信息
    
    # 绘制MSL转换数据点（使用process_msl_data.py中的样式）
    # 参考process_msl_data.py中的test_plot函数样式：
    # 1. 红色横线表示峰位置
    # 2. 灰色虚线表示E±σ范围
    # 3. 灰色垂直线连接E±σ
    
    for i, (Z, energy, error) in enumerate(zip(msl_Z, msl_energies, msl_errors)):
        # 绘制红色横线表示峰位置
        ax.hlines(y=energy, xmin=Z-0.3, xmax=Z+0.3, color='red', 
                  linewidth=1.5, alpha=0.7, zorder=6)
        
        # 绘制灰色虚线表示E±σ范围
        ax.hlines(y=energy+error, xmin=Z-0.2, xmax=Z+0.2, color='gray', 
                  linewidth=1, alpha=0.6, linestyle='--', zorder=5)
        ax.hlines(y=energy-error, xmin=Z-0.2, xmax=Z+0.2, color='gray', 
                  linewidth=1, alpha=0.6, linestyle='--', zorder=5)
        
        # 绘制灰色垂直线连接E±σ
        ax.vlines(x=Z, ymin=energy-error, ymax=energy+error, color='gray', 
                  linewidth=1, alpha=0.6, zorder=5)
    
    # 添加元素符号标签（MSL数据）
    for i, (Z, energy, symbol) in enumerate(zip(msl_Z, msl_energies, msl_symbols)):
        ax.text(Z, energy + msl_errors[i] + 0.5, symbol, fontsize=9, 
                ha='center', va='bottom', alpha=0.8, zorder=7)
    
    # 设置坐标轴标签
    ax.set_xlabel('原子序数 Z', fontsize=12)
    ax.set_ylabel('能量 (keV)', fontsize=12)
    
    # 添加网格和图例
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 创建自定义图例项
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', label='数据库相对论理论线（修正后）',
               linewidth=1, alpha=0.6),
        Line2D([0], [0], color='green', label='数据库实验数据（修正后）',
               linewidth=1.5, alpha=0.7),
        Line2D([0], [0], color='red', label='MSL转换峰位置',
               linewidth=1.5, alpha=0.7),
        Line2D([0], [0], color='gray', linestyle='--', label='MSL E±σ范围',
               linewidth=1, alpha=0.6)
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11)
    
    # 设置坐标轴范围
    all_Z = msl_Z + db_Z
    # 使用修正后的数据库理论值和MSL转换能量中的最大值
    all_E = msl_energies + db_rel_corrected
    ax.set_xlim(min(all_Z) - 2, max(all_Z) + 2)
    ax.set_ylim(0, max(all_E) * 1.1)
    
    # 更新标题，包含sigma参数
    ax.set_title(f'MSL转换能量 vs 数据库Kα线能量 (b={b_value:.6f}, σ={sigma:.6f})', fontsize=14, fontweight='bold')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    output_image = os.path.join(os.path.dirname(os.path.abspath(__file__)), "msl_vs_database_comparison.png")
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"\n图像已保存到: {output_image}")
    
    # 显示图像
    print("正在显示对比图...")
    plt.show()
    
    # 打印详细数据（使用修正后的数据库理论值）
    print("\n" + "=" * 80)
    print("MSL转换能量与修正后数据库理论值对比表:")
    print("-" * 120)
    print(f"{'元素':<6} {'原子序数':<10} {'MSL峰位置':<15} {'MSL转换能量(keV)':<20} {'修正后数据库理论值(keV)':<20} {'残差(keV)':<15} {'修正因子':<15}")
    print("-" * 120)
    
    # 计算残差用于表格显示
    for i, Z in enumerate(msl_Z):
        # 找到对应的数据库理论值
        db_index = None
        for j, db_z in enumerate(db_Z):
            if db_z == Z:
                db_index = j
                break
        
        if db_index is not None:
            element = msl_symbols[i]
            index = msl_df[msl_df['element'] == element]['peak_position'].values[0]
            msl_energy = msl_energies[i]
            db_theory_corrected = db_rel_corrected[db_index]
            residual = msl_energy - db_theory_corrected
            
            # 计算修正因子
            if Z > 1 and Z - 1 != 0:
                correction_factor = (Z - sigma) / (Z - 1)
            else:
                correction_factor = 1.0
            
            print(f"{element:<8} {Z:<12} {index:<18.4f} {msl_energy:<23.4f} {db_theory_corrected:<23.4f} {residual:<18.4f} {correction_factor:<18.6f}")
    
    print("-" * 120)
    
    # 计算统计指标
    residuals = []
    for i, Z in enumerate(msl_Z):
        db_index = None
        for j, db_z in enumerate(db_Z):
            if db_z == Z:
                db_index = j
                break
        
        if db_index is not None:
            msl_energy = msl_energies[i]
            db_theory_corrected = db_rel_corrected[db_index]
            residual = msl_energy - db_theory_corrected
            residuals.append(residual)
    
    if residuals:
        avg_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        avg_abs_residual = np.mean(np.abs(residuals))
        print(f"平均残差: {avg_residual:.4f} keV")
        print(f"残差标准差: {std_residual:.4f} keV")
        print(f"平均绝对残差: {avg_abs_residual:.4f} keV")


def generate_figures(output_dir=None, image_name=None):
    """
    生成MSL转换能量与数据库Kα线能量对比图
    
    参数:
    output_dir: 输出目录，如果为None则保存到脚本所在目录
    image_name: 图片名称（可选），用于指定生成哪个图片
    """
    print("=" * 60)
    print("MSL转换能量与数据库Kα线能量对比 - 生成图片")
    print("=" * 60)
    
    if image_name:
        print(f"指定生成图片: {image_name}")
    
    # 获取b值
    print("\n1. 获取MSL index到keV的转换系数b...")
    b_value = get_b_value_from_fitting()
    print(f"   获取到的b值: {b_value:.6f}")
    
    # 绘制对比图（使用默认sigma=2.0）
    print("\n2. 绘制MSL转换能量与数据库Kα线能量对比图...")
    sigma = 2.0  # 默认sigma值
    
    # 调用plot_comparison函数，但修改它以支持保存到指定目录
    # 由于plot_comparison函数显示图形但不保存，我们需要修改它
    # 这里我们创建一个简化的版本
    
    # 加载MSL数据
    print("加载MSL数据...")
    msl_df = load_msl_data()
    if msl_df is None:
        return []
    
    # 收集数据库数据
    print("收集数据库数据...")
    db_Z, db_exp, db_nonrel, db_rel, db_symbols = collect_database_data()
    if db_Z is None:
        return []
    
    # 准备MSL数据
    msl_Z = []
    msl_energies = []
    msl_symbols = []
    msl_errors = []
    
    for _, row in msl_df.iterrows():
        element = row['element']
        index = row['peak_position']
        std = row['peak_std']
        
        # 获取原子序数
        Z = get_atomic_number(element)
        if Z is None:
            print(f"警告: 无法获取元素 {element} 的原子序数，跳过")
            continue
        
        # 转换能量：E = index × b
        energy = index * b_value
        error = std * b_value  # 误差传播
        
        msl_Z.append(Z)
        msl_energies.append(energy)
        msl_symbols.append(element)
        msl_errors.append(error)
    
    if not msl_Z:
        print("错误: 没有有效的MSL数据")
        return []
    
    # 应用修正因子到数据库数据
    print("\n应用修正因子到数据库数据...")
    db_exp_corrected = []
    db_nonrel_corrected = []
    db_rel_corrected = []
    
    for i, Z in enumerate(db_Z):
        # 计算修正因子 (Z-sigma)/(Z-1)
        if Z > 1 and Z - 1 != 0:
            correction_factor = (Z - sigma) / (Z - 1)
        else:
            correction_factor = 1.0
            print(f"警告: 元素 {db_symbols[i]} (Z={Z}) 的Z-1=0，无法计算修正因子，使用1.0")
        
        # 应用修正因子
        if db_exp[i] is not None:
            db_exp_corrected.append(db_exp[i] * correction_factor)
        else:
            db_exp_corrected.append(None)
        
        db_nonrel_corrected.append(db_nonrel[i] * correction_factor)
        db_rel_corrected.append(db_rel[i] * correction_factor)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 设置中文字体
    chinese_font_available = setup_chinese_font()
    
    print("\n绘制数据库数据与MSL转换数据对比图...")
    
    # 绘制数据库理论线（使用修正后的数据）
    db_Z_array = np.array(db_Z)
    db_rel_corrected_array = np.array(db_rel_corrected)
    
    # 对原子序数进行排序以便绘制平滑曲线
    sorted_indices = np.argsort(db_Z_array)
    db_Z_sorted = db_Z_array[sorted_indices]
    db_rel_corrected_sorted = db_rel_corrected_array[sorted_indices]
    
    # 绘制修正后的相对论理论线
    ax.plot(db_Z_sorted, db_rel_corrected_sorted, 'b-', linewidth=1, 
             label='数据库相对论理论线（修正后）', zorder=2, alpha=0.6)
    
    # 绘制数据库实验数据点（使用修正后的数据，绿色横线格式）
    exp_Z = []
    exp_E_corrected = []
    for i, (Z, exp_corrected) in enumerate(zip(db_Z, db_exp_corrected)):
        if exp_corrected is not None:
            exp_Z.append(Z)
            exp_E_corrected.append(exp_corrected)
    
    if exp_Z:
        # 使用绿色横线表示修正后的数据库实验数据点
        for Z, energy in zip(exp_Z, exp_E_corrected):
            ax.hlines(y=energy, xmin=Z-0.3, xmax=Z+0.3, color='green', 
                      linewidth=1.5, alpha=0.7, zorder=5)
    
    # 绘制MSL转换数据点
    for i, (Z, energy, error) in enumerate(zip(msl_Z, msl_energies, msl_errors)):
        # 绘制红色横线表示峰位置
        ax.hlines(y=energy, xmin=Z-0.3, xmax=Z+0.3, color='red', 
                  linewidth=1.5, alpha=0.7, zorder=6)
        
        # 绘制灰色虚线表示E±σ范围
        ax.hlines(y=energy+error, xmin=Z-0.2, xmax=Z+0.2, color='gray', 
                  linewidth=1, alpha=0.6, linestyle='--', zorder=5)
        ax.hlines(y=energy-error, xmin=Z-0.2, xmax=Z+0.2, color='gray', 
                  linewidth=1, alpha=0.6, linestyle='--', zorder=5)
        
        # 绘制灰色垂直线连接E±σ
        ax.vlines(x=Z, ymin=energy-error, ymax=energy+error, color='gray', 
                  linewidth=1, alpha=0.6, zorder=5)
    
    # 添加元素符号标签（MSL数据）
    for i, (Z, energy, symbol) in enumerate(zip(msl_Z, msl_energies, msl_symbols)):
        ax.text(Z, energy + msl_errors[i] + 0.5, symbol, fontsize=9, 
                ha='center', va='bottom', alpha=0.8, zorder=7)
    
    # 设置坐标轴标签
    ax.set_xlabel('原子序数 Z', fontsize=12)
    ax.set_ylabel('能量 (keV)', fontsize=12)
    
    # 添加网格和图例
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 创建自定义图例项
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', label='数据库相对论理论线（修正后）',
               linewidth=1, alpha=0.6),
        Line2D([0], [0], color='green', label='数据库实验数据（修正后）',
               linewidth=1.5, alpha=0.7),
        Line2D([0], [0], color='red', label='MSL转换峰位置',
               linewidth=1.5, alpha=0.7),
        Line2D([0], [0], color='gray', linestyle='--', label='MSL E±σ范围',
               linewidth=1, alpha=0.6)
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11)
    
    # 设置坐标轴范围
    all_Z = msl_Z + db_Z
    all_E = msl_energies + db_rel_corrected
    ax.set_xlim(min(all_Z) - 2, max(all_Z) + 2)
    ax.set_ylim(0, max(all_E) * 1.1)
    
    # 设置标题
    ax.set_title(f'MSL转换能量 vs 数据库Kα线能量 (b={b_value:.6f}, σ={sigma:.6f})', fontsize=14, fontweight='bold')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    if output_dir:
        if image_name:
            output_filename = os.path.join(output_dir, f"{image_name}.png")
        else:
            output_filename = os.path.join(output_dir, "msl_vs_database_comparison.png")
    else:
        if image_name:
            output_filename = f"{image_name}.png"
        else:
            output_filename = "msl_vs_database_comparison.png"
    
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\n图像已保存到: {output_filename}")
    plt.close()
    
    generated_images = [output_filename]
    
    print("\n" + "=" * 60)
    print(f"生成完成！共生成 {len(generated_images)} 个图片")
    print("=" * 60)
    
    return generated_images


def main():
    """主函数"""
    print("=" * 60)
    print("MSL转换能量与数据库Kα线能量对比程序")
    print("功能: 将MSL峰位置转换为keV单位，并与数据库理论值对比")
    print("=" * 60)
    
    # 获取b值
    print("\n1. 获取MSL index到keV的转换系数b...")
    b_value = get_b_value_from_fitting()
    print(f"   获取到的b值: {b_value:.6f}")
    
    # 绘制对比图（使用默认sigma=2.0）
    print("\n2. 绘制MSL转换能量与数据库Kα线能量对比图...")
    sigma = 2.0  # 默认sigma值
    plot_comparison(b_value, sigma)
    
    print("\n" + "=" * 60)
    print("程序执行完毕！")
    print("=" * 60)


if __name__ == "__main__":
    main()
