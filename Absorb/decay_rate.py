#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
衰减率分析程序
功能：
1. 读取absorb_processed_data.json数据
2. 计算每个元素的事件数比值：事件数/无衰减事件数（0个衰减片的事件数）
3. 用exp[-kN]拟合这个比例（N是衰减片层数）
4. 绘制事件数比值随衰减片层数变化的图
5. 绘制衰减系数k vs 无衰减峰位置的图
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def load_processed_data():
    """
    加载处理后的数据
    
    返回:
    data_dict: 从JSON文件加载的数据字典
    """
    # 获取当前脚本的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, "absorb_processed_data.json")
    
    if not os.path.exists(data_file):
        print(f"错误: 数据文件不存在 - {data_file}")
        print("请先运行ReadData.py生成数据文件")
        return None
    
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
        print(f"成功加载数据文件: {data_file}")
        return data_dict
    except Exception as e:
        print(f"加载数据文件失败: {e}")
        return None


def calculate_event_ratios(data_dict):
    """
    计算每个元素的事件数比值：事件数/无衰减事件数
    
    参数:
    data_dict: 数据字典
    
    返回:
    ratio_data: 比值数据字典，键为元素名，值为(层数列表, 比值列表)
    zero_layer_events: 0层衰减片对应的事件数字典
    zero_layer_positions: 0层衰减片对应的峰位置字典
    """
    ratio_data = {}
    zero_layer_events = {}
    zero_layer_positions = {}
    
    for element_name, element_data in data_dict.items():
            
        # 提取数据
        layers_list = []
        events_list = []
        ratios_list = []
        zero_layer_event = None
        zero_layer_pos = None
        
        for row in element_data:
            layers = row[0]
            events = row[1]  # 总事件数是第2个分量
            peak_position = row[3]  # 峰位置是第4个分量
            
            layers_list.append(layers)
            events_list.append(events)
            
            if layers == 0:
                zero_layer_event = events
                zero_layer_pos = peak_position
        
        if zero_layer_event is None or zero_layer_event == 0:
            print(f"警告: 元素 {element_name} 没有0层衰减片的数据或0层事件数为0")
            continue
        
        zero_layer_events[element_name] = zero_layer_event
        zero_layer_positions[element_name] = zero_layer_pos
        
        # 计算比值：事件数/无衰减事件数
        ratios = [events / zero_layer_event for events in events_list]
        
        # 按层数排序
        sorted_data = sorted(zip(layers_list, ratios))
        sorted_layers = [item[0] for item in sorted_data]
        sorted_ratios = [item[1] for item in sorted_data]
        
        ratio_data[element_name] = (sorted_layers, sorted_ratios)
        
        print(f"元素 {element_name}: 0层事件数 = {zero_layer_event}, 0层峰位置 = {zero_layer_pos:.4f}")
    
    return ratio_data, zero_layer_events, zero_layer_positions


def exponential_decay_fit(N, k):
    """
    指数衰减函数：exp(-k*N)
    
    参数:
    N: 衰减片层数
    k: 衰减系数
    
    返回:
    y: 衰减后的比值
    """
    return np.exp(-k * N)


def fit_decay_coefficients(ratio_data):
    """
    对每个元素的比值数据进行指数衰减拟合：y = exp(-k*N)
    
    参数:
    ratio_data: 比值数据字典
    
    返回:
    decay_coefficients: 衰减系数字典，键为元素名，值为衰减系数k
    fit_errors: 拟合误差字典，键为元素名，值为拟合误差
    """
    decay_coefficients = {}
    fit_errors = {}
    
    for element_name, (layers, ratios) in ratio_data.items():
        # 转换为numpy数组
        N = np.array(layers, dtype=float)
        y = np.array(ratios, dtype=float)
        
        # 确保数据有效
        if len(N) < 2:
            print(f"警告: 元素 {element_name} 数据点不足，无法拟合")
            decay_coefficients[element_name] = 0.0
            fit_errors[element_name] = 0.0
            continue
        
        try:
            # 使用curve_fit进行指数衰减拟合
            # 初始猜测值：k=0.1
            popt, pcov = curve_fit(exponential_decay_fit, N, y, p0=[0.1], bounds=(0, np.inf))
            k = popt[0]
            perr = np.sqrt(np.diag(pcov))[0] if len(pcov) > 0 else 0.0
            
            decay_coefficients[element_name] = k
            fit_errors[element_name] = perr
            
            print(f"元素 {element_name}: 衰减系数 k = {k:.6f} ± {perr:.6f}")
            
        except Exception as e:
            print(f"元素 {element_name} 拟合失败: {e}")
            decay_coefficients[element_name] = 0.0
            fit_errors[element_name] = 0.0
    
    return decay_coefficients, fit_errors


def plot_event_ratios(ratio_data, decay_coefficients):
    """
    绘制事件数比值随衰减片层数变化的图
    
    参数:
    ratio_data: 比值数据字典
    decay_coefficients: 衰减系数字典
    """
    if not ratio_data:
        print("错误: 没有可绘制的数据")
        return
    
    # 设置中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass
    
    # 创建图形
    plt.figure(figsize=(14, 8))
    
    # 定义颜色和标记样式
    colors = plt.cm.tab20(np.linspace(0, 1, len(ratio_data)))
    markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
    
    # 按照衰减系数k从大到小排序元素
    sorted_elements = sorted(
        ratio_data.items(),
        key=lambda x: decay_coefficients.get(x[0], 0.0),
        reverse=True  # 从大到小排序
    )
    
    # 绘制每个元素的事件数比值（按照衰减率从大到小排序）
    for i, (element_name, (layers, ratios)) in enumerate(sorted_elements):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        # 绘制数据点（只画点，不连线）
        plt.plot(layers, ratios, marker=marker, color=color, linewidth=0, 
                 markersize=8, label=f'{element_name}', alpha=0.8)
        
        # 绘制拟合曲线
        k = decay_coefficients.get(element_name, 0.0)
        if k > 0:
            # 生成平滑的拟合曲线
            N_fit = np.linspace(min(layers), max(layers), 100)
            y_fit = exponential_decay_fit(N_fit, k)
            plt.plot(N_fit, y_fit, color=color, linestyle='-', linewidth=2, alpha=0.8)
    
    # 设置坐标轴标签
    plt.xlabel('衰减片层数 (N)', fontsize=14, fontweight='bold')
    plt.ylabel('事件数比值 (事件数/无衰减事件数)', fontsize=14, fontweight='bold')
    
    # 设置标题
    plt.title('X射线标识谱 - 事件数比值随衰减片层数变化 (拟合: y = exp(-kN))', fontsize=16, fontweight='bold')
    
    # 添加网格
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 添加图例
    plt.legend(loc='best', fontsize=11, ncol=2)
    
    # 设置y轴范围
    plt.ylim(0, 1.1)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    output_image = os.path.join(os.path.dirname(os.path.abspath(__file__)), "event_ratio_plot.png")
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"\n事件数比值图已保存到: {output_image}")
    
    # 关闭图形，避免阻塞
    plt.close()


def plot_decay_coefficients(decay_coefficients, zero_layer_positions, fit_errors):
    """
    绘制衰减系数k vs 无衰减峰位置的图
    
    参数:
    decay_coefficients: 衰减系数字典
    zero_layer_positions: 0层衰减片对应的峰位置字典
    fit_errors: 拟合误差字典
    """
    if not decay_coefficients:
        print("错误: 没有可绘制的数据")
        return
    
    # 设置中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass
    
    # 准备数据
    elements = []
    peak_positions = []
    k_values = []
    k_errors = []
    
    for element_name, k in decay_coefficients.items():
        if element_name in zero_layer_positions:
            elements.append(element_name)
            peak_positions.append(zero_layer_positions[element_name])
            k_values.append(k)
            k_errors.append(fit_errors.get(element_name, 0.0))
    
    if not elements:
        print("错误: 没有有效的数据点")
        return
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 绘制散点图
    plt.errorbar(peak_positions, k_values, yerr=k_errors, fmt='o', 
                 markersize=10, capsize=5, alpha=0.8, color='steelblue')
    
    # 添加元素标签
    for i, (element, x, y) in enumerate(zip(elements, peak_positions, k_values)):
        plt.annotate(element, (x, y), xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold', alpha=0.8)
    
    # 设置坐标轴标签
    plt.xlabel('谱线能量 (0层衰减片对应的峰位置)', fontsize=14, fontweight='bold')
    plt.ylabel('衰减系数 (拟合: y = exp(-kN))', fontsize=14, fontweight='bold')
    
    # 设置标题
    plt.title('X射线标识谱 - 衰减系数 vs 谱线能量', fontsize=16, fontweight='bold')
    
    # 添加网格
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    output_image = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decay_coefficient_plot.png")
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"\n衰减系数图已保存到: {output_image}")
    
    # 关闭图形
    plt.close()


def plot_log_log_relationship(decay_coefficients, zero_layer_positions, fit_errors):
    """
    绘制log-log图：横轴是log(谱线能量)，纵轴是log(衰减系数)
    根据经验定律：衰减系数 ∝ (谱线能量)^(-3)
    即：log(衰减系数) = -3 * log(谱线能量) + 常数
    
    参数:
    decay_coefficients: 衰减系数字典
    zero_layer_positions: 0层衰减片对应的峰位置字典
    fit_errors: 拟合误差字典
    """
    if not decay_coefficients:
        print("错误: 没有可绘制的数据")
        return
    
    # 设置中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass
    
    # 准备数据
    elements = []
    log_energies = []
    log_k_values = []
    log_k_errors = []
    
    for element_name, k in decay_coefficients.items():
        if element_name in zero_layer_positions:
            energy = zero_layer_positions[element_name]
            # 确保能量和衰减系数为正数
            if energy > 0 and k > 0:
                elements.append(element_name)
                log_energies.append(np.log10(energy))
                log_k_values.append(np.log10(k))
                # 误差传播：Δ(log10(k)) ≈ Δk / (k * ln(10))
                err = fit_errors.get(element_name, 0.0)
                if k > 0 and err > 0:
                    log_err = err / (k * np.log(10))
                else:
                    log_err = 0.0
                log_k_errors.append(log_err)
    
    if len(elements) < 2:
        print("错误: 没有足够的数据点进行log-log分析")
        return
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 绘制散点图
    plt.errorbar(log_energies, log_k_values, yerr=log_k_errors, fmt='o', 
                 markersize=10, capsize=5, alpha=0.8, color='coral')
    
    # 添加元素标签
    for i, (element, x, y) in enumerate(zip(elements, log_energies, log_k_values)):
        plt.annotate(element, (x, y), xytext=(5, 5), textcoords='offset points',
                    fontsize=10, fontweight='bold', alpha=0.8)
    
    # 线性拟合：log(k) = slope * log(E) + intercept
    log_E = np.array(log_energies)
    log_k = np.array(log_k_values)
    
    # 使用加权最小二乘法进行线性拟合（考虑误差）
    A = np.vstack([log_E, np.ones(len(log_E))]).T
    # 使用误差的倒数作为权重
    weights = 1.0 / np.array(log_k_errors) if np.all(np.array(log_k_errors) > 0) else None
    
    if weights is not None:
        # 加权最小二乘
        W = np.diag(weights)
        coeffs = np.linalg.lstsq(A.T @ W @ A, A.T @ W @ log_k, rcond=None)[0]
    else:
        # 普通最小二乘
        coeffs = np.linalg.lstsq(A, log_k, rcond=None)[0]
    
    slope = coeffs[0]
    intercept = coeffs[1]
    
    # 计算相关系数R
    k_pred = slope * log_E + intercept
    # 计算Pearson相关系数
    correlation_matrix = np.corrcoef(log_E, log_k)
    r_value = correlation_matrix[0, 1] if correlation_matrix.shape == (2, 2) else 0
    
    # 绘制拟合直线
    x_fit = np.linspace(min(log_E), max(log_E), 100)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, 'r-', linewidth=2, alpha=0.8, 
             label=f'拟合: log(k) = {slope:.3f}·log(E) + {intercept:.3f}\nR = {r_value:.4f}')
    
    # 绘制理论直线：log(k) = -3 * log(E) + 常数
    # 使用平均点确定理论直线的截距
    mean_log_E = np.mean(log_E)
    mean_log_k = np.mean(log_k)
    theoretical_intercept = mean_log_k - (-3) * mean_log_E
    y_theory = -3 * x_fit + theoretical_intercept
    plt.plot(x_fit, y_theory, 'b--', linewidth=2, alpha=0.6, 
             label='理论: log(k) = -3·log(E) + 常数')
    
    # 设置坐标轴标签
    plt.xlabel('log(谱线能量)', fontsize=14, fontweight='bold')
    plt.ylabel('log(衰减系数)', fontsize=14, fontweight='bold')
    
    # 设置标题
    plt.title('X射线标识谱 - log(衰减系数) vs log(谱线能量) 关系', fontsize=16, fontweight='bold')
    
    # 添加网格
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 添加图例
    plt.legend(loc='best', fontsize=12)
    
    # 添加说明文本
    slope_diff = abs(slope + 3)  # 斜率与-3的差异
    if slope_diff < 0.5:
        conclusion = "与理论值(-3)基本一致"
    elif slope_diff < 1.0:
        conclusion = "与理论值(-3)有一定偏差"
    else:
        conclusion = "与理论值(-3)差异较大"
    
    plt.text(0.02, 0.02, f'拟合斜率: {slope:.3f} (理论值: -3.000)\n{conclusion}', 
             transform=plt.gca().transAxes, fontsize=11,
             verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    output_image = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log_log_relationship_plot.png")
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"\nlog-log关系图已保存到: {output_image}")
    
    # 打印拟合结果
    print(f"\nlog-log关系分析结果:")
    print(f"  拟合斜率: {slope:.6f}")
    print(f"  拟合截距: {intercept:.6f}")
    print(f"  相关系数R: {r_value:.6f}")
    print(f"  理论斜率: -3.000")
    print(f"  斜率差异: {abs(slope + 3):.6f}")
    print(f"  结论: {conclusion}")
    
    # 关闭图形
    plt.close()


def save_decay_coefficients_csv(decay_coefficients, zero_layer_positions, fit_errors):
    """
    将衰减系数vs谱线能量的数据保存为CSV文件
    
    参数:
    decay_coefficients: 衰减系数字典
    zero_layer_positions: 0层衰减片对应的峰位置字典
    fit_errors: 拟合误差字典
    """
    if not decay_coefficients:
        print("错误: 没有可保存的数据")
        return
    
    # 准备数据
    data_rows = []
    for element_name, k in decay_coefficients.items():
        if element_name in zero_layer_positions:
            energy = zero_layer_positions[element_name]
            data_rows.append([element_name, energy, k])
    
    if not data_rows:
        print("错误: 没有有效的数据点")
        return
    
    # 按元素名字排序
    data_rows.sort(key=lambda x: x[0])
    
    # 保存为CSV文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(current_dir, "decay_coefficients.csv")
    
    try:
        # 使用UTF-8 with BOM编码确保中文正确显示
        with open(csv_file, 'w', encoding='utf-8-sig') as f:
            # 写入表头（根据用户要求：第零列是元素名字，第一列是谱线能量，第二列是衰减系数）
            f.write("元素,谱线能量,衰减系数\n")
            
            # 写入数据
            for row in data_rows:
                element_name, energy, k = row
                f.write(f"{element_name},{energy:.6f},{k:.6f}\n")
        
        print(f"\n衰减系数数据已保存到CSV文件: {csv_file}")
        print(f"  文件格式: 元素名字, 谱线能量, 衰减系数")
        print(f"  数据行数: {len(data_rows)}")
        
    except Exception as e:
        print(f"保存CSV文件失败: {e}")


def print_summary(ratio_data, decay_coefficients, zero_layer_positions, fit_errors):
    """
    打印分析摘要
    
    参数:
    ratio_data: 比值数据字典
    decay_coefficients: 衰减系数字典
    zero_layer_positions: 0层衰减片对应的峰位置字典
    fit_errors: 拟合误差字典
    """
    print("\n" + "=" * 80)
    print("衰减率分析摘要:")
    print("=" * 80)
    
    # 打印表格头
    print(f"{'元素':<8} {'0层峰位置':<15} {'衰减系数k':<15} {'拟合误差':<15} {'说明':<20}")
    print("-" * 80)
    
    for element_name in sorted(ratio_data.keys()):
        peak_pos = zero_layer_positions.get(element_name, 0.0)
        k = decay_coefficients.get(element_name, 0.0)
        err = fit_errors.get(element_name, 0.0)
        
        # 根据衰减系数添加说明
        if k < 0.01:
            description = "衰减很慢"
        elif k < 0.1:
            description = "中等衰减"
        else:
            description = "快速衰减"
        
        print(f"{element_name:<8} {peak_pos:<15.4f} {k:<15.6f} {err:<15.6f} {description:<20}")
    
    print("-" * 80)
    
    # 计算平均衰减系数
    if decay_coefficients:
        avg_k = np.mean(list(decay_coefficients.values()))
        print(f"平均衰减系数: {avg_k:.6f}")
        print(f"分析元素数量: {len(decay_coefficients)}")


def main():
    """
    主函数
    """
    print("=" * 60)
    print("衰减率分析程序")
    print("=" * 60)
    
    # 1. 加载数据
    print("1. 加载处理后的数据...")
    data_dict = load_processed_data()
    
    if data_dict is None:
        return
    
    print(f"   加载了 {len(data_dict)} 个元素的数据")
    
    # 2. 计算事件数比值
    print("\n2. 计算事件数比值...")
    ratio_data, zero_layer_events, zero_layer_positions = calculate_event_ratios(data_dict)
    
    if not ratio_data:
        print("错误: 无法计算事件数比值")
        return
    
    print(f"   计算了 {len(ratio_data)} 个元素的事件数比值")
    
    # 3. 进行指数衰减拟合
    print("\n3. 进行指数衰减拟合 (y = exp(-kN))...")
    decay_coefficients, fit_errors = fit_decay_coefficients(ratio_data)
    
    # 4. 打印摘要
    print_summary(ratio_data, decay_coefficients, zero_layer_positions, fit_errors)
    
    # 5. 绘制事件数比值图
    print("\n4. 绘制事件数比值图...")
    plot_event_ratios(ratio_data, decay_coefficients)
    
    # 6. 绘制衰减系数k vs 无衰减峰位置图
    print("\n5. 绘制衰减系数k vs 无衰减峰位置图...")
    plot_decay_coefficients(decay_coefficients, zero_layer_positions, fit_errors)
    
    # 7. 保存衰减系数数据到CSV文件
    print("\n6. 保存衰减系数数据到CSV文件...")
    save_decay_coefficients_csv(decay_coefficients, zero_layer_positions, fit_errors)
    
    # 8. 绘制log-log关系图
    print("\n7. 绘制log-log关系图 (验证衰减系数 ∝ 能量^{-3})...")
    plot_log_log_relationship(decay_coefficients, zero_layer_positions, fit_errors)
    
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
