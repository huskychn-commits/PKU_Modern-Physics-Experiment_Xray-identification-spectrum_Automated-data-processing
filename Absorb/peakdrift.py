#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制峰偏移图
功能：
1. 读取absorb_processed_data.json数据
2. 计算每个元素的峰偏移：衰减后峰位置 - 衰减前峰位置（0个衰减片对应的峰位置）
3. 绘制峰偏移图：横轴是衰减片数目、纵轴是峰偏移
4. 每个元素画一条线，所有元素画在同一个图中
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

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


def calculate_peak_drift(data_dict):
    """
    计算每个元素的峰偏移
    
    参数:
    data_dict: 数据字典
    
    返回:
    drift_data: 峰偏移数据字典，键为元素名，值为(层数列表, 峰偏移列表)
    zero_layer_positions: 0层衰减片对应的峰位置字典
    """
    drift_data = {}
    zero_layer_positions = {}
    
    # 需要排除的元素
    excluded_elements = ['Fe', 'Ti']
    
    for element_name, element_data in data_dict.items():
        # 排除Fe和Ti元素
        if element_name in excluded_elements:
            print(f"跳过元素: {element_name} (根据要求排除)")
            continue
            
        # 提取0层衰减片的峰位置
        zero_layer_pos = None
        layers_list = []
        peak_positions = []
        
        for row in element_data:
            layers = row[0]
            peak_position = row[3]  # 峰位置是第4个分量
            
            layers_list.append(layers)
            peak_positions.append(peak_position)
            
            if layers == 0:
                zero_layer_pos = peak_position
        
        if zero_layer_pos is None:
            print(f"警告: 元素 {element_name} 没有0层衰减片的数据")
            continue
        
        zero_layer_positions[element_name] = zero_layer_pos
        
        # 计算峰偏移：衰减后峰位置 - 衰减前峰位置
        peak_drifts = [pos - zero_layer_pos for pos in peak_positions]
        
        # 按层数排序
        sorted_data = sorted(zip(layers_list, peak_drifts))
        sorted_layers = [item[0] for item in sorted_data]
        sorted_drifts = [item[1] for item in sorted_data]
        
        drift_data[element_name] = (sorted_layers, sorted_drifts)
        
        print(f"元素 {element_name}: 0层峰位置 = {zero_layer_pos:.4f}, 层数范围 = {min(layers_list)}-{max(layers_list)}")
    
    return drift_data, zero_layer_positions


def plot_peak_drift(drift_data, zero_layer_positions):
    """
    绘制峰偏移图
    
    参数:
    drift_data: 峰偏移数据字典
    zero_layer_positions: 0层衰减片对应的峰位置字典
    """
    if not drift_data:
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
    colors = plt.cm.tab20(np.linspace(0, 1, len(drift_data)))
    markers = ['o', 's', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
    
    # 绘制每个元素的峰偏移线
    for i, (element_name, (layers, drifts)) in enumerate(drift_data.items()):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        
        # 绘制线图
        plt.plot(layers, drifts, marker=marker, color=color, linewidth=2, 
                 markersize=8, label=element_name, alpha=0.8)
    
    # 设置坐标轴标签
    plt.xlabel('衰减片层数', fontsize=14, fontweight='bold')
    plt.ylabel('峰偏移 (衰减后峰位置 - 衰减前峰位置)', fontsize=14, fontweight='bold')
    
    # 设置标题
    plt.title('X射线标识谱 - 峰偏移随衰减片层数变化', fontsize=16, fontweight='bold')
    
    # 添加网格
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 添加图例
    plt.legend(loc='best', fontsize=11, ncol=2)
    
    # 添加零线（y=0）
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    output_image = os.path.join(os.path.dirname(os.path.abspath(__file__)), "peak_drift_plot.png")
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"\n图像已保存到: {output_image}")
    
    # 显示图像
    plt.show()




def calculate_linear_fit_slope(layers, drifts):
    """
    计算线性拟合斜率（强制截距b=0，即正比例拟合 y = kx）
    
    参数:
    layers: 层数列表
    drifts: 峰偏移列表
    
    返回:
    slope: 拟合斜率k
    """
    # 转换为numpy数组
    x = np.array(layers, dtype=float)
    y = np.array(drifts, dtype=float)
    
    # 强制截距为0的正比例拟合：y = kx
    # 最小二乘法：k = Σ(xy) / Σ(x²)
    if np.sum(x**2) == 0:
        return 0.0
    
    k = np.sum(x * y) / np.sum(x**2)
    return k


def print_drift_summary(drift_data, zero_layer_positions):
    """
    打印峰偏移摘要
    
    参数:
    drift_data: 峰偏移数据字典
    zero_layer_positions: 0层衰减片对应的峰位置字典
    """
    print("\n" + "=" * 80)
    print("峰偏移数据摘要:")
    print("=" * 80)
    
    # 存储线性拟合斜率用于表格
    slope_data = []
    
    for element_name, (layers, drifts) in drift_data.items():
        zero_pos = zero_layer_positions.get(element_name, 0)
        
        # 计算线性拟合斜率（强制截距b=0）
        slope = calculate_linear_fit_slope(layers, drifts)
        slope_data.append((element_name, slope))
        
        print(f"\n元素: {element_name}")
        print(f"  0层峰位置: {zero_pos:.4f}")
        print(f"  层数范围: {min(layers)} - {max(layers)}")
        print(f"  峰偏移范围: {min(drifts):.4f} - {max(drifts):.4f}")
        print(f"  平均峰偏移: {np.mean(drifts):.4f}")
        print(f"  最大峰偏移: {max(drifts, key=abs):.4f} (在 {layers[drifts.index(max(drifts, key=abs))]} 层)")
        print(f"  线性拟合斜率 (y = kx): {slope:.6f}")
        
        # 打印详细数据
        print("  详细数据:")
        for layer, drift in zip(layers, drifts):
            print(f"    层数 {layer}: 峰偏移 = {drift:.4f}")
    
    # 打印线性拟合斜率表格
    print("\n" + "=" * 80)
    print("线性拟合斜率表格 (强制截距b=0，拟合方程: y = kx):")
    print("=" * 80)
    print(f"{'元素':<8} {'斜率(k)':<15} {'说明':<30}")
    print("-" * 60)
    
    # 计算所有斜率的平均值
    all_slopes = [slope for _, slope in slope_data]
    avg_slope = np.mean(all_slopes) if all_slopes else 0.0
    
    for element_name, slope in slope_data:
        # 根据斜率值添加说明
        if abs(slope) < 0.001:
            description = "几乎无变化"
        elif slope > 0:
            description = "随层数增加而增加"
        else:
            description = "随层数增加而减小"
        
        print(f"{element_name:<8} {slope:<15.6f} {description:<30}")
    
    print("-" * 60)
    # 添加平均斜率行
    avg_description = "所有元素平均" if avg_slope != 0 else "平均"
    print(f"{'平均':<8} {avg_slope:<15.6f} {avg_description:<30}")
    print("-" * 60)
    print(f"总计: {len(slope_data)} 个元素")


def main():
    """
    主函数
    """
    print("=" * 60)
    print("峰偏移图绘制程序")
    print("=" * 60)
    
    # 1. 加载数据
    print("1. 加载处理后的数据...")
    data_dict = load_processed_data()
    
    if data_dict is None:
        return
    
    print(f"   加载了 {len(data_dict)} 个元素的数据")
    
    # 2. 计算峰偏移
    print("\n2. 计算峰偏移...")
    drift_data, zero_layer_positions = calculate_peak_drift(data_dict)
    
    if not drift_data:
        print("错误: 无法计算峰偏移")
        return
    
    print(f"   计算了 {len(drift_data)} 个元素的峰偏移")
    
    # 3. 打印摘要
    print_drift_summary(drift_data, zero_layer_positions)
    
    # 4. 绘制综合峰偏移图（所有元素在一起）
    print("\n3. 绘制综合峰偏移图（所有元素在一起）...")
    plot_peak_drift(drift_data, zero_layer_positions)
    
    print("\n" + "=" * 60)
    print("绘制完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
