#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
绘制核电荷数（原子序数）与Kα线能量的关系图
显示数据库实验线、非相对论理论线和相对论理论线
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# 添加当前目录到路径，以便导入xray_element_tool模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from xray_element_tool import ElementDatabase
except ImportError:
    print("错误: 无法导入xray_element_tool模块")
    print("请确保xray_element_tool.py在同一目录下")
    sys.exit(1)

# 设置中文字体支持
def setup_chinese_font():
    """设置matplotlib中文字体支持"""
    try:
        # 尝试使用系统中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        return True
    except:
        return False

# 初始化中文字体
chinese_font_available = setup_chinese_font()
if not chinese_font_available:
    print("警告: 中文字体不可用，图形中的中文可能显示为方框")

# 全局不透明度设置
ALPHA_MULTIPLIER = 0.8


def collect_element_data(db):
    """
    收集所有元素的核电荷数和Kα线能量数据
    
    参数:
    db: ElementDatabase实例
    
    返回:
    tuple: (Z_list, exp_list, nonrel_list, rel_list)
      Z_list: 原子序数列表
      exp_list: 实验Kα线能量列表（keV）
      nonrel_list: 非相对论理论Kα线能量列表（keV）
      rel_list: 相对论理论Kα线能量列表（keV）
    """
    Z_list = []
    exp_list = []
    nonrel_list = []
    rel_list = []
    
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
    
    return Z_list, exp_list, nonrel_list, rel_list


def plot_kalpha_on_axis(ax, db, Z_list, exp_list, nonrel_list, rel_list):
    """
    在指定的axis上绘制核电荷数与Kα线能量的关系图
    
    参数:
    ax: matplotlib的axis对象
    db: ElementDatabase实例
    Z_list: 原子序数列表
    exp_list: 实验Kα线能量列表（keV）
    nonrel_list: 非相对论理论Kα线能量列表（keV）
    rel_list: 相对论理论Kα线能量列表（keV）
    """
    # 准备数据
    Z_array = np.array(Z_list)
    
    # 分离有实验数据和无实验数据的点
    exp_Z = []
    exp_E = []
    no_exp_Z = []
    no_exp_E = []
    
    for i, (Z, exp) in enumerate(zip(Z_list, exp_list)):
        if exp is not None:
            exp_Z.append(Z)
            exp_E.append(exp)
        else:
            no_exp_Z.append(Z)
            no_exp_E.append(rel_list[i])  # 使用相对论值作为参考
    
    # 绘制实验数据点（有实验值的）- 使用橙色，不那么显眼
    if exp_Z:
        ax.scatter(exp_Z, exp_E, color='orange', s=50, 
                  label='实验数据点', zorder=5, alpha=0.7 * ALPHA_MULTIPLIER)
    
    # 绘制非实验数据点（无实验值的）
    if no_exp_Z:
        ax.scatter(no_exp_Z, no_exp_E, color='gray', s=30, 
                  label='无实验数据元素', zorder=4, alpha=0.5 * ALPHA_MULTIPLIER)
    
    # 绘制非相对论理论线
    # 对原子序数进行排序以便绘制平滑曲线
    sorted_indices = np.argsort(Z_array)
    Z_sorted = Z_array[sorted_indices]
    nonrel_sorted = np.array(nonrel_list)[sorted_indices]
    rel_sorted = np.array(rel_list)[sorted_indices]
    
    # 绘制理论线，应用不透明度
    ax.plot(Z_sorted, nonrel_sorted, 'b-', linewidth=2, 
           label='非相对论理论线', zorder=3, alpha=ALPHA_MULTIPLIER)
    ax.plot(Z_sorted, rel_sorted, 'g-', linewidth=2, 
           label='相对论理论线', zorder=2, alpha=ALPHA_MULTIPLIER)
    
    # 添加图例
    ax.legend(loc='upper left', fontsize=12)
    
    # 设置坐标轴标签
    ax.set_xlabel('核电荷数（原子序数 Z）', fontsize=14)
    ax.set_ylabel('Kα线能量（keV）', fontsize=14)
    
    # 设置标题
    ax.set_title('核电荷数与Kα线能量的关系', fontsize=16, fontweight='bold')
    
    # 添加网格
    ax.grid(True, alpha=0.3 * ALPHA_MULTIPLIER, linestyle='--')
    
    # 设置坐标轴范围
    ax.set_xlim(min(Z_list) - 1, max(Z_list) + 1)
    ax.set_ylim(0, max(rel_list) * 1.1)
    
    # 显示元素符号标签（每隔5个元素显示一个）
    for i, Z in enumerate(Z_list):
        if i % 5 == 0:  # 每隔5个元素显示一个标签
            element_info = db.get_element_by_atomic_number(Z)
            if element_info:
                symbol = element_info['symbol']
                # 在理论线上方添加标签
                ax.text(Z, rel_list[i] * 1.02, symbol, fontsize=8,
                       ha='center', va='bottom', rotation=45, 
                       alpha=0.7 * ALPHA_MULTIPLIER)
    
    return ax


def plot_kalpha_vs_z(db, Z_list, exp_list, nonrel_list, rel_list):
    """
    绘制核电荷数与Kα线能量的关系图（兼容旧版本）
    
    参数:
    db: ElementDatabase实例
    Z_list: 原子序数列表
    exp_list: 实验Kα线能量列表（keV）
    nonrel_list: 非相对论理论Kα线能量列表（keV）
    rel_list: 相对论理论Kα线能量列表（keV）
    """
    # 创建图形和axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 在新的axis上绘图
    plot_kalpha_on_axis(ax, db, Z_list, exp_list, nonrel_list, rel_list)
    
    # 调整布局
    plt.tight_layout()
    
    # 显示图形
    print("正在显示图形...")
    plt.show()


def generate_figures(output_dir=None, image_name=None):
    """
    生成核电荷数与Kα线能量关系图并保存到指定目录
    
    参数:
    output_dir: 输出目录，如果为None则保存到脚本所在目录
    image_name: 图片名称，如果为None则使用默认名称
    """
    print("=" * 60)
    print("核电荷数与Kα线能量关系图绘制程序 - 生成图片")
    print("=" * 60)
    
    # 初始化元素数据库
    print("初始化元素数据库...")
    db = ElementDatabase()
    
    # 收集数据
    print("收集元素数据...")
    Z_list, exp_list, nonrel_list, rel_list = collect_element_data(db)
    
    print(f"成功收集 {len(Z_list)} 个元素的数据")
    print(f"其中有实验数据的元素: {sum(1 for exp in exp_list if exp is not None)} 个")
    
    # 显示一些示例数据
    print("\n示例数据（前10个元素）:")
    print("原子序数 | 实验值(keV) | 非相对论理论值(keV) | 相对论理论值(keV)")
    print("-" * 70)
    for i in range(min(10, len(Z_list))):
        exp_str = f"{exp_list[i]:.3f}" if exp_list[i] is not None else "N/A"
        print(f"{Z_list[i]:^9} | {exp_str:^12} | {nonrel_list[i]:^18.3f} | {rel_list[i]:^18.3f}")
    
    # 生成图片
    generated_images = []
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 在新的axis上绘图
    plot_kalpha_on_axis(ax, db, Z_list, exp_list, nonrel_list, rel_list)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    if output_dir:
        if image_name:
            output_filename = os.path.join(output_dir, f"{image_name}.png")
        else:
            output_filename = os.path.join(output_dir, "相对论非相对论理论对比.png")
    else:
        if image_name:
            output_filename = f"{image_name}.png"
        else:
            output_filename = "相对论非相对论理论对比.png"
    
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    generated_images.append(output_filename)
    print(f"图片已保存到: {output_filename}")
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
    
    print("程序执行完毕！")


if __name__ == "__main__":
    main()
