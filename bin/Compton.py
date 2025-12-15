"""
Compton.py - 计算康普顿散射中电子吸收的能量
对于17keV的光子撞击静止电子
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

def compton_electron_energy(E0_keV, theta):
    """
    计算康普顿散射中电子吸收的能量
    
    参数:
    E0_keV: 光子初始能量 (keV)
    theta: 散射角度 (弧度)
    
    返回:
    E_electron: 电子吸收的能量 (keV)
    """
    # 电子静止能量 (keV)
    m_e_c2 = 511.0  # keV
    
    # 康普顿散射公式: 散射后光子能量
    # E' = E0 / [1 + (E0/(m_e c^2)) * (1 - cosθ)]
    E_prime = E0_keV / (1 + (E0_keV / m_e_c2) * (1 - np.cos(theta)))
    
    # 电子吸收的能量 = 初始能量 - 散射后光子能量
    E_electron = E0_keV - E_prime
    
    return E_electron

def electron_recoil_angle(E0_keV, theta):
    """
    计算电子反冲角（电子出射角）
    
    参数:
    E0_keV: 光子初始能量 (keV)
    theta: 光子散射角度 (弧度)
    
    返回:
    phi: 电子反冲角 (弧度)
    """
    # 电子静止能量 (keV)
    m_e_c2 = 511.0  # keV
    
    # 康普顿散射电子反冲角公式:
    # cot(φ) = (1 + E0/(m_e c^2)) * tan(θ/2)
    # 注意处理θ=0和θ=π的情况
    if theta == 0:
        # θ=0时，φ=π/2（电子垂直出射）
        return np.pi / 2
    elif theta == np.pi:
        # θ=π时，φ=0（电子前向出射）
        return 0.0
    else:
        cot_phi = (1 + E0_keV / m_e_c2) * np.tan(theta / 2)
        phi = np.arctan(1.0 / cot_phi)
        return phi

def compton_differential_cross_section(E0_keV, theta, dimensional=True):

    """
    计算康普顿散射的微分散射截面（Klein-Nishina公式）
    
    参数:
    E0_keV: 光子初始能量 (keV)
    theta: 散射角度 (弧度)
    dimensional: 是否返回有量纲的值（True: cm²/sr, False: 相对值）
    
    返回:
    dsigma_dOmega: 微分散射截面 (cm²/sr 或 相对值)
    """
    # 电子静止能量 (keV)
    m_e_c2 = 511.0  # keV
    
    # 散射后光子能量
    E_prime = E0_keV / (1 + (E0_keV / m_e_c2) * (1 - np.cos(theta)))
    
    # 能量比
    ratio = E_prime / E0_keV
    
    # Klein-Nishina公式（忽略常数因子r0^2/2）
    dsigma_dOmega_rel = ratio**2 * (ratio + 1/ratio - np.sin(theta)**2)
    
    if dimensional:
        # 经典电子半径 r0 = 2.818e-13 cm
        r0 = 2.818e-13  # cm
        # 有量纲的微分散射截面: (r0^2/2) * dsigma_dOmega_rel
        dsigma_dOmega = (r0**2 / 2) * dsigma_dOmega_rel  # cm²/sr
        return dsigma_dOmega
    else:
        return dsigma_dOmega_rel
    

def generate_figures(output_dir=None, image_name=None):
    """
    生成康普顿散射图片并保存到指定目录
    
    参数:
    output_dir: 输出目录，如果为None则保存到脚本所在目录
    image_name: 图片名称，如果为None则使用默认名称
    """
    # 光子初始能量 (keV)
    E0 = 17.0  # keV
    
    # 生成角度数组：从-π到π，共1000个点（用于笛卡尔坐标图）
    theta_cartesian = np.linspace(-np.pi, np.pi, 1000)
    
    # 计算电子吸收的能量（笛卡尔坐标）- 作为光子散射角θ的函数
    E_electron_theta = compton_electron_energy(E0, theta_cartesian)
    
    # 计算电子反冲角φ（电子出射角）
    phi_cartesian = np.array([electron_recoil_angle(E0, theta) for theta in theta_cartesian])
    
    # 对于极坐标图，需要0到2π的范围
    # 由于函数是对称的：E(θ) = E(-θ)，我们可以将-π到0的部分映射到π到2π
    theta_polar = np.linspace(0, 2*np.pi, 1000)
    # 创建对应的能量数组：0到π使用正角度，π到2π使用负角度的绝对值
    E_electron_theta_polar = np.zeros_like(theta_polar)
    phi_polar = np.zeros_like(theta_polar)
    for i, angle in enumerate(theta_polar):
        if angle <= np.pi:
            # 0到π：直接使用角度
            theta_val = angle
        else:
            # π到2π：使用负角度（对称性）
            theta_val = angle - 2*np.pi
        
        E_electron_theta_polar[i] = compton_electron_energy(E0, theta_val)
        phi_polar[i] = electron_recoil_angle(E0, theta_val)
        
        # 对于极坐标图，我们需要将φ映射到0到2π范围
        if phi_polar[i] < 0:
            phi_polar[i] += 2*np.pi
    
    # 计算康普顿散射的微分散射截面（有量纲，单位：cm²/sr）
    dsigma_theta = np.array([compton_differential_cross_section(E0, theta, dimensional=True) for theta in theta_cartesian])
    
    # 对于极坐标图的微分散射截面（有量纲）
    dsigma_polar = np.zeros_like(theta_polar)
    for i, angle in enumerate(theta_polar):
        if angle <= np.pi:
            # 0到π：直接使用角度
            theta_val = angle
        else:
            # π到2π：使用负角度（对称性）
            theta_val = angle - 2*np.pi
        dsigma_polar[i] = compton_differential_cross_section(E0, theta_val, dimensional=True)
    
    # 计算总截面sigma_tot（数值积分，有量纲）
    # 总截面 = 2π ∫₀^π (dσ/dΩ) sinθ dθ
    theta_for_integration = np.linspace(0, np.pi, 1000)
    dsigma_for_integration = np.array([compton_differential_cross_section(E0, theta, dimensional=True) for theta in theta_for_integration])
    
    # 使用梯形法进行数值积分
    sigma_tot = 2 * np.pi * np.trapz(dsigma_for_integration * np.sin(theta_for_integration), theta_for_integration)
    
    # 计算能量贡献：E_electron * (dσ/dΩ) / sigma_total
    # 注意：这里dσ/dΩ和sigma_tot都是有量纲的，但比例是无量纲的
    energy_contribution_theta = E_electron_theta * dsigma_theta / sigma_tot
    
    # 对于极坐标图的能量贡献
    energy_contribution_polar = np.zeros_like(theta_polar)
    for i, angle in enumerate(theta_polar):
        if angle <= np.pi:
            theta_val = angle
        else:
            theta_val = angle - 2*np.pi
        E_electron_val = compton_electron_energy(E0, theta_val)
        dsigma_val = compton_differential_cross_section(E0, theta_val, dimensional=True)
        energy_contribution_polar[i] = E_electron_val * dsigma_val / sigma_tot
    
    # 计算电子获得的平均能量
    # 平均能量 = ∫ E(θ) * (dσ/dΩ) dΩ / σ_tot = ∫ energy_contribution dΩ
    # 由于energy_contribution已经除以了σ_tot，我们需要重新计算未归一化的值
    # 平均能量 = ∫ E(θ) * (dσ/dΩ) dΩ / σ_tot
    # 使用数值积分计算分子：∫ E(θ) * (dσ/dΩ) sinθ dθ dφ
    # 由于对称性，对φ积分得到2π
    theta_for_avg = np.linspace(0, np.pi, 1000)
    E_for_avg = np.array([compton_electron_energy(E0, theta) for theta in theta_for_avg])
    dsigma_for_avg = np.array([compton_differential_cross_section(E0, theta, dimensional=True) for theta in theta_for_avg])
    
    # 计算积分：2π ∫₀^π E(θ) * (dσ/dΩ) sinθ dθ
    integral_numerator = 2 * np.pi * np.trapz(E_for_avg * dsigma_for_avg * np.sin(theta_for_avg), theta_for_avg)
    
    # 平均能量 = 积分分子 / σ_tot
    average_electron_energy = integral_numerator / sigma_tot
    
    # 创建图形，两行两列，进一步增加高度以避免文字重叠
    fig = plt.figure(figsize=(14, 14))
    
    # 第一行：电子能量
    # 第一个子图：笛卡尔坐标 - 电子能量
    ax1 = fig.add_subplot(2, 2, 1)
    # 红色线：电子能量 vs 光子散射角θ
    ax1.plot(theta_cartesian, E_electron_theta, 'r-', linewidth=2, label='电子能量 vs 光子散射角θ')
    # 蓝色线：电子能量 vs 电子反冲角φ
    ax1.plot(phi_cartesian, E_electron_theta, 'b-', linewidth=2, label='电子能量 vs 电子反冲角φ')
    ax1.set_xlabel('角度 (弧度)', fontsize=12)
    ax1.set_ylabel('电子吸收的能量 (keV)', fontsize=12)
    ax1.set_title(f'康普顿散射：电子能量随角度变化\n(光子初始能量 E0 = {E0} keV)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-np.pi, np.pi)
    ax1.legend(loc='upper right', fontsize=10)
    
    # 在图上标注重要角度
    important_angles = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
    angle_labels = ['-π', '-π/2', '0', 'π/2', 'π']
    for angle, label in zip(important_angles, angle_labels):
        ax1.axvline(x=angle, color='gray', linestyle='--', alpha=0.5)
        ax1.text(angle, max(E_electron_theta)*0.95, label, 
                horizontalalignment='center', fontsize=10)
    
    # 第二个子图：极坐标 - 电子能量
    ax2 = fig.add_subplot(2, 2, 2, projection='polar')
    # 红色线：电子能量 vs 光子散射角θ（极坐标）
    ax2.plot(theta_polar, E_electron_theta_polar, 'r-', linewidth=2, label='电子能量 vs 光子散射角θ')
    # 蓝色线：电子能量 vs 电子反冲角φ（极坐标）
    ax2.plot(phi_polar, E_electron_theta_polar, 'b-', linewidth=2, label='电子能量 vs 电子反冲角φ')
    ax2.set_title(f'康普顿散射：电子能量极坐标图\n(光子初始能量 E0 = {E0} keV)', fontsize=14, pad=20)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1.3, 1.0))
    
    # 设置极坐标的角度标签（0到2π，只使用弧度）
    # 注意：0和2π在同一个位置，所以不显示2π标签
    ax2.set_xticks(np.arange(0, 2*np.pi, np.pi/4))
    ax2.set_xticklabels(['0', 'π/4', 'π/2', '3π/4', 'π', '5π/4', '3π/2', '7π/4'])
    # 移除r方向刻度线
    ax2.set_rgrids([])
    
    # 第二行：能量贡献和微分散射截面（双y轴）
    # 第三个子图：笛卡尔坐标 - 能量贡献和微分散射截面
    ax3 = fig.add_subplot(2, 2, 3)
    
    # 创建双y轴
    ax3_left = ax3
    ax3_right = ax3.twinx()
    
    # 红色线：能量贡献 vs 光子散射角θ（左y轴）
    line1, = ax3_left.plot(theta_cartesian, energy_contribution_theta, 'r-', linewidth=2, label='能量贡献 (E·dσ/dΩ/σ_tot)')
    ax3_left.set_xlabel('光子散射角 θ (弧度)', fontsize=12)
    ax3_left.set_ylabel('能量贡献 (keV)', fontsize=12, color='black')
    ax3_left.tick_params(axis='y', labelcolor='black')
    ax3_left.grid(True, alpha=0.3)
    ax3_left.set_xlim(-np.pi, np.pi)
    
    # 绿色线：微分散射截面 vs 光子散射角θ（右y轴，有量纲）
    line2, = ax3_right.plot(theta_cartesian, dsigma_theta, 'g-', linewidth=2, label='微分散射截面')
    ax3_right.set_ylabel('微分散射截面 (cm^2/sr)', fontsize=12, color='black')
    ax3_right.tick_params(axis='y', labelcolor='black')
    
    # 设置标题，总截面使用科学计数法
    ax3.set_title(f'康普顿散射：能量贡献和微分散射截面\n(光子初始能量 E0 = {E0} keV, σ_tot = {sigma_tot:.2e} cm^2)', fontsize=14)
    
    # 合并图例
    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax3.legend(lines, labels, loc='upper right', fontsize=10)
    
    # 在图上标注重要角度
    for angle, label in zip(important_angles, angle_labels):
        ax3.axvline(x=angle, color='gray', linestyle='--', alpha=0.5)
        ax3.text(angle, max(energy_contribution_theta)*0.95, label, 
                horizontalalignment='center', fontsize=10)
    
    # 第四个子图：极坐标 - 能量贡献和微分散射截面
    ax4 = fig.add_subplot(2, 2, 4, projection='polar')
    
    # 红色线：能量贡献 vs 光子散射角θ（极坐标）
    line3, = ax4.plot(theta_polar, energy_contribution_polar, 'r-', linewidth=2, label='能量贡献')
    
    # 创建第二个极坐标轴用于微分散射截面
    # 注意：matplotlib极坐标图不支持双y轴，我们使用不同的缩放比例
    # 将微分散射截面缩放到与能量贡献相近的范围以便显示
    dsigma_scaled = dsigma_polar * (max(energy_contribution_polar) / max(dsigma_polar)) * 0.8
    
    # 绿色线：微分散射截面 vs 光子散射角θ（极坐标，缩放后）
    line4, = ax4.plot(theta_polar, dsigma_scaled, 'g-', linewidth=2, label='微分散射截面（缩放）')
    
    ax4.set_title(f'康普顿散射：能量贡献和微分散射截面（极坐标）\n(光子初始能量 E0 = {E0} keV, σ_tot = {sigma_tot:.2e} cm^2)', fontsize=14, pad=20)
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right', fontsize=10, bbox_to_anchor=(1.3, 1.0))
    
    # 设置极坐标的角度标签（0到2π，只使用弧度）
    # 注意：0和2π在同一个位置，所以不显示2π标签
    ax4.set_xticks(np.arange(0, 2*np.pi, np.pi/4))
    ax4.set_xticklabels(['0', 'π/4', 'π/2', '3π/4', 'π', '5π/4', '3π/2', '7π/4'])
    # 移除r方向刻度线
    ax4.set_rgrids([])
    
    # 调整布局，进一步增加行间距以避免文字重叠
    plt.subplots_adjust(hspace=0.4)
    
    # 保存图形
    if output_dir:
        if image_name:
            output_filename = os.path.join(output_dir, f"{image_name}.png")
        else:
            output_filename = os.path.join(output_dir, "Compton_scattering_plots.png")
    else:
        if image_name:
            output_filename = f"{image_name}.png"
        else:
            output_filename = "Compton_scattering_plots.png"
    
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"康普顿散射图形已保存到: {output_filename}")
    
    # 打印一些关键值
    print(f"光子初始能量: {E0} keV")
    print(f"电子静止能量: 511.0 keV")
    print(f"\n最大电子能量 (θ=π): {compton_electron_energy(E0, np.pi):.4f} keV")
    print(f"电子获得的平均能量: {average_electron_energy:.4f} keV")
    
    return [output_filename]


def main():
    """主函数：计算并绘制康普顿散射电子能量随角度的变化"""
    # 调用generate_figures函数，不指定输出目录（保存到脚本所在目录）
    generated_images = generate_figures()
    
    # 显示图形
    if generated_images:
        print(f"\n共生成 {len(generated_images)} 个图片")
        for img in generated_images:
            print(f"  - {img}")
    
    # 打印一些额外信息
    E0 = 17.0  # keV
    print(f"\n--- 物理解释 ---")
    print("1. θ=0（前向散射）: 电子获得最小能量 (0 keV)")
    print("2. θ=π（背向散射）: 电子获得最大能量")
    print("3. 曲线关于θ=0对称")
    print("4. 对于17 keV光子，康普顿效应相对较小")
    
    # 计算百分比
    E_max = compton_electron_energy(E0, np.pi)
    percentage = (E_max / E0) * 100
    print(f"   电子获得能量占比: {percentage:.2f}%")


if __name__ == "__main__":
    main()
