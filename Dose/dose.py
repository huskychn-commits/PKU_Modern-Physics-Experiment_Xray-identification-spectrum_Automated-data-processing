#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
剂量计算程序
功能：
1. 从多个.out文件中读取数据（Cs137.out, Pu238.out, ionAm241.out）
2. 计算人体总质量
3. 计算总吸收能量
4. 计算加权平均剂量（Gy）
5. 比较不同放射源的结果
6. 输出结果
"""

import os
import re
import datetime

def read_out_file(file_path):
    """
    读取.out文件，提取事件数和器官数据
    
    参数:
    file_path: .out文件路径
    
    返回:
    num_events: 事件数
    organ_data: 器官数据列表，每个元素为[organ_id, mass_g, dose_Gy_per_source, relative_error]
    """
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 - {file_path}")
        return None, None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        num_events = None
        organ_data = []
        in_data_section = False
        
        for line in lines:
            line = line.strip()
            
            # 提取事件数
            if "Number of event processed" in line:
                match = re.search(r'Number of event processed : (\d+)', line)
                if match:
                    num_events = int(match.group(1))
                    print(f"读取到事件数: {num_events}")
            
            # 检测数据部分开始
            if "organ ID|" in line and "Organ Mass (g)" in line and "Dose (Gy/source)" in line:
                in_data_section = True
                continue
            
            # 检测数据部分结束
            if line.startswith("===") and in_data_section:
                in_data_section = False
                continue
            
            # 解析数据行
            if in_data_section and line and "|" in line:
                # 数据格式：器官ID| 质量(g) 剂量(Gy/source) 相对误差
                # 例如：100|               8.683          2.411e-19              1.000
                
                # 首先分割竖线部分
                line_parts = line.split('|')
                if len(line_parts) >= 2:
                    try:
                        # 第一部分是器官ID
                        organ_id = int(line_parts[0].strip())
                        
                        # 第二部分包含质量、剂量和误差，用空格分割
                        data_part = line_parts[1].strip()
                        # 使用正则表达式分割多个空格
                        data_values = re.split(r'\s+', data_part)
                        
                        # 过滤空字符串
                        data_values = [v for v in data_values if v]
                        
                        if len(data_values) >= 3:
                            mass_g = float(data_values[0])
                            dose_Gy = float(data_values[1])
                            relative_error = float(data_values[2])
                            
                            organ_data.append([organ_id, mass_g, dose_Gy, relative_error])
                            
                            # 调试：打印前10行数据
                            if len(organ_data) <= 10:
                                print(f"  解析成功: ID={organ_id}, 质量={mass_g}g, 剂量={dose_Gy}Gy, 误差={relative_error}")
                        else:
                            print(f"警告: 数据值不足 {len(data_values)} 个: {line}")
                            
                    except ValueError as e:
                        print(f"警告: 无法解析行: {line}")
                        print(f"错误: {e}")
                        continue
                else:
                    print(f"警告: 行格式不正确，缺少竖线分隔: {line}")
        
        if num_events is None:
            print("警告: 未找到事件数")
            num_events = 1000000  # 使用默认值
        
        return num_events, organ_data
    
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None, None


def calculate_dose_metrics(num_events, organ_data):
    """
    计算剂量指标
    
    参数:
    num_events: 事件数
    organ_data: 器官数据列表
    
    返回:
    total_mass_kg: 总质量 (kg)
    total_energy_J: 总吸收能量 (J)
    weighted_dose_Gy: 加权平均剂量 (Gy)
    """
    if not organ_data:
        print("错误: 没有器官数据")
        return 0.0, 0.0, 0.0
    
    total_mass_g = 0.0
    total_mass_kg = 0.0
    total_energy_J = 0.0
    
    print("\n器官数据统计:")
    print("-" * 80)
    print(f"{'器官ID':<10} {'质量(g)':<12} {'剂量(Gy/source)':<20} {'相对误差':<12}")
    print("-" * 80)
    
    for i, (organ_id, mass_g, dose_Gy, error) in enumerate(organ_data):
        total_mass_g += mass_g
        
        # 质量从g转换为kg
        mass_kg = mass_g / 1000.0
        
        # 计算该器官吸收的能量：E = m * D * N
        # 其中：m = 质量(kg), D = 剂量(Gy/source), N = 事件数
        # 1 Gy = 1 J/kg，所以能量单位是J
        energy_J = mass_kg * dose_Gy * num_events
        total_energy_J += energy_J
        
        # 每100个器官打印一次进度
        if i % 100 == 0 or i == len(organ_data) - 1:
            print(f"{organ_id:<10} {mass_g:<12.3f} {dose_Gy:<20.3e} {error:<12.3f}")
    
    total_mass_kg = total_mass_g / 1000.0
    
    # 计算加权平均剂量：总能量 / (总质量 * 事件数)
    if total_mass_kg > 0 and num_events > 0:
        weighted_dose_Gy = total_energy_J / (total_mass_kg * num_events)
    else:
        weighted_dose_Gy = 0.0
    
    return total_mass_kg, total_energy_J, weighted_dose_Gy


def main():
    """
    主函数 - 处理多个放射源数据文件
    """
    print("=" * 80)
    print("多放射源剂量计算程序")
    print("=" * 80)
    
    # 定义要处理的放射源文件
    sources = [
        {"name": "Cs-137", "filename": "Cs137.out"},
        {"name": "Pu-238", "filename": "Pu238.out"},
        {"name": "Am-241", "filename": "ionAm241.out"}
    ]
    
    # 存储每个放射源的结果
    results = []
    
    # 获取数据文件目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "..", "..", "Dose", "ICRP145_HumanPhantomsAir", "bin")
    
    # 处理每个放射源
    for source in sources:
        print(f"\n{'='*60}")
        print(f"处理放射源: {source['name']}")
        print(f"{'='*60}")
        
        # 1. 读取数据文件
        out_file = os.path.join(data_dir, source['filename'])
        out_file = os.path.abspath(out_file)
        
        print(f"1. 读取数据文件: {out_file}")
        num_events, organ_data = read_out_file(out_file)
        
        if num_events is None or organ_data is None:
            print(f"错误: 无法读取数据文件 {source['filename']}")
            continue
        
        print(f"   读取到 {len(organ_data)} 个器官的数据")
        
        # 2. 计算剂量指标
        print("\n2. 计算剂量指标...")
        total_mass_kg, total_energy_J, weighted_dose_Gy = calculate_dose_metrics(num_events, organ_data)
        
        # 3. 验证计算
        print("\n3. 验证计算...")
        # 方法1：通过总能量计算
        dose_from_energy = total_energy_J / (total_mass_kg * num_events)
        
        # 方法2：通过质量加权平均计算
        total_mass_weighted_dose = 0.0
        total_mass_g = 0.0
        
        for organ_id, mass_g, dose_Gy, error in organ_data:
            total_mass_weighted_dose += mass_g * dose_Gy
            total_mass_g += mass_g
        
        if total_mass_g > 0:
            dose_from_weighted_avg = total_mass_weighted_dose / total_mass_g
        else:
            dose_from_weighted_avg = 0.0
        
        # 检查两种方法是否一致
        if abs(dose_from_energy - dose_from_weighted_avg) < 1e-20:
            print("   ✓ 两种计算方法结果一致")
        else:
            print(f"   ⚠ 两种计算方法结果有微小差异: {abs(dose_from_energy - dose_from_weighted_avg):.2e} Gy/source")
        
        # 计算1居里活度下达到25毫希沃特所需时间
        # 1居里 = 3.7e10 Bq，25 mSv = 0.025 Sv ≈ 0.025 Gy（对于光子辐射）
        activity_Ci = 1.0  # 居里
        activity_Bq = activity_Ci * 3.7e10  # 贝可勒尔
        target_dose_Sv = 0.025  # 25毫希沃特
        target_dose_Gy = target_dose_Sv  # 对于光子辐射，1 Gy ≈ 1 Sv
        
        # 剂量率 = 每个源粒子的剂量 × 活度
        dose_rate_Gy_per_sec = weighted_dose_Gy * activity_Bq
        
        if dose_rate_Gy_per_sec > 0:
            time_to_target_sec = target_dose_Gy / dose_rate_Gy_per_sec
            time_to_target_min = time_to_target_sec / 60.0
            time_to_target_hour = time_to_target_min / 60.0
            time_to_target_day = time_to_target_hour / 24.0
            time_to_target_year = time_to_target_day / 365.25
        else:
            time_to_target_sec = float('inf')
            time_to_target_min = float('inf')
            time_to_target_hour = float('inf')
            time_to_target_day = float('inf')
            time_to_target_year = float('inf')
        
        # 存储结果
        result = {
            "name": source['name'],
            "filename": source['filename'],
            "num_events": num_events,
            "organ_count": len(organ_data),
            "total_mass_kg": total_mass_kg,
            "total_energy_J": total_energy_J,
            "weighted_dose_Gy": weighted_dose_Gy,
            "dose_pGy": weighted_dose_Gy * 1e12,
            "dose_rate_Gy_per_sec": dose_rate_Gy_per_sec,
            "time_to_25mSv_sec": time_to_target_sec,
            "time_to_25mSv_min": time_to_target_min,
            "time_to_25mSv_hour": time_to_target_hour,
            "time_to_25mSv_day": time_to_target_day,
            "time_to_25mSv_year": time_to_target_year,
            "organ_data": organ_data
        }
        
        results.append(result)
        
        # 4. 输出单个放射源结果
        print(f"\n{source['name']} 剂量计算结果:")
        print(f"  事件数 (N): {num_events:,}")
        print(f"  总质量: {total_mass_kg:.6f} kg ({total_mass_kg*1000:.2f} g)")
        print(f"  总吸收能量: {total_energy_J:.6e} J")
        print(f"  加权平均剂量: {weighted_dose_Gy:.6e} Gy/source")
        print(f"  加权平均剂量: {weighted_dose_Gy*1e12:.6f} pGy/source")
        print(f"  1居里活度下的剂量率: {dose_rate_Gy_per_sec:.3e} Gy/s")
        if time_to_target_sec != float('inf'):
            print(f"  达到25毫希沃特所需时间:")
            print(f"    {time_to_target_sec:.3e} 秒")
            print(f"    {time_to_target_min:.3e} 分钟")
            print(f"    {time_to_target_hour:.3e} 小时")
            print(f"    {time_to_target_day:.3e} 天")
            print(f"    {time_to_target_year:.3e} 年")
        else:
            print(f"  达到25毫希沃特所需时间: 无限长（剂量率为0）")
    
    # 5. 比较所有放射源的结果
    if results:
        print("\n" + "=" * 80)
        print("放射源剂量比较")
        print("=" * 80)
        
        print(f"\n{'放射源':<10} {'事件数':<15} {'器官数':<10} {'总质量(kg)':<15} {'剂量(Gy/source)':<20} {'剂量(pGy/source)':<20} {'达到25mSv时间(年)':<25}")
        print("-" * 120)
        
        for result in results:
            if result['time_to_25mSv_year'] != float('inf'):
                time_str = f"{result['time_to_25mSv_year']:.3e}"
            else:
                time_str = "无限长"
            print(f"{result['name']:<10} {result['num_events']:<15,} {result['organ_count']:<10} "
                  f"{result['total_mass_kg']:<15.6f} {result['weighted_dose_Gy']:<20.3e} {result['dose_pGy']:<20.3f} {time_str:<25}")
        
        # 6. 图表生成已禁用（根据用户要求）
        print("\n" + "=" * 80)
        print("图表生成状态")
        print("=" * 80)
        print("注: 图表生成功能已禁用")
        print("如需生成图表，请取消代码注释")
        
        # 7. 保存综合结果到文件
        print("\n" + "=" * 80)
        print("保存综合结果")
        print("=" * 80)
        
        output_file = os.path.join(current_dir, "multi_source_dose_results.txt")
        try:
            with open(output_file, 'w', encoding='utf-8-sig') as f:
                f.write("=" * 80 + "\n")
                f.write("多放射源剂量计算结果\n")
                f.write("=" * 80 + "\n\n")
                
                f.write("数据文件目录: " + data_dir + "\n")
                f.write("计算时间: " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n\n")
                
                f.write("各放射源详细结果:\n")
                f.write("-" * 100 + "\n")
                
                for result in results:
                    f.write(f"\n放射源: {result['name']} ({result['filename']})\n")
                    f.write(f"事件数 (N): {result['num_events']:,}\n")
                    f.write(f"器官数量: {result['organ_count']}\n")
                    f.write(f"总质量: {result['total_mass_kg']:.6f} kg\n")
                    f.write(f"总吸收能量: {result['total_energy_J']:.6e} J\n")
                    f.write(f"加权平均剂量: {result['weighted_dose_Gy']:.6e} Gy/source\n")
                    f.write(f"加权平均剂量: {result['dose_pGy']:.6f} pGy/source\n")
                    f.write(f"1居里活度下的剂量率: {result['dose_rate_Gy_per_sec']:.3e} Gy/s\n")
                    
                    if result['time_to_25mSv_year'] != float('inf'):
                        f.write(f"达到25毫希沃特所需时间:\n")
                        f.write(f"  {result['time_to_25mSv_sec']:.3e} 秒\n")
                        f.write(f"  {result['time_to_25mSv_min']:.3e} 分钟\n")
                        f.write(f"  {result['time_to_25mSv_hour']:.3e} 小时\n")
                        f.write(f"  {result['time_to_25mSv_day']:.3e} 天\n")
                        f.write(f"  {result['time_to_25mSv_year']:.3e} 年\n")
                    else:
                        f.write(f"达到25毫希沃特所需时间: 无限长（剂量率为0）\n")
                    f.write("-" * 80 + "\n")
                
                f.write("\n\n注: 1 Gy = 1 J/kg, 1 pGy = 10^{-12} Gy\n")
                f.write("剂量计算基于ICRP145人体模型和空气介质\n")
                f.write("计算假设: 放射源活度 = 1居里 = 3.7×10^10 Bq，目标剂量 = 25 mSv = 0.025 Sv\n")
                f.write("对于光子辐射，假设辐射权重因子 w_R = 1，因此 1 Gy ≈ 1 Sv\n")
            
            print(f"综合结果已保存到: {output_file}")
        
        except Exception as e:
            print(f"保存综合结果文件失败: {e}")
    
    print("\n" + "=" * 80)
    print("多放射源剂量分析完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
