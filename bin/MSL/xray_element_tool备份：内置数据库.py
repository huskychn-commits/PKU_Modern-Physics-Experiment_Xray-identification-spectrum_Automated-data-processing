#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X射线标识谱元素辅助工具
功能：
1. 输入元素名字（不区分大小写）输出核电荷数（原子序数）
2. 查询相对原子量和其他物理化学性质
3. 未来可扩展其他功能：电子排布、X射线特征峰等
"""

import sys
import re

class ElementDatabase:
    """元素数据库类，存储元素信息"""
    
    def __init__(self):
        """初始化元素数据库"""
        self.elements = self._create_element_database()
    
    def _create_element_database(self):
        """创建元素数据库，包含常见元素的名称、符号、原子序数和属性"""
        # 只包含部分常见元素以保持文件简洁，但包含属性字典
        elements = {
            # 元素符号: (原子序数, 中文名称, 英文名称列表, 属性字典)
            'H': (1, '氢', ['hydrogen'], {'atomic_mass': 1.008}),
            'He': (2, '氦', ['helium'], {'atomic_mass': 4.0026}),
            'Li': (3, '锂', ['lithium'], {'atomic_mass': 6.94}),
            'Be': (4, '铍', ['beryllium'], {'atomic_mass': 9.0122}),
            'B': (5, '硼', ['boron'], {'atomic_mass': 10.81}),
            'C': (6, '碳', ['carbon'], {'atomic_mass': 12.011}),
            'N': (7, '氮', ['nitrogen'], {'atomic_mass': 14.007}),
            'O': (8, '氧', ['oxygen'], {'atomic_mass': 15.999}),
            'F': (9, '氟', ['fluorine'], {'atomic_mass': 18.998}),
            'Ne': (10, '氖', ['neon'], {'atomic_mass': 20.180}),
            'Na': (11, '钠', ['sodium', 'natrium'], {'atomic_mass': 22.990}),
            'Mg': (12, '镁', ['magnesium'], {'atomic_mass': 24.305}),
            'Al': (13, '铝', ['aluminum', 'aluminium'], {'atomic_mass': 26.982}),
            'Si': (14, '硅', ['silicon'], {'atomic_mass': 28.085}),
            'P': (15, '磷', ['phosphorus'], {'atomic_mass': 30.974}),
            'S': (16, '硫', ['sulfur', 'sulphur'], {'atomic_mass': 32.06}),
            'Cl': (17, '氯', ['chlorine'], {'atomic_mass': 35.45}),
            'Ar': (18, '氩', ['argon'], {'atomic_mass': 39.948}),
            'K': (19, '钾', ['potassium', 'kalium'], {'atomic_mass': 39.098}),
            'Ca': (20, '钙', ['calcium'], {'atomic_mass': 40.078}),
            'Sc': (21, '钪', ['scandium'], {'atomic_mass': 44.956}),
            'Ti': (22, '钛', ['titanium'], {'atomic_mass': 47.867}),
            'V': (23, '钒', ['vanadium'], {'atomic_mass': 50.942}),
            'Cr': (24, '铬', ['chromium'], {'atomic_mass': 51.996}),
            'Mn': (25, '锰', ['manganese'], {'atomic_mass': 54.938}),
            'Fe': (26, '铁', ['iron', 'ferrum'], {'atomic_mass': 55.845, 'k_alpha_exp': 6.404}),
            'Co': (27, '钴', ['cobalt'], {'atomic_mass': 58.933, 'k_alpha_exp': 6.930}),
            'Ni': (28, '镍', ['nickel'], {'atomic_mass': 58.693, 'k_alpha_exp': 7.478}),
            'Cu': (29, '铜', ['copper', 'cuprum'], {'atomic_mass': 63.546, 'k_alpha_exp': 8.047}),
            'Zn': (30, '锌', ['zinc'], {'atomic_mass': 65.38, 'k_alpha_exp': 8.638}),
            'Ga': (31, '镓', ['gallium'], {'atomic_mass': 69.723, 'k_alpha_exp': 9.251}),
            'Ge': (32, '锗', ['germanium'], {'atomic_mass': 72.630, 'k_alpha_exp': 9.886}),
            'As': (33, '砷', ['arsenic'], {'atomic_mass': 74.922, 'k_alpha_exp': 10.543}),
            'Se': (34, '硒', ['selenium'], {'atomic_mass': 78.971, 'k_alpha_exp': 11.222}),
            'Br': (35, '溴', ['bromine'], {'atomic_mass': 79.904, 'k_alpha_exp': 11.924}),
            'Kr': (36, '氪', ['krypton'], {'atomic_mass': 83.798, 'k_alpha_exp': 12.649}),
            'Rb': (37, '铷', ['rubidium'], {'atomic_mass': 85.468, 'k_alpha_exp': 13.395}),
            'Sr': (38, '锶', ['strontium'], {'atomic_mass': 87.62, 'k_alpha_exp': 14.165}),
            'Y': (39, '钇', ['yttrium'], {'atomic_mass': 88.906, 'k_alpha_exp': 14.958}),
            'Zr': (40, '锆', ['zirconium'], {'atomic_mass': 91.224, 'k_alpha_exp': 15.775}),
            'Nb': (41, '铌', ['niobium'], {'atomic_mass': 92.906, 'k_alpha_exp': 16.615}),
            'Mo': (42, '钼', ['molybdenum'], {'atomic_mass': 95.95, 'k_alpha_exp': 17.479}),
            'Tc': (43, '锝', ['technetium'], {'atomic_mass': 98.0}),
            'Ru': (44, '钌', ['ruthenium'], {'atomic_mass': 101.07, 'k_alpha_exp': 19.279}),
            'Rh': (45, '铑', ['rhodium'], {'atomic_mass': 102.91, 'k_alpha_exp': 20.216}),
            'Pd': (46, '钯', ['palladium'], {'atomic_mass': 106.42, 'k_alpha_exp': 21.177}),
            'Ag': (47, '银', ['silver', 'argentum'], {'atomic_mass': 107.87, 'k_alpha_exp': 22.163}),
            'Cd': (48, '镉', ['cadmium'], {'atomic_mass': 112.41, 'k_alpha_exp': 23.173}),
            'In': (49, '铟', ['indium'], {'atomic_mass': 114.82, 'k_alpha_exp': 24.209}),
            'Sn': (50, '锡', ['tin', 'stannum'], {'atomic_mass': 118.71, 'k_alpha_exp': 25.271}),
            'Sb': (51, '锑', ['antimony', 'stibium'], {'atomic_mass': 121.76, 'k_alpha_exp': 26.359}),
            'Te': (52, '碲', ['tellurium'], {'atomic_mass': 127.60, 'k_alpha_exp': 27.472}),
            'I': (53, '碘', ['iodine'], {'atomic_mass': 126.90, 'k_alpha_exp': 28.612}),
            'Xe': (54, '氙', ['xenon'], {'atomic_mass': 131.29, 'k_alpha_exp': 29.779}),
            'Cs': (55, '铯', ['cesium', 'caesium'], {'atomic_mass': 132.91, 'k_alpha_exp': 30.973}),
            'Ba': (56, '钡', ['barium'], {'atomic_mass': 137.33, 'k_alpha_exp': 32.194}),
            'La': (57, '镧', ['lanthanum'], {'atomic_mass': 138.91, 'k_alpha_exp': 33.442}),
            'Ce': (58, '铈', ['cerium'], {'atomic_mass': 140.12, 'k_alpha_exp': 34.719}),
            'Pr': (59, '镨', ['praseodymium'], {'atomic_mass': 140.91, 'k_alpha_exp': 36.023}),
            'Nd': (60, '钕', ['neodymium'], {'atomic_mass': 144.24, 'k_alpha_exp': 37.355}),
            'Pm': (61, '钷', ['promethium'], {'atomic_mass': 145.0}),
            'Sm': (62, '钐', ['samarium'], {'atomic_mass': 150.36, 'k_alpha_exp': 40.118}),
            'Eu': (63, '铕', ['europium'], {'atomic_mass': 151.96, 'k_alpha_exp': 41.542}),
            'Gd': (64, '钆', ['gadolinium'], {'atomic_mass': 157.25, 'k_alpha_exp': 42.996}),
            'Tb': (65, '铽', ['terbium'], {'atomic_mass': 158.93, 'k_alpha_exp': 44.481}),
            'Dy': (66, '镝', ['dysprosium'], {'atomic_mass': 162.50, 'k_alpha_exp': 45.997}),
            'Ho': (67, '钬', ['holmium'], {'atomic_mass': 164.93, 'k_alpha_exp': 47.544}),
            'Er': (68, '铒', ['erbium'], {'atomic_mass': 167.26, 'k_alpha_exp': 49.123}),
            'Tm': (69, '铥', ['thulium'], {'atomic_mass': 168.93, 'k_alpha_exp': 50.734}),
            'Yb': (70, '镱', ['ytterbium'], {'atomic_mass': 173.05, 'k_alpha_exp': 52.377}),
            'Lu': (71, '镥', ['lutetium'], {'atomic_mass': 174.97, 'k_alpha_exp': 54.053}),
            # 为简洁起见，只包含到71号元素，但结构已展示
        }
        return elements
    
    def get_atomic_number(self, element_input):
        """
        根据元素输入获取原子序数（核电荷数）
        
        参数:
        element_input: 元素输入（可以是符号、中文名、英文名，不区分大小写）
        
        返回:
        atomic_number: 原子序数，如果找不到返回None
        """
        # 转换为小写以便不区分大小写比较
        input_lower = element_input.strip().lower()
        
        # 首先检查是否是元素符号（直接匹配）
        for symbol, (atomic_num, chinese_name, english_names, properties) in self.elements.items():
            # 检查元素符号（不区分大小写）
            if symbol.lower() == input_lower:
                return atomic_num
            
            # 检查中文名称
            if chinese_name.lower() == input_lower:
                return atomic_num
            
            # 检查英文名称
            for eng_name in english_names:
                if eng_name.lower() == input_lower:
                    return atomic_num
        
        # 如果没有找到，尝试部分匹配
        for symbol, (atomic_num, chinese_name, english_names, properties) in self.elements.items():
            # 检查是否以元素符号开头
            if input_lower.startswith(symbol.lower()):
                return atomic_num
            
            # 检查是否以中文名开头
            if chinese_name.lower().startswith(input_lower):
                return atomic_num
            
            # 检查是否以英文名开头
            for eng_name in english_names:
                if eng_name.lower().startswith(input_lower):
                    return atomic_num
        
        return None
    
    def get_atomic_mass(self, element_input):
        """
        根据元素输入获取相对原子量
        
        参数:
        element_input: 元素输入（可以是符号、中文名、英文名，不区分大小写）
        
        返回:
        atomic_mass: 相对原子量，如果找不到返回None
        """
        # 转换为小写以便不区分大小写比较
        input_lower = element_input.strip().lower()
        
        for symbol, (atomic_num, chinese_name, english_names, properties) in self.elements.items():
            # 检查元素符号（不区分大小写）
            if symbol.lower() == input_lower:
                return properties.get('atomic_mass')
            
            # 检查中文名称
            if chinese_name.lower() == input_lower:
                return properties.get('atomic_mass')
            
            # 检查英文名称
            for eng_name in english_names:
                if eng_name.lower() == input_lower:
                    return properties.get('atomic_mass')
        
        # 如果没有找到，尝试部分匹配
        for symbol, (atomic_num, chinese_name, english_names, properties) in self.elements.items():
            # 检查是否以元素符号开头
            if input_lower.startswith(symbol.lower()):
                return properties.get('atomic_mass')
            
            # 检查是否以中文名开头
            if chinese_name.lower().startswith(input_lower):
                return properties.get('atomic_mass')
            
            # 检查是否以英文名开头
            for eng_name in english_names:
                if eng_name.lower().startswith(input_lower):
                    return properties.get('atomic_mass')
        
        return None
    
    def get_element_info(self, element_input):
        """
        根据元素输入获取完整元素信息
        
        参数:
        element_input: 元素输入（可以是符号、中文名、英文名，不区分大小写）
        
        返回:
        dict: 包含元素信息的字典，如果找不到返回None
        """
        # 转换为小写以便不区分大小写比较
        input_lower = element_input.strip().lower()
        
        for symbol, (atomic_num, chinese_name, english_names, properties) in self.elements.items():
            # 检查元素符号（不区分大小写）
            if symbol.lower() == input_lower:
                return {
                    'symbol': symbol,
                    'atomic_number': atomic_num,
                    'chinese_name': chinese_name,
                    'english_names': english_names,
                    'properties': properties
                }
            
            # 检查中文名称
            if chinese_name.lower() == input_lower:
                return {
                    'symbol': symbol,
                    'atomic_number': atomic_num,
                    'chinese_name': chinese_name,
                    'english_names': english_names,
                    'properties': properties
                }
            
            # 检查英文名称
            for eng_name in english_names:
                if eng_name.lower() == input_lower:
                    return {
                        'symbol': symbol,
                        'atomic_number': atomic_num,
                        'chinese_name': chinese_name,
                        'english_names': english_names,
                        'properties': properties
                    }
        
        # 如果没有找到，尝试部分匹配
        for symbol, (atomic_num, chinese_name, english_names, properties) in self.elements.items():
            # 检查是否以元素符号开头
            if input_lower.startswith(symbol.lower()):
                return {
                    'symbol': symbol,
                    'atomic_number': atomic_num,
                    'chinese_name': chinese_name,
                    'english_names': english_names,
                    'properties': properties
                }
            
            # 检查是否以中文名开头
            if chinese_name.lower().startswith(input_lower):
                return {
                    'symbol': symbol,
                    'atomic_number': atomic_num,
                    'chinese_name': chinese_name,
                    'english_names': english_names,
                    'properties': properties
                }
            
            # 检查是否以英文名开头
            for eng_name in english_names:
                if eng_name.lower().startswith(input_lower):
                    return {
                        'symbol': symbol,
                        'atomic_number': atomic_num,
                        'chinese_name': chinese_name,
                        'english_names': english_names,
                        'properties': properties
                    }
        
        return None
    
    def list_all_elements(self):
        """列出所有元素"""
        elements_list = []
        for symbol, (atomic_num, chinese_name, english_names, properties) in sorted(
            self.elements.items(), key=lambda x: x[1][0]):
            elements_list.append({
                'symbol': symbol,
                'atomic_number': atomic_num,
                'chinese_name': chinese_name,
                'english_names': english_names,
                'atomic_mass': properties.get('atomic_mass')
            })
        return elements_list
    
    def get_k_alpha_energy(self, element_input):
        """
        计算类氢原子的Kα线能量（单位为千电子伏特）
        
        参数:
        element_input: 元素输入（可以是符号、中文名、英文名，不区分大小写）
        
        返回:
        k_alpha_energy: Kα线能量（keV），如果找不到元素返回None
        
        物理原理:
        对于类氢原子（只有一个电子），Kα线对应电子从n=2能级跃迁到n=1能级（K壳层）时发射的X射线。
        能级公式: E_n = -Z^2 * R_H / n^2，其中Z是原子序数，R_H是里德伯常数（约13.6 eV）
        Kα线能量: ΔE = E_2 - E_1 = Z^2 * R_H * (1/1^2 - 1/2^2) = Z^2 * R_H * (3/4)
        """
        # 首先获取原子序数
        atomic_number = self.get_atomic_number(element_input)
        if atomic_number is None:
            return None
        
        # 物理常数定义（在方法开头集中列出）
        # 电子电量 (C)
        e = 1.602176634e-19
        # 电子质量 (kg)
        m_e = 9.10938356e-31
        # 普朗克常数 (J·s)
        h = 6.62607015e-34
        # 真空介电常数 (F/m)
        epsilon_0 = 8.8541878128e-12
        # 真空光速 (m/s)
        c = 299792458
        
        # 计算里德伯常数 (eV)
        # 公式: R_H = (m_e * e^4) / (8 * epsilon_0^2 * h^2 * c) * (1/(4*pi*epsilon_0)) 
        # 简化: 使用已知值 R_H = 13.605693122994 eV
        R_H = 13.605693122994  # eV
        
        # 计算Kα线能量 (eV)
        # ΔE = (Z-1)^2 * R_H * (3/4)  (考虑电子屏蔽效应)
        k_alpha_energy_eV = (atomic_number-1)**2 * R_H * (3/4)
        
        # 转换为keV (1 keV = 1000 eV)
        k_alpha_energy_keV = k_alpha_energy_eV / 1000.0
        
        return k_alpha_energy_keV
    
    def get_k_alpha_energy_relativistic(self, element_input):
        """
        计算相对论修正的Kα线能量（单位为千电子伏特）
        
        参数:
        element_input: 元素输入（可以是符号、中文名、英文名，不区分大小写）
        
        返回:
        k_alpha_energy: Kα线能量（keV），如果找不到元素返回None
        
        物理原理:
        相对论修正的类氢原子能级公式:
        E = m_e c^2 (1 + (α (Z-σ)/(n - (j+1/2) + √((j+1/2)² - (αZ)²)))²)^{-1/2}
        其中:
        - m_e: 电子质量
        - c: 光速
        - α: 精细结构常数 ≈ 1/137.036
        - Z: 原子序数
        - σ: 屏蔽系数 (对于Kα线，σ=1)
        - n: 主量子数
        - j: 总角动量
        
        Kα谱线是(n=2,j=3/2)->(n=1,j=1/2)和(n=2,j=1/2)->(n=1,j=1/2)的混合，比例2:1
        """
        # 首先获取原子序数
        atomic_number = self.get_atomic_number(element_input)
        if atomic_number is None:
            return None
        
        # 物理常数
        # 电子质量 (kg)
        m_e = 9.10938356e-31
        # 光速 (m/s)
        c = 299792458
        # 精细结构常数
        alpha = 1.0 / 137.035999084
        
        # 屏蔽系数 (对于Kα线)
        sigma = 1.0
        
        # 计算电子静能 (J)
        m_e_c2 = m_e * c**2  # 焦耳
        
        # 转换为电子伏特 (1 eV = 1.602176634e-19 J)
        m_e_c2_eV = m_e_c2 / 1.602176634e-19
        
        # 定义相对论能级函数
        def relativistic_energy(n, j, Z):
            """计算相对论修正的能级 (eV)"""
            # 公式: E = m_e c^2 (1 + (α (Z-σ)/(n - (j+1/2) + √((j+1/2)² - (αZ)²)))²)^{-1/2}
            denominator = n - (j + 0.5) + ((j + 0.5)**2 - (alpha * Z)**2)**0.5
            if denominator <= 0:
                # 避免除零或负数，使用近似值
                denominator = 0.1
            x = alpha * (Z - sigma) / denominator
            energy = m_e_c2_eV / (1 + x**2)**0.5
            return energy
        
        # 计算四个能级
        # 初始态: n=2, j=3/2 和 j=1/2
        # 终态: n=1, j=1/2
        Z = atomic_number
        
        # 能级 (eV)
        E_2_j32 = relativistic_energy(2, 1.5, Z)  # n=2, j=3/2
        E_2_j12 = relativistic_energy(2, 0.5, Z)  # n=2, j=1/2
        E_1_j12 = relativistic_energy(1, 0.5, Z)  # n=1, j=1/2
        
        # 计算两个跃迁的能量差
        delta_E1 = E_2_j32 - E_1_j12  # (n=2,j=3/2) -> (n=1,j=1/2)
        delta_E2 = E_2_j12 - E_1_j12  # (n=2,j=1/2) -> (n=1,j=1/2)
        
        # 按2:1比例加权平均
        # Kα1: (n=2,j=3/2)->(n=1,j=1/2) 强度2
        # Kα2: (n=2,j=1/2)->(n=1,j=1/2) 强度1
        weighted_energy_eV = (2 * delta_E1 + delta_E2) / 3.0
        
        # 转换为keV
        weighted_energy_keV = weighted_energy_eV / 1000.0
        
        return weighted_energy_keV
    
    def get_element_by_atomic_number(self, atomic_number):
        """
        通过原子序数查询元素信息
        
        参数:
        atomic_number: 原子序数（整数）
        
        返回:
        dict: 包含元素信息的字典，如果找不到返回None
        """
        # 确保原子序数是整数
        try:
            atomic_number = int(atomic_number)
        except (ValueError, TypeError):
            return None
        
        # 遍历所有元素，查找匹配的原子序数
        for symbol, (atomic_num, chinese_name, english_names, properties) in self.elements.items():
            if atomic_num == atomic_number:
                return {
                    'symbol': symbol,
                    'atomic_number': atomic_num,
                    'chinese_name': chinese_name,
                    'english_names': english_names,
                    'properties': properties
                }
        
        return None


class XRayElementTool:
    """X射线元素工具主类"""
    
    def __init__(self):
        """初始化工具"""
        self.db = ElementDatabase()
        self.running = True
    
    def print_banner(self):
        """打印程序横幅"""
        print("=" * 60)
        print("X射线标识谱元素辅助工具")
        print("=" * 60)
        print("功能: 输入元素名字（不区分大小写）输出核电荷数（原子序数）")
        print("      查询相对原子量和其他物理化学性质")
        print("支持: 元素符号 (如 Fe, Ag), 中文名 (如 铁, 银), 英文名 (如 iron, silver)")
        print("命令: 'help' 显示帮助, 'list' 列出所有元素, 'exit' 退出程序")
        print("=" * 60)
    
    def print_help(self):
        """打印帮助信息"""
        print("\n可用命令:")
        print("  help              - 显示此帮助信息")
        print("  list              - 列出所有元素")
        print("  mass <元素名>     - 查询元素的相对原子量")
        print("  kalpha <元素名>   - 查询类氢原子的Kα线能量（keV）")
        print("  kalpha_rel <元素名> - 查询相对论修正的Kα线能量（keV）")
        print("  exit 或 quit     - 退出程序")
        print("  <元素名>          - 查询元素的核电荷数")
        print("  <原子序数>        - 通过原子序数查询元素信息")
        print("\n查询示例:")
        print("  > Fe              - 查询铁元素的核电荷数")
        print("  > iron            - 同上（英文名）")
        print("  > 铁              - 同上（中文名）")
        print("  > 26              - 通过原子序数26查询铁元素信息")
        print("  > mass Fe         - 查询铁元素的相对原子量")
        print("  > mass 铁         - 同上（中文名）")
        print("  > kalpha Fe       - 查询铁元素的类氢原子Kα线能量")
        print("  > kalpha_rel Fe   - 查询铁元素的相对论修正Kα线能量")
        print("  > kalpha 银       - 查询银元素的类氢原子Kα线能量")
        print("  > kalpha_rel 银   - 查询银元素的相对论修正Kα线能量")
        print("  > Ag              - 查询银元素的核电荷数")
        print("  > mass Ag         - 查询银元素的相对原子量")
        print("  > 47              - 通过原子序数47查询银元素信息")
    
    def list_elements(self):
        """列出所有元素"""
        elements = self.db.list_all_elements()
        print(f"\n元素列表 (共{len(elements)}个元素):")
        print("-" * 80)
        print(f"{'原子序数':<8} {'符号':<6} {'中文名':<8} {'英文名':<20} {'相对原子量':<12}")
        print("-" * 80)
        
        for elem in elements:
            # 获取主要英文名
            main_english = elem['english_names'][0] if elem['english_names'] else ""
            atomic_mass = elem['atomic_mass']
            atomic_mass_str = f"{atomic_mass:.4f}" if atomic_mass else "N/A"
            print(f"{elem['atomic_number']:<10} {elem['symbol']:<8} {elem['chinese_name']:<10} {main_english:<20} {atomic_mass_str:<12}")
        
        print("-" * 80)
        print(f"提示: 可以使用以上任意名称进行查询")
    
    def get_element(self, element_input):
        """获取元素信息"""
        # 获取原子序数
        atomic_number = self.db.get_atomic_number(element_input)
        
        if atomic_number is not None:
            # 获取完整信息用于显示
            element_info = self.db.get_element_info(element_input)
            if element_info:
                print(f"\n查询结果:")
                print(f"  元素符号: {element_info['symbol']}")
                print(f"  中文名称: {element_info['chinese_name']}")
                print(f"  英文名称: {', '.join(element_info['english_names'])}")
                print(f"  核电荷数（原子序数）: {atomic_number}")
                
                # 显示相对原子量
                atomic_mass = element_info['properties'].get('atomic_mass')
                if atomic_mass:
                    print(f"  相对原子量: {atomic_mass}")
                
                # 显示相关提示
                if len(element_info['english_names']) > 1:
                    print(f"  其他英文名: {', '.join(element_info['english_names'][1:])}")
            else:
                print(f"\n查询结果: {element_input} → 核电荷数: {atomic_number}")
        else:
            print(f"\n错误: 未找到元素 '{element_input}'")
            print("提示: 请检查拼写，或使用 'list' 命令查看所有可用元素")
    
    def get_atomic_mass(self, element_input):
        """获取元素的相对原子量"""
        atomic_mass = self.db.get_atomic_mass(element_input)
        
        if atomic_mass is not None:
            element_info = self.db.get_element_info(element_input)
            if element_info:
                print(f"\n查询结果:")
                print(f"  元素符号: {element_info['symbol']}")
                print(f"  中文名称: {element_info['chinese_name']}")
                print(f"  相对原子量: {atomic_mass}")
            else:
                print(f"\n查询结果: {element_input} → 相对原子量: {atomic_mass}")
        else:
            print(f"\n错误: 未找到元素 '{element_input}' 或该元素没有相对原子量数据")
            print("提示: 请检查拼写，或使用 'list' 命令查看所有可用元素")
    
    def get_k_alpha_energy(self, element_input):
        """获取类氢原子的Kα线能量"""
        k_alpha_energy = self.db.get_k_alpha_energy(element_input)
        
        if k_alpha_energy is not None:
            element_info = self.db.get_element_info(element_input)
            if element_info:
                print(f"\n查询结果:")
                print(f"  元素符号: {element_info['symbol']}")
                print(f"  中文名称: {element_info['chinese_name']}")
                print(f"  原子序数: {element_info['atomic_number']}")
                print(f"  类氢原子Kα线能量: {k_alpha_energy:.4f} keV")
                
                # 显示物理原理说明
                print(f"\n  物理原理:")
                print(f"    对于类氢原子（只有一个电子），Kα线对应电子从n=2能级跃迁到n=1能级时发射的X射线。")
                print(f"    计算公式: ΔE = (Z-1)² × R_H × (3/4)  (考虑电子屏蔽效应)")
                print(f"    其中 Z = {element_info['atomic_number']} (原子序数), R_H = 13.6057 eV (里德伯常数)")
                print(f"    计算值: {k_alpha_energy:.4f} keV = {k_alpha_energy*1000:.2f} eV")
                print(f"    注: 使用(Z-1)而非Z是为了考虑内层电子对核电荷的屏蔽效应")
            else:
                print(f"\n查询结果: {element_input} → 类氢原子Kα线能量: {k_alpha_energy:.4f} keV")
        else:
            print(f"\n错误: 未找到元素 '{element_input}'")
            print("提示: 请检查拼写，或使用 'list' 命令查看所有可用元素")
    
    def get_k_alpha_energy_relativistic(self, element_input):
        """获取相对论修正的Kα线能量"""
        k_alpha_energy = self.db.get_k_alpha_energy_relativistic(element_input)
        
        if k_alpha_energy is not None:
            element_info = self.db.get_element_info(element_input)
            if element_info:
                print(f"\n查询结果:")
                print(f"  元素符号: {element_info['symbol']}")
                print(f"  中文名称: {element_info['chinese_name']}")
                print(f"  原子序数: {element_info['atomic_number']}")
                print(f"  相对论修正Kα线能量: {k_alpha_energy:.4f} keV")
                
                # 显示物理原理说明
                print(f"\n  物理原理:")
                print(f"    相对论修正的类氢原子能级公式:")
                print(f"    E = m_e c² [1 + (α(Z-σ)/(n - (j+1/2) + √((j+1/2)² - (αZ)²)))²]^{-1/2}")
                print(f"    其中:")
                print(f"      m_e = 9.10938356e-31 kg (电子质量)")
                print(f"      c = 299792458 m/s (光速)")
                print(f"      α = 1/137.036 (精细结构常数)")
                print(f"      Z = {element_info['atomic_number']} (原子序数)")
                print(f"      σ = 1 (屏蔽系数，对于Kα线)")
                print(f"      n = 主量子数 (1, 2)")
                print(f"      j = 总角动量 (1/2, 3/2)")
                print(f"    Kα谱线是(n=2,j=3/2)->(n=1,j=1/2)和(n=2,j=1/2)->(n=1,j=1/2)的混合")
                print(f"    按2:1比例加权平均得到最终能量")
                print(f"    计算值: {k_alpha_energy:.4f} keV = {k_alpha_energy*1000:.2f} eV")
            else:
                print(f"\n查询结果: {element_input} → 相对论修正Kα线能量: {k_alpha_energy:.4f} keV")
        else:
            print(f"\n错误: 未找到元素 '{element_input}'")
            print("提示: 请检查拼写，或使用 'list' 命令查看所有可用元素")
    
    def get_element_by_atomic_number(self, atomic_number_input):
        """通过原子序数查询元素信息"""
        element_info = self.db.get_element_by_atomic_number(atomic_number_input)
        
        if element_info is not None:
            print(f"\n查询结果:")
            print(f"  原子序数: {element_info['atomic_number']}")
            print(f"  元素符号: {element_info['symbol']}")
            print(f"  中文名称: {element_info['chinese_name']}")
            print(f"  英文名称: {', '.join(element_info['english_names'])}")
            
            # 显示相对原子量
            atomic_mass = element_info['properties'].get('atomic_mass')
            if atomic_mass:
                print(f"  相对原子量: {atomic_mass}")
            
            # 显示实验Kα线能量（如果存在）
            k_alpha_exp = element_info['properties'].get('k_alpha_exp')
            if k_alpha_exp:
                print(f"  实验Kα线能量: {k_alpha_exp} keV")
            
            # 显示相关提示
            if len(element_info['english_names']) > 1:
                print(f"  其他英文名: {', '.join(element_info['english_names'][1:])}")
        else:
            print(f"\n错误: 未找到原子序数为 '{atomic_number_input}' 的元素")
            print("提示: 使用 'list' 命令查看所有可用元素及其原子序数")
    
    def run_interactive(self):
        """运行交互式模式"""
        self.print_banner()
        
        while self.running:
            try:
                # 获取用户输入
                user_input = input("\n> ").strip()
                
                # 处理空输入
                if not user_input:
                    continue
                
                # 处理命令
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("退出程序。")
                    self.running = False
                elif user_input.lower() == 'help':
                    self.print_help()
                elif user_input.lower() == 'list':
                    self.list_elements()
                elif user_input.lower().startswith('mass '):
                    # 查询相对原子量
                    element_name = user_input[5:].strip()
                    if element_name:
                        self.get_atomic_mass(element_name)
                    else:
                        print("错误: 请指定元素名，例如 'mass Fe' 或 'mass 铁'")
                elif user_input.lower().startswith('kalpha '):
                    # 查询Kα线能量
                    element_name = user_input[7:].strip()
                    if element_name:
                        self.get_k_alpha_energy(element_name)
                    else:
                        print("错误: 请指定元素名，例如 'kalpha Fe' 或 'kalpha 铁'")
                elif user_input.lower().startswith('kalpha_rel '):
                    # 查询相对论修正的Kα线能量
                    element_name = user_input[11:].strip()
                    if element_name:
                        self.get_k_alpha_energy_relativistic(element_name)
                    else:
                        print("错误: 请指定元素名，例如 'kalpha_rel Fe' 或 'kalpha_rel 铁'")
                else:
                    # 检查输入是否是数字（原子序数）
                    if user_input.isdigit():
                        # 通过原子序数查询元素信息
                        self.get_element_by_atomic_number(user_input)
                    else:
                        # 查询元素
                        self.get_element(user_input)
                    
            except KeyboardInterrupt:
                print("\n\n检测到中断信号，退出程序。")
                self.running = False
            except EOFError:
                print("\n\n检测到文件结束，退出程序。")
                self.running = False
            except Exception as e:
                print(f"\n错误: {e}")
                print("提示: 如果问题持续，请重启程序。")
    
    def run_command_line(self, element_input):
        """命令行模式：直接查询元素并输出两个数值：实验Kα线能量和相对论计算值"""
        # 首先检查输入是否是数字（原子序数）
        if element_input.isdigit():
            # 通过原子序数查询元素信息
            element_info = self.db.get_element_by_atomic_number(element_input)
            if element_info is not None:
                # 获取实验Kα线能量（如果没查到返回0）
                k_alpha_exp = element_info['properties'].get('k_alpha_exp', 0.0)
                # 获取相对论计算值
                k_alpha_rel = self.db.get_k_alpha_energy_relativistic(element_info['symbol'])
                if k_alpha_rel is None:
                    k_alpha_rel = 0.0
                # 输出两个数值，用空格分隔
                print(f"{k_alpha_exp:.6f} {k_alpha_rel:.6f}")
                return 0  # 成功退出码
            else:
                print(f"错误: 未找到原子序数为 '{element_input}' 的元素")
                print("提示: 使用 'xray_element_tool.py list' 查看所有可用元素")
                return 1  # 错误退出码
        else:
            # 通过元素名查询
            atomic_number = self.db.get_atomic_number(element_input)
            
            if atomic_number is not None:
                element_info = self.db.get_element_info(element_input)
                if element_info:
                    # 获取实验Kα线能量（如果没查到返回0）
                    k_alpha_exp = element_info['properties'].get('k_alpha_exp', 0.0)
                    # 获取相对论计算值
                    k_alpha_rel = self.db.get_k_alpha_energy_relativistic(element_input)
                    if k_alpha_rel is None:
                        k_alpha_rel = 0.0
                    # 输出两个数值，用空格分隔
                    print(f"{k_alpha_exp:.6f} {k_alpha_rel:.6f}")
                else:
                    # 如果找不到完整元素信息，只使用原子序数
                    # 获取实验Kα线能量（默认为0）
                    k_alpha_exp = 0.0
                    # 获取相对论计算值
                    k_alpha_rel = self.db.get_k_alpha_energy_relativistic(element_input)
                    if k_alpha_rel is None:
                        k_alpha_rel = 0.0
                    # 输出两个数值，用空格分隔
                    print(f"{k_alpha_exp:.6f} {k_alpha_rel:.6f}")
                return 0  # 成功退出码
            else:
                print(f"错误: 未找到元素 '{element_input}'")
                print("提示: 使用 'xray_element_tool.py list' 查看所有可用元素")
                return 1  # 错误退出码


def main():
    """主函数"""
    tool = XRayElementTool()
    
    # 检查命令行参数
    if len(sys.argv) == 1:
        # 没有参数，进入交互模式
        tool.run_interactive()
    elif len(sys.argv) == 2:
        # 一个参数，可能是元素名或命令
        arg = sys.argv[1]
        
        if arg.lower() == 'list':
            # 列出所有元素
            tool.list_elements()
        elif arg.lower() in ['help', '--help', '-h']:
            # 显示帮助
            tool.print_banner()
            tool.print_help()
            print("\n命令行用法:")
            print("  xray_element_tool.py                    - 进入交互模式")
            print("  xray_element_tool.py list              - 列出所有元素")
            print("  xray_element_tool.py <元素名或原子序数> - 查询元素的Kα线能量")
            print("    输出格式: 实验值 计算值 (两个数值，空格分隔)")
            print("    示例: xray_element_tool.py Fe  # 输出: 6.404000 6.404000")
            print("  xray_element_tool.py help              - 显示此帮助信息")
        else:
            # 查询元素
            return tool.run_command_line(arg)
    else:
        # 多个参数，只处理第一个
        print("警告: 只接受一个参数，将使用第一个参数进行查询")
        return tool.run_command_line(sys.argv[1])
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
