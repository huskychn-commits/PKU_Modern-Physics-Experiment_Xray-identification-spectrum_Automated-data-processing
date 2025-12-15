#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试X射线元素工具
"""

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from xray_element_tool import XRayElementTool

def test_tool():
    """测试工具功能"""
    print("测试X射线元素工具...")
    print("=" * 60)
    
    # 创建工具实例
    tool = XRayElementTool()
    
    # 测试元素查询
    print("1. 测试元素查询:")
    print("-" * 40)
    
    test_cases = [
        "Fe",      # 元素符号
        "铁",      # 中文名
        "iron",    # 英文名
        "Ag",      # 银
        "银",      # 中文名
        "silver",  # 英文名
        "Cu",      # 铜
        "铜",      # 中文名
        "copper",  # 英文名
    ]
    
    for test_input in test_cases:
        print(f"\n查询: {test_input}")
        tool.get_element(test_input)
    
    # 测试相对原子量
    print("\n\n2. 测试相对原子量查询:")
    print("-" * 40)
    
    for test_input in ["Fe", "Ag", "Cu"]:
        print(f"\n查询: {test_input}")
        tool.get_atomic_mass(test_input)
    
    # 测试Kα线能量
    print("\n\n3. 测试Kα线能量查询:")
    print("-" * 40)
    
    for test_input in ["Fe", "Ag", "Cu"]:
        print(f"\n查询: {test_input}")
        tool.get_k_alpha_energy(test_input)
    
    # 测试相对论修正的Kα线能量
    print("\n\n4. 测试相对论修正的Kα线能量查询:")
    print("-" * 40)
    
    for test_input in ["Fe", "Ag", "Cu"]:
        print(f"\n查询: {test_input}")
        tool.get_k_alpha_energy_relativistic(test_input)
    
    # 测试列表功能
    print("\n\n5. 测试元素列表:")
    print("-" * 40)
    tool.list_elements()
    
    print("\n" + "=" * 60)
    print("测试完成！")

if __name__ == "__main__":
    test_tool()
