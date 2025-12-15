#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto.py - 自动生成实验报告图片

功能：
1. 读取bin/实验报告图片生成对应表.csv
2. 按顺序调用程序中的方法生成对应的图片
3. 将生成的图片保存到Figures文件夹中

使用方法：
python Auto.py
"""

import os
import sys
import csv
import importlib.util
import subprocess
import shutil
from pathlib import Path

# 设置路径
BASE_DIR = Path(__file__).parent
BIN_DIR = BASE_DIR / "bin"
FIGURES_DIR = BASE_DIR / "Figures"
CSV_FILE = BIN_DIR / "实验报告图片生成对应表.csv"

# 确保Figures目录存在
FIGURES_DIR.mkdir(exist_ok=True)

def load_csv():
    """加载CSV文件，返回图片信息列表"""
    if not CSV_FILE.exists():
        print(f"错误: 找不到CSV文件 {CSV_FILE}")
        return []
    
    images_info = []
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            images_info.append({
                '序号': row['图片序号'],
                '名称': row['图片名称'],
                '脚本': row['生成代码']
            })
    
    print(f"从CSV文件加载了 {len(images_info)} 个图片信息")
    return images_info

def import_and_call_generate_figures(script_path, output_dir, image_name=None):
    """
    导入脚本并调用generate_figures函数
    
    参数:
    script_path: 脚本路径
    output_dir: 输出目录
    image_name: 图片名称（可选），用于指定生成哪个图片
    
    返回:
    list: 生成的图片文件列表
    """
    script_path = Path(script_path)
    if not script_path.exists():
        print(f"错误: 脚本不存在 {script_path}")
        return []
    
    # 获取模块名（去掉.py扩展名）
    module_name = script_path.stem
    
    # 添加脚本所在目录到Python路径
    script_dir = str(script_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    
    print(f"导入模块: {module_name}")
    try:
        # 动态导入模块
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # 检查是否有generate_figures函数
        if hasattr(module, 'generate_figures'):
            print(f"调用 {module_name}.generate_figures()")
            # 调用generate_figures函数，传递image_name参数
            if image_name:
                # 尝试调用带image_name参数的版本
                try:
                    generated_images = module.generate_figures(output_dir=str(output_dir), image_name=image_name)
                except TypeError as e:
                    print(f"  警告: 函数不接受image_name参数，使用默认调用: {e}")
                    # 如果函数不接受image_name参数，使用默认调用
                    generated_images = module.generate_figures(output_dir=str(output_dir))
            else:
                generated_images = module.generate_figures(output_dir=str(output_dir))
                
            if generated_images:
                print(f"  成功生成 {len(generated_images)} 个图片")
                return [Path(img) for img in generated_images]
            else:
                print(f"  警告: generate_figures函数未返回图片")
                return []
        else:
            print(f"  警告: 模块 {module_name} 没有generate_figures函数")
            return []
            
    except Exception as e:
        print(f"导入或调用模块时出错: {e}")
        import traceback
        traceback.print_exc()
        return []

def run_script_directly(script_path, output_dir, image_name=None):
    """
    直接运行脚本，并捕获其生成的图片（回退方案）
    
    参数:
    script_path: 脚本路径
    output_dir: 输出目录
    image_name: 图片名称（可选），用于重命名生成的图片
    
    返回:
    list: 生成的图片文件列表
    """
    script_path = Path(script_path)
    if not script_path.exists():
        print(f"错误: 脚本不存在 {script_path}")
        return []
    
    # 获取脚本所在目录
    script_dir = script_path.parent
    
    # 运行脚本
    print(f"运行脚本: {script_path}")
    try:
        # 切换到脚本目录
        original_cwd = os.getcwd()
        os.chdir(script_dir)
        
        # 运行脚本
        result = subprocess.run([sys.executable, script_path.name], 
                               capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode != 0:
            print(f"脚本运行失败: {result.stderr}")
            return []
        
        print(f"脚本运行成功")
        
        # 查找脚本目录中新生成的图片文件
        generated_images = []
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.pdf']:
            for img_file in script_dir.glob(f'*{ext}'):
                # 检查文件是否是新生成的（根据修改时间）
                # 这里简单地将所有图片文件都视为生成的
                generated_images.append(img_file)
        
        # 将图片复制到输出目录
        copied_images = []
        for i, img_file in enumerate(generated_images):
            # 生成输出文件名
            if image_name:
                # 如果有多个图片，为每个图片添加序号
                if len(generated_images) > 1:
                    output_filename = f"{image_name}_{i+1}.png"
                else:
                    output_filename = f"{image_name}.png"
            else:
                output_filename = f"{script_path.stem}_{img_file.name}"
            
            output_path = output_dir / output_filename
            
            # 复制文件
            shutil.copy2(img_file, output_path)
            copied_images.append(output_path)
            print(f"  复制图片: {img_file.name} -> {output_path.name}")
        
        return copied_images
        
    except Exception as e:
        print(f"运行脚本时出错: {e}")
        return []
    finally:
        # 恢复原始工作目录
        os.chdir(original_cwd)

def main():
    """主函数"""
    print("=" * 60)
    print("Auto.py - 自动生成实验报告图片")
    print("=" * 60)
    
    # 加载CSV文件
    images_info = load_csv()
    if not images_info:
        return
    
    # 处理每个图片
    all_generated_images = []
    for i, info in enumerate(images_info):
        print(f"\n[{i+1}/{len(images_info)}] 处理: {info['序号']} - {info['名称']}")
        print(f"  脚本: {info['脚本']}")
        
        # 构建脚本路径
        script_path = BIN_DIR / info['脚本']
        
        # 构建图片名称：使用"序号+名称"的形式
        image_name = f"{info['序号']} - {info['名称']}"
        
        # 首先尝试导入并调用generate_figures函数
        generated_images = import_and_call_generate_figures(script_path, FIGURES_DIR, image_name)
        
        # 如果失败，回退到直接运行脚本
        if not generated_images:
            print(f"  尝试回退到直接运行脚本...")
            generated_images = run_script_directly(script_path, FIGURES_DIR, image_name)
        
        if generated_images:
            print(f"  成功生成 {len(generated_images)} 个图片")
            all_generated_images.extend(generated_images)
        else:
            print(f"  警告: 未生成图片")
    
    print("\n" + "=" * 60)
    print("处理完成！")
    print(f"共生成 {len(all_generated_images)} 个图片")
    print(f"所有图片已保存到: {FIGURES_DIR}")
    
    # 列出生成的图片
    if all_generated_images:
        print("\n生成的图片列表:")
        for img in all_generated_images:
            print(f"  - {img.name}")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
