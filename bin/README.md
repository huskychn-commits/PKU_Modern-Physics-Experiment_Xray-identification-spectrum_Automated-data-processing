# X射线标识谱数据处理程序

本目录包含X射线标识谱实验的数据处理程序，分为四个主要模块：Absorb（衰减分析）、Dose（剂量计算）、MSL（Moseley定律分析）和Compton（康普顿散射分析）。

## 目录结构

```
数据/数据处理/
├── Absorb/                    # 衰减分析模块
│   ├── ReadData.py           # 数据读取程序
│   ├── decay_rate.py         # 衰减率分析主程序
│   ├── peakdrift.py          # 峰位漂移分析
│   ├── absorb_processed_data.json  # 处理后的数据
│   ├── decay_coefficients.csv      # 衰减系数数据
│   ├── decay_coefficient_plot.png  # 衰减系数图
│   ├── event_ratio_plot.png        # 事件数比值图
│   ├── log_log_relationship_plot.png  # log-log关系图
│   └── peak_drift_plot.png         # 峰位漂移图
├── Dose/                     # 剂量计算模块
│   ├── dose.py              # 剂量计算主程序
│   └── dose_results.txt     # 计算结果
├── MSL/                     # Moseley定律分析模块
│   ├── process_msl_data.py  # MSL数据处理
│   ├── 高斯峰拟合.py        # 高斯峰拟合程序
│   ├── xray_element_tool.py # X射线元素分析工具
│   ├── msl_linearfit.py     # Moseley定律线性拟合
│   ├── plot_kalpha_vs_z.py  # Kα线 vs 原子序数图
│   ├── plot_msl_vs_database.py  # 与数据库对比图
│   ├── element_database.json    # 元素数据库
│   ├── msl_peak_results.csv     # 峰位结果
│   └── 多个分析结果图
└── Compton.py               # 康普顿散射分析程序
    Compton_scattering_plots.png  # 康普顿散射分析图
```

## 各模块功能说明

### 1. Absorb模块（衰减分析）

#### 1.1 ReadData.py
**功能**：从`数据/Absorb`文件夹读取原始数据，处理并保存为JSON格式。

**使用方法**：
```bash
cd 数据/数据处理/Absorb
python ReadData.py
```

**输出文件**：
- `absorb_processed_data.json`：处理后的数据，包含各元素的衰减片层数、总事件数、高斯峰面积、高斯峰位置等信息。

#### 1.2 decay_rate.py
**功能**：分析X射线在不同衰减片层数下的衰减特性，验证衰减系数与能量的关系。

**主要功能**：
1. 计算事件数比值：事件数/无衰减事件数
2. 用exp[-kN]拟合衰减曲线
3. 绘制事件数比值随衰减片层数变化图
4. 绘制衰减系数k vs 谱线能量图
5. 绘制log-log关系图验证衰减系数 ∝ 能量^(-3)
6. 保存衰减系数数据到CSV文件

**使用方法**：
```bash
cd 数据/数据处理/Absorb
python decay_rate.py
```

**输出文件**：
- `decay_coefficients.csv`：衰减系数数据（元素,谱线能量,衰减系数）
- `event_ratio_plot.png`：事件数比值图
- `decay_coefficient_plot.png`：衰减系数vs谱线能量图
- `log_log_relationship_plot.png`：log-log关系图

#### 1.3 peakdrift.py
**功能**：分析峰位随衰减片层数的漂移。

**使用方法**：
```bash
cd 数据/数据处理/Absorb
python peakdrift.py
```

**输出文件**：
- `peak_drift_plot.png`：峰位漂移图

### 2. Dose模块（剂量计算）

#### 2.1 dose.py
**功能**：计算Cs-137源对人体模型的剂量分布。

**主要功能**：
1. 从`Cs137.out`文件读取器官剂量数据
2. 计算人体总质量
3. 计算总吸收能量
4. 计算加权平均剂量
5. 验证计算结果

**使用方法**：
```bash
cd 数据/数据处理/Dose
python dose.py
```

**输入文件**：
- `数据/Dose/ICRP145_HumanPhantomsAir/bin/Cs137.out`：器官剂量数据

**输出文件**：
- `dose_results.txt`：剂量计算结果
  - 事件数：1,000,000
  - 总质量：72.985700 kg
  - 总吸收能量：3.374063e-10 J
  - 加权平均剂量：4.622909e-18 Gy/source

### 3. MSL模块（Moseley定律分析）

#### 3.1 process_msl_data.py
**功能**：处理MSL实验数据，提取峰位信息。

**使用方法**：
```bash
cd 数据/数据处理/MSL
python process_msl_data.py
```

#### 3.2 高斯峰拟合.py
**功能**：对X射线谱进行高斯峰拟合，提取峰位、峰高、峰宽等参数。

**使用方法**：
```bash
cd 数据/数据处理/MSL
python 高斯峰拟合.py
```

#### 3.3 xray_element_tool.py
**功能**：X射线元素分析工具，包含元素数据库和特征X射线能量计算。

**使用方法**：
```bash
cd 数据/数据处理/MSL
python xray_element_tool.py
```

#### 3.4 msl_linearfit.py
**功能**：对Moseley定律进行线性拟合，验证√ν ∝ Z关系。

**使用方法**：
```bash
cd 数据/数据处理/MSL
python msl_linearfit.py
```

**输出文件**：
- `msl_linearfit_Sigma=2.py`：考虑σ=2的线性拟合
- `msl_peak_results.csv`：峰位拟合结果
- `moseley_law_fit.png`：Moseley定律拟合图

#### 3.5 plot_kalpha_vs_z.py
**功能**：绘制Kα线能量与原子序数的关系图。

**使用方法**：
```bash
cd 数据/数据处理/MSL
python plot_kalpha_vs_z.py
```

#### 3.6 plot_msl_vs_database.py
**功能**：将实验结果与数据库值进行对比。

**使用方法**：
```bash
cd 数据/数据处理/MSL
python plot_msl_vs_database.py
```

### 4. Compton模块（康普顿散射分析）

#### 4.1 Compton.py
**功能**：计算康普顿散射中电子吸收的能量，针对17keV的光子撞击静止电子。

**主要功能**：
1. 计算电子吸收能量随光子散射角度的变化
2. 计算电子反冲角（电子出射角）
3. 计算康普顿散射的微分散射截面（Klein-Nishina公式）
4. 计算总散射截面
5. 计算电子获得的平均能量
6. 生成多种可视化图表：
   - 电子能量随角度变化图（笛卡尔坐标和极坐标）
   - 能量贡献和微分散射截面图
   - 关键物理量的极坐标可视化

**物理公式**：
- 康普顿散射公式：E' = E₀ / [1 + (E₀/(mₑc²)) * (1 - cosθ)]
- 电子吸收能量：Eₑ = E₀ - E'
- 电子反冲角：cotφ = (1 + E₀/(mₑc²)) * tan(θ/2)
- Klein-Nishina微分散射截面：dσ/dΩ = (r₀²/2) * (E'/E₀)² * (E'/E₀ + E₀/E' - sin²θ)

**使用方法**：
```bash
cd 数据/数据处理
python Compton.py
```

**输出文件**：
- `Compton_scattering_plots.png`：包含4个子图的综合可视化结果

**关键计算结果**：
- 光子初始能量：17.0 keV
- 电子静止能量：511.0 keV
- 最大电子能量（θ=π）：1.06 keV（占初始能量的6.2%）
- 电子获得的平均能量：0.53 keV
- 总散射截面：约2.49×10⁻²⁵ cm²

**物理解释**：
1. θ=0（前向散射）：电子获得最小能量（0 keV）
2. θ=π（背向散射）：电子获得最大能量
3. 曲线关于θ=0对称
4. 对于17 keV光子，康普顿效应相对较小

## 数据处理流程

### 完整分析流程

1. **数据准备阶段**
   ```bash
   # 1. 读取原始数据
   cd 数据/数据处理/Absorb
   python ReadData.py
   
   # 2. 处理MSL数据
   cd ../MSL
   python process_msl_data.py
   ```

2. **衰减分析阶段**
   ```bash
   cd ../Absorb
   python decay_rate.py
   python peakdrift.py
   ```

3. **剂量计算阶段**
   ```bash
   cd ../Dose
   python dose.py
   ```

4. **Moseley定律分析阶段**
   ```bash
   cd ../MSL
   python 高斯峰拟合.py
   python msl_linearfit.py
   python plot_kalpha_vs_z.py
   python plot_msl_vs_database.py
   ```

5. **康普顿散射分析阶段**
   ```bash
   cd ..
   python Compton.py
   ```

## 关键结果

### 衰减分析结果
- **衰减系数范围**：0.013176 (Ag) 到 1.448859 (Ti)
- **平均衰减系数**：0.319270
- **log-log关系斜率**：-2.917287（与理论值-3基本一致）
- **结论**：验证了衰减系数 ∝ 能量^(-3)的经验定律

### 剂量计算结果
- **总质量**：72.985700 kg
- **总吸收能量**：3.374063e-10 J
- **加权平均剂量**：4.622909e-18 Gy/source
- **单次粒子吸收能量**：3.374063e-16 J/particle

### Moseley定律结果
- 验证了√ν ∝ Z的线性关系
- 获得了各元素的特征X射线峰位
- 与数据库值对比验证了实验准确性

### 康普顿散射分析结果
- **光子初始能量**：17.0 keV
- **电子静止能量**：511.0 keV
- **最大电子能量（θ=π）**：1.06 keV（占初始能量的6.2%）
- **电子获得的平均能量**：0.53 keV
- **总散射截面**：约2.49×10⁻²⁵ cm²
- **关键结论**：对于17 keV光子，康普顿效应相对较小，电子仅获得约6.2%的初始光子能量

## 注意事项

1. **Python环境**：所有程序需要Python 3.x环境，并安装以下库：
   - numpy
   - matplotlib
   - scipy
   - pandas（部分程序需要）

2. **文件路径**：程序中的文件路径为相对路径，请确保在正确的目录下运行程序。

3. **数据依赖**：部分程序依赖其他程序生成的中间文件，请按顺序执行。

4. **中文显示**：部分程序使用中文字体，确保系统支持中文字体显示。

## 更新日志

### 2025-12-15
- 添加Compton.py康普顿散射分析模块
- 更新README文档，添加Compton模块功能说明
- 更新目录结构，包含Compton.py和输出图像
- 更新数据处理流程，添加康普顿散射分析阶段
- 更新关键结果，添加康普顿散射分析结果

### 2025-12-08
- 创建完整的README文档
- 整理所有程序功能说明
- 提供完整的数据处理流程
- 汇总关键分析结果

### 程序更新记录
- `Compton.py`：新增康普顿散射分析程序，计算电子吸收能量和散射截面
- `decay_rate.py`：修复了图例排序问题，按衰减率从大到小排序
- `dose.py`：修复了中文编码问题，使用utf-8-sig编码
- 所有程序：优化了错误处理和用户提示

## 联系信息

如有问题或建议，请联系相关实验人员。

---
*本README最后更新于2025年12月15日*
