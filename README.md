# X射线标识谱实验数据处理系统

## 项目概述

本项目是一个自动化的X射线标识谱实验数据处理系统，能够根据实验数据自动生成实验报告中所需的所有图片。系统通过读取配置文件，按顺序调用相应的数据处理脚本，生成标准化的实验图片。

## 🚀 傻瓜式使用方法（三步搞定）

如果你只想快速生成实验报告图片，请按以下步骤操作：

### 准备工作：确保原始数据文件结构正确

北京大学近物实验"X射线标识谱"实验包含三个部分，原始数据应分别存储在以下三个目录中：

1. **莫塞莱实验数据**：`MSL\` 目录（包含 `MSL_N4000000_Ag.txt`, `MSL_N4000000_Fe.txt` 等文件）
2. **铝片吸收实验数据**：`Absorb\` 目录（包含各元素子目录，如 `Ag\`, `Co\`, `Cu\` 等）
3. **模拟人体吸收剂量实验数据**：`Dose\` 目录

**关键要求**：`MSL\`、`Absorb\`、`Dose\` 这三个目录必须在**同一父目录**下。

### 数据处理步骤

1. **下载自动化工具**：下载 [`YouOnlyNeedToDownloadThis.zip`](YouOnlyNeedToDownloadThis.zip) 压缩包
2. **解压文件**：将压缩包解压到 **与 `MSL\`、`Absorb\`、`Dose\` 目录相同的父目录**下
   - 解压后会自动创建 `数据处理\` 文件夹，其中包含 `Auto.py` 和所有必要的脚本
3. **运行程序**：打开命令行（CMD），进入 `数据处理\` 目录，运行以下命令：
   ```bash
   cd "数据处理"
   python Auto.py
   ```
   或者直接运行：
   ```bash
   python "数据处理\Auto.py"
   ```
4. **查看结果**：程序运行完成后，所有生成的图片将保存在 `数据处理\Figures\` 目录中

### 文件结构示意（示例）
```
任意目录\                      # 可以是任意目录，如 D:\实验数据\
├── MSL\                    # 莫塞莱实验原始数据
├── Absorb\                 # 铝片吸收实验原始数据  
├── Dose\                   # 模拟人体吸收剂量实验原始数据
└── 数据处理\               # 自动化处理工具（由YouOnlyNeedToDownloadThis.zip解压得到）
    ├── Auto.py            # 主程序
    ├── bin\               # 数据处理脚本
    └── Figures\           # 生成的图片保存目录
```

**重要提示**：只要 `MSL\`、`Absorb\`、`Dose\` 和 `数据处理\` 这四个目录在同一父目录下，程序就能正常工作。

> **注意**：运行前请确保已安装 Python 3.7+。如果遇到"ModuleNotFoundError"错误，请运行 `pip install numpy matplotlib pandas scipy` 安装所需依赖。

## 自动生成的图片

系统会自动生成以下10张实验报告图片：

| 图片序号 | 图片名称 | 生成脚本 | 描述 |
|---------|---------|---------|------|
| 图III.1.1 | 信号处理例 | `MSL/高斯峰拟合.py` | 展示高斯峰拟合的信号处理示例 |
| 图III.1.2 | 谱线与原子序数关系图 | `MSL/process_msl_data.py` | 展示谱线位置与原子序数的关系 |
| 图III.1.3 | 莫塞莱定律的验证 | `MSL/process_msl_data.py` | 验证莫塞莱定律的实验数据 |
| 图III.1.4 | 通过对比实验数据和资料，确定道系数和屏蔽系数 | `MSL/msl_linearfit_Sigma2.py` | 确定道系数和屏蔽系数的对比分析 |
| 图III.1.5 | 实验数据与相对论理论预言对比图 | `MSL/plot_msl_vs_database.py` | 实验数据与相对论理论预言的对比 |
| 图III.2.1 | 不同靶材特征谱线在铝板中的衰减 | `Absorb/decay_rate.py` | 不同靶材特征谱线在铝板中的衰减曲线 |
| 图III.2.2 | 不同靶材特征谱线在铝板中的衰减系数 | `Absorb/decay_rate.py` | 衰减系数的统计分析 |
| 图III.2.3 | 经验标度率的确定 | `Absorb/decay_rate.py` | 经验标度率的确定与分析 |
| 图V.1.1 | 相对论理论和非相对论理论与数据的对比 | `MSL/plot_kalpha_vs_z.py` | 相对论与非相对论理论与实验数据的对比 |
| 图VI-1.4.1 | 康普顿散射中电子获得的能量 | `Compton.py` | 康普顿散射中电子获得能量的分析 |

## 快速开始

### 1. 安装依赖

首先安装所需的Python包：

```bash
pip install numpy matplotlib pandas scipy
```

### 2. 一键生成所有图片

只需运行`Auto.py`程序，系统将自动生成所有实验报告图片：

```bash
python Auto.py
```

### 3. 运行结果

程序运行后，所有生成的图片将保存在`Figures/`目录中，文件名格式为`图片序号 - 图片名称.png`。

### 4. 查看生成的图片

生成的图片包括：
- `Figures/图III.1.1 - 信号处理例.png`
- `Figures/图III.1.2 - 谱线与原子序数关系图.png`
- `Figures/图III.1.3 - 莫塞莱定律的验证.png`
- `Figures/图III.1.4 - 通过对比实验数据和资料，确定道系数和屏蔽系数.png`
- `Figures/图III.1.5 - 实验数据与相对论理论预言对比图.png`
- `Figures/图III.2.1 - 不同靶材特征谱线在铝板中的衰减.png`
- `Figures/图III.2.2 - 不同靶材特征谱线在铝板中的衰减系数.png`
- `Figures/图III.2.3 - 经验标度率的确定.png`
- `Figures/图V.1.1 - 相对论理论和非相对论理论与数据的对比.png`
- `Figures/图VI-1.4.1 - 康普顿散射中电子获得的能量.png`

## 项目结构

```
.
├── README.md                    # 项目说明文档（本文件）
├── Auto.py                      # 主程序：自动生成所有图片
├── bin/                         # 数据处理脚本目录
│   ├── 实验报告图片生成对应表.csv # 图片生成配置表
│   ├── Absorb/                  # 吸收实验数据处理脚本
│   ├── MSL/                     # 莫塞莱定律实验数据处理脚本
│   ├── Compton.py               # 康普顿散射数据处理脚本
│   └── README.md                # 脚本说明文档
├── Figures/                     # 生成的图片保存目录
└── .gitignore                   # Git忽略文件配置
```

## 工作原理

Auto.py程序的工作流程如下：

1. **读取配置**：从`bin/实验报告图片生成对应表.csv`读取图片生成配置
2. **按顺序处理**：按照CSV文件中的顺序，依次处理每个图片生成任务
3. **动态导入**：根据配置中的"生成代码"字段，动态导入对应的Python模块
4. **调用生成函数**：调用每个模块的`generate_figures()`函数，传入图片名称参数
5. **保存图片**：将生成的图片保存到`Figures/`目录，文件名格式为`图片序号 - 图片名称.png`

## 配置说明

图片生成顺序和对应脚本由`bin/实验报告图片生成对应表.csv`文件控制：

```csv
图片序号,图片名称,生成代码
图III.1.1,信号处理例,MSL/高斯峰拟合.py
图III.1.2,谱线与原子序数关系图,MSL/process_msl_data.py
...
```

## 系统要求

- Python 3.7+
- 所需Python包：
  - numpy
  - matplotlib
  - pandas
  - scipy

## 常见问题

### Q1: 运行`python Auto.py`时出现"ModuleNotFoundError"错误
**A**: 请确保已安装所有必需的Python包。使用`pip install numpy matplotlib pandas scipy`安装依赖。

### Q2: 生成的图片在哪里？
**A**: 所有生成的图片保存在`Figures/`目录中，文件名格式为`图片序号 - 图片名称.png`。

### Q3: 如何修改图片生成顺序或添加新图片？
**A**: 编辑`bin/实验报告图片生成对应表.csv`文件，按照"图片序号,图片名称,生成代码"的格式添加或修改行。

### Q4: 为什么`Figures/`目录被.gitignore忽略？
**A**: 生成的图片文件可能较大且会频繁变化，因此被排除在版本控制之外以保持仓库清洁。如需共享图片，请使用其他方式。

## 注意事项

1. 首次运行前，请确保已安装所有必需的Python包
2. 生成的图片将保存在`Figures/`目录中，该目录已被`.gitignore`忽略
3. 如需修改图片生成顺序或添加新图片，请编辑`bin/实验报告图片生成对应表.csv`文件
4. 各数据处理脚本已针对实验数据进行了优化，如需调整参数请直接修改对应脚本

## 版本历史

- v1.1 (2025-12-16): 修复关键bug，优化用户体验
  - **修复图III.1.4的bug**：原`msl_linearfit_Sigma=2.py`文件名包含`=`字符导致导入失败而只能使用sigma非修正的版本；现已更正文件名及相关调用方法
  - **修复图III.1.5的bug**：找回`plot_msl_vs_database.py`并添加`generate_figures()`函数，修正了先前与图III.1.4完全一样的问题
  - **更新配置表**：同步更新`bin/实验报告图片生成对应表.csv`中的脚本路径
  - **添加傻瓜式使用方法与指南**：新增"傻瓜式使用方法"指南部分，提供从下载到运行的全流程指导，优化用户体验

- v1.0 (2025-12-15): 实现了全自动图像生成系统
  - 创建Auto.py主程序
  - 重构项目结构，将脚本组织到bin/目录
  - 添加.gitignore忽略生成的文件
  - 添加README文档

## 作者

北京大学 陈思源  
邮箱：siyuan_chen@stu.pku.edu.cn
电话：+86 181 0128 0283

## 附录：参考文件系统结构

以下是一个实际的文件系统结构示例（基于 `D:\课程作业\2025秋\近物实验\X射线标识谱\数据` 目录），供参考使用：

```
数据\
├── 数据处理\                    # 自动化处理工具目录
│   ├── Auto.py                # 主程序
│   ├── README.md              # 说明文档（本文件）
│   ├── YouOnlyNeedToDownloadThis.zip  # 压缩包
│   ├── bin\                   # 数据处理脚本
│   │   ├── 实验报告图片生成对应表.csv
│   │   ├── Absorb\           # 吸收实验脚本
│   │   ├── MSL\              # 莫塞莱定律实验脚本
│   │   └── Compton.py        # 康普顿散射脚本
│   └── Figures\              # 生成的图片保存目录
├── MSL\                       # 莫塞莱实验原始数据
│   ├── MSL_N4000000_Ag.txt   # 银(Ag)元素数据
│   ├── MSL_N4000000_Co.txt   # 钴(Co)元素数据
│   ├── MSL_N4000000_Cu.txt   # 铜(Cu)元素数据
│   ├── MSL_N4000000_Fe.txt   # 铁(Fe)元素数据
│   ├── MSL_N4000000_Mo.txt   # 钼(Mo)元素数据
│   ├── MSL_N4000000_Ni.txt   # 镍(Ni)元素数据
│   ├── MSL_N4000000_Se.txt   # 硒(Se)元素数据
│   ├── MSL_N4000000_Sr.txt   # 锶(Sr)元素数据
│   ├── MSL_N4000000_Ti.txt   # 钛(Ti)元素数据
│   ├── MSL_N4000000_Zn.txt   # 锌(Zn)元素数据
│   └── MSL_N4000000_Zr.txt   # 锆(Zr)元素数据
├── Absorb\                    # 铝片吸收实验原始数据
│   ├── Ag\                   # 银元素吸收数据（示例详细结构）
│   │   ├── clearAllResult.sh
│   │   ├── createRun.sh
│   │   ├── SAg_E22.1N1000000Sn0Layers.txt  # 0层铝片数据
│   │   ├── SAg_E22.1N1000000Sn1Layers.txt  # 1层铝片数据
│   │   ├── SAg_E22.1N1000000Sn2Layers.txt  # 2层铝片数据
│   │   ├── SAg_E22.1N1000000Sn4Layers.txt  # 4层铝片数据
│   │   ├── SAg_E22.1N1000000Sn8Layers.txt  # 8层铝片数据
│   │   ├── SAg_E22.1N1000000Sn16Layers.txt # 16层铝片数据
│   │   └── SAg_E22.1N1000000Sn32Layers.txt # 32层铝片数据
│   ├── Co\                   # 钴元素吸收数据（类似结构）
│   │   ├── SCo_E6.925N1000000Sn0Layers.txt  # 0层铝片数据
│   │   ├── SCo_E6.925N1000000Sn1Layers.txt  # 1层铝片数据
│   │   ├── SCo_E6.925N1000000Sn2Layers.txt  # 2层铝片数据
│   │   ├── SCo_E6.925N1000000Sn3Layers.txt  # 3层铝片数据
│   │   ├── SCo_E6.925N1000000Sn4Layers.txt  # 4层铝片数据
│   │   ├── SCo_E6.925N1000000Sn5Layers.txt  # 5层铝片数据
│   │   └── SCo_E6.925N1000000Sn6Layers.txt  # 6层铝片数据
│   ├── Cu\                   # 铜元素吸收数据（类似结构）
│   │   ├── SCu_E8.041N1000000Sn0Layers.txt
│   │   ├── SCu_E8.041N1000000Sn1Layers.txt
│   │   ├── SCu_E8.041N1000000Sn2Layers.txt
│   │   ├── SCu_E8.041N1000000Sn3Layers.txt
│   │   ├── SCu_E8.041N1000000Sn4Layers.txt
│   │   ├── SCu_E8.041N1000000Sn5Layers.txt
│   │   └── SCu_E8.041N1000000Sn6Layers.txt
│   ├── Fe\                   # 铁元素吸收数据（类似结构）
│   │   ├── SFe_E6.403N100000Sn0Layers.txt
│   │   ├── SFe_E6.403N100000Sn1Layers.txt
│   │   ├── SFe_E6.403N100000Sn2Layers.txt
│   │   ├── SFe_E6.403N100000Sn3Layers.txt
│   │   ├── SFe_E6.403N100000Sn4Layers.txt
│   │   ├── SFe_E6.403N100000Sn5Layers.txt
│   │   └── SFe_E6.403N100000Sn6Layers.txt
│   ├── Mo\                   # 钼元素吸收数据（类似结构）
│   ├── Ni\                   # 镍元素吸收数据（类似结构）
│   ├── Se\                   # 硒元素吸收数据（类似结构）
│   ├── Sr\                   # 锶元素吸收数据（类似结构）
│   ├── Ti\                   # 钛元素吸收数据（类似结构）
│   ├── Zn\                   # 锌元素吸收数据（类似结构）
│   └── Zr\                   # 锆元素吸收数据（类似结构）
└── Dose\                     # 模拟人体吸收剂量实验原始数据
    ├── B1\                   # B1剂量数据（本实验未使用）
    └── ICRP145_HumanPhantomsAir\  # ICRP145人体模型数据（实际使用的数据）
        ├── CMakeLists.txt
        ├── example_female.out
        ├── example_male.out
        ├── example.in
        ├── History
        ├── ICRP145phantoms.cc
        ├── ICRP145Phantoms.out
        ├── init_vis.mac
        ├── ionCs137.mac
        ├── README
        ├── README_general
        ├── source.mac
        ├── vis.mac
        ├── bin\               # 关键数据文件目录
        │   ├── Am241.out     # 镅-241剂量数据（实际使用）
        │   ├── clean.sh
        │   ├── cmake_install.cmake
        │   ├── CMakeCache.txt
        │   ├── Cs137.out     # 铯-137剂量数据（实际使用）
        │   ├── example.in
        │   ├── G4History.macro
        │   ├── ICRP145phantoms
        │   ├── init_vis.mac
        │   ├── ionAm241.out
        │   ├── ionCs137.mac
        │   ├── Makefile
        │   ├── Pu238.out     # 钚-238剂量数据（实际使用）
        │   ├── source.mac
        │   ├── vis.mac
        │   ├── CMakeFiles\
        │   ├── ICRP145data\
        │   └── ICRP145data-prefix\
        ├── include\
        └── src\
```

**重要说明**：
1. 此结构仅供参考，实际目录名称和位置可能有所不同
2. 关键要求：`MSL\`、`Absorb\`、`Dose\` 和 `数据处理\` 这四个目录必须在同一父目录下
3. 如果遇到文件找不到的错误，请检查实际文件路径是否与此结构匹配

## 许可证

本项目仅供学术研究使用。
