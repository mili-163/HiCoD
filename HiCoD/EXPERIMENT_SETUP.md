# HiCoD 实验设置说明

## 实验场景概述

HiCoD框架支持两种不同的缺失模态场景，完全按照论文设置实现：

### 1. 固定缺失模态场景 (Fixed Missing Modality Scenario)

系统性地在整个评估过程中丢弃特定模态：

#### 单模态缺失
- **{L}**: 仅缺失语言模态 (text)
- **{A}**: 仅缺失音频模态 (audio)  
- **{V}**: 仅缺失视觉模态 (vision)

#### 双模态缺失
- **{L, A}**: 缺失语言和音频模态
- **{L, V}**: 缺失语言和视觉模态
- **{A, V}**: 缺失音频和视觉模态

### 2. 随机缺失模态场景 (Random Missing Modality Scenario)

随机选择缺失模态，通过缺失率量化整体缺失程度：

#### 缺失率定义
```
r_miss = (1 - Σ(m_i)/(N×M)) × 100%
```

其中：
- **N**: 样本总数
- **M**: 模态总数 (3个模态: text, audio, vision)
- **m_i**: 第i个样本的可用模态数量

#### 缺失率范围
- **8个缺失率值**: [0%, 10%, 20%, 30%, 40%, 50%, 60%, 70%]
- **70%**: 最大近似缺失率，确保任何时刻至少有一个模态可用

## 实验运行方式

### 综合实验 (推荐)

运行所有实验场景：

```bash
# 运行所有实验 (固定缺失 + 随机缺失)
python experiment_comprehensive.py --dataset mosi --seeds 1111 2222 3333

# 仅运行固定缺失模态实验
python experiment_comprehensive.py --dataset mosi --scenario fixed --seeds 1111 2222 3333

# 仅运行随机缺失模态实验  
python experiment_comprehensive.py --dataset mosi --scenario random --seeds 1111 2222 3333
```

### 单独实验

运行特定场景的实验：

```bash
# 固定缺失模态场景
python run.py --model hicod --dataset mosi --scenario fixed --missing_modalities text
python run.py --model hicod --dataset mosi --scenario fixed --missing_modalities text audio

# 随机缺失模态场景
python run.py --model hicod --dataset mosi --scenario random --mr 0.3
```

## 实验配置

### 数据集支持
- **MOSI**: CMU Multimodal Opinion Sentiment and Intensity
- **MOSEI**: CMU Multimodal Opinion Sentiment, Emotions and Attributes

### 随机种子
- 默认种子: [1111, 2222, 3333]
- 可自定义种子列表确保实验可重现性

### 结果保存
- 实验结果自动保存到 `results/hicod_{dataset_name}/` 目录
- 支持JSON格式的结果文件
- 综合实验结果包含所有场景的汇总

## 实验验证

### 验证缺失模态逻辑
```bash
python validate_missing_logic.py
```

验证内容：
- 固定缺失模态场景的正确性
- 随机缺失模态场景的概率性
- 缺失率计算公式的准确性

### 验证完整流程
```bash
python validate_simple_model.py
python validate_complete_pipeline.py
```

验证内容：
- 模型前向传播
- 缺失模态处理
- 训练组件功能

## 实验输出

### 固定缺失模态实验
- 6个实验场景的结果
- 每个场景包含多种子的平均性能
- 保存到 `fixed_missing_experiments.json`

### 随机缺失模态实验  
- 8个缺失率的结果
- 每个缺失率包含多种子的平均性能
- 保存到 `random_missing_experiments.json`

### 综合实验结果
- 包含所有实验场景的汇总结果
- 实验信息元数据
- 保存到 `comprehensive_experiments.json`

## 性能指标

根据论文描述，HiCoD在以下基准测试中取得了SOTA结果：

- **CMU-MOSI**: 提升Acc-2达6.4点
- **CMU-MOSEI**: 在固定和随机缺失设置下都表现优异
- **其他基准**: 在多个数据集上验证了有效性

## 注意事项

1. **确保至少一个模态可用**: 在随机缺失场景中，系统确保任何时刻至少有一个模态可用
2. **实验可重现性**: 使用固定随机种子确保实验结果可重现
3. **资源需求**: 根据数据集大小和模型复杂度调整计算资源
4. **结果验证**: 运行测试脚本验证实验设置的正确性 