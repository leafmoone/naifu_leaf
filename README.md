# Naifu-Neta训练器

一个功能强大、模块化的深度学习训练框架，专门用于训练各种生成模型，包括Stable Diffusion、PixArt、LLaVA、Stable Cascade等多种先进的AI模型。该项目基于PyTorch Lightning构建，提供了灵活的配置系统和高效的分布式训练能力。

## 🌟 主要特性

### 🎯 多模型支持
- **Stable Diffusion系列**: SDXL、SD1.5、Refiner等
- **文本到图像模型**: PixArt-α/σ、Stable Cascade
- **多模态模型**: LLaVA (视觉语言模型)
- **语言模型**: GPT-2、Phi-2、Mistral等
- **控制网络**: ControlNet、IP-Adapter
- **高级训练**: LoRA、LyCORIS、DPO等

### ⚡ 高性能训练
- **分布式训练**: 支持DDP、FSDP、DeepSpeed策略
- **混合精度**: 16-bit、BF16混合精度训练
- **内存优化**: 梯度检查点、8-bit优化器
- **高效数据加载**: 多分辨率桶采样、潜在空间缓存

### 🔧 灵活配置
- **YAML配置系统**: 模块化、可复用的配置文件
- **动态模块加载**: 支持自定义模型和数据处理器
- **多种优化器**: AdamW、8-bit优化器、自适应学习率
- **调度器支持**: 常数、余弦、线性等多种学习率调度

### 📊 完整工具链
- **数据预处理**: 图像编码、标签生成、数据清洗
- **自动标注**: WD14 Tagger、DeepDanbooru支持
- **实时监控**: Wandb集成、CSV日志记录
- **模型推理**: 内置采样和生成脚本

## 🚀 快速开始

### 环境要求
- Python 3.9+
- CUDA 11.8+ (推荐)
- 16GB+ GPU内存 (取决于模型和批次大小)

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/your-repo/naifu-neta_noob.git
cd naifu-neta_noob

# 安装依赖
pip install -r requirements.txt

# 可选：安装xformers以获得更好的性能
pip install xformers
```

### 基础使用

```bash
# 训练SDXL模型
python trainer.py config/train_sdxl.yaml

# 训练PixArt模型
python trainer.py config/train_pixart.yaml

# 训练LLaVA多模态模型
python trainer.py config/train_llava.yaml

# 使用自定义配置
python trainer.py --config your_config.yaml
```

## 📋 支持的模型类型

### 图像生成模型

#### Stable Diffusion XL (SDXL)
- **配置文件**: `config/train_sdxl.yaml`
- **特性**: 1024x1024高分辨率生成、双文本编码器
- **支持**: LoRA、ControlNet、IP-Adapter、DPO训练

```yaml
# 基础SDXL训练配置示例
name: sdxl-training
target: modules.train_sdxl.setup

trainer:
  model_path: sd_xl_base_1.0_0.9vae.safetensors
  batch_size: 4
  max_epochs: 60
  
dataset:
  name: data.bucket.AspectRatioDataset
  img_path: "/path/to/your/images"
  target_area: 1_048_576  # 1024x1024
```

#### PixArt-α/σ
- **配置文件**: `config/train_pixart.yaml`
- **特性**: Transformer架构、高质量文本到图像生成
- **优势**: 更好的文本理解和图像质量

#### Stable Cascade
- **配置文件**: `config/train_cascade_stage_c.yaml`
- **特性**: 多阶段生成、高效的潜在空间表示
- **组件**: Stage A/B/C分阶段训练

### 多模态模型

#### LLaVA (Large Language and Vision Assistant)
- **配置文件**: `config/train_llava.yaml`
- **特性**: 视觉问答、图像描述、多模态对话
- **支持**: LoRA微调、视觉塔训练

```yaml
# LLaVA训练配置示例
model_config:
  version: v1
  vision_tower: openai/clip-vit-large-patch14-336
  mm_projector_type: mlp2x_gelu
  tune_mm_vision_tower: true
  
use_lora: true
lora_params:
  r: 128
  lora_alpha: 256
```

### 语言模型
- **GPT-2**: `config/train_gpt2.yaml`
- **Phi-2**: `config/train_phi2.yaml`
- **Mistral**: `config/train_mistral_lora.yaml`

## 🗂️ 项目结构

```
naifu-neta_noob/
├── trainer.py              # 主训练脚本
├── requirements.txt        # 依赖列表
├── config/                 # 配置文件目录
│   ├── train_sdxl.yaml    # SDXL训练配置
│   ├── train_pixart.yaml  # PixArt训练配置
│   ├── train_llava.yaml   # LLaVA训练配置
│   └── ...                # 其他模型配置
├── common/                 # 通用组件
│   ├── trainer.py         # 训练器核心逻辑
│   ├── utils.py           # 工具函数
│   └── logging.py         # 日志系统
├── modules/                # 模型模块
│   ├── train_sdxl.py      # SDXL训练模块
│   ├── train_pixart.py    # PixArt训练模块
│   ├── sdxl_model.py      # SDXL模型定义
│   └── ...                # 其他模型模块
├── data/                   # 数据处理
│   ├── bucket.py          # 桶采样数据集
│   ├── processors.py      # 数据预处理器
│   └── image_storage.py   # 图像存储系统
├── models/                 # 模型架构
│   ├── sgm/               # Stable Diffusion模型
│   ├── pixart/            # PixArt模型
│   ├── llava/             # LLaVA模型
│   └── ...                # 其他模型架构
├── scripts/                # 实用脚本
│   ├── wd14_tagger.py     # WD14自动标注
│   ├── deepdanbooru.py    # DeepDanbooru标注
│   └── ...                # 其他工具脚本
└── data_loader/            # 数据加载工具
    ├── csv2arrow.py       # CSV转Arrow格式
    ├── build_yaml.py      # 配置文件生成
    └── ...                # 其他数据工具
```

## 📊 数据处理

### 数据格式支持
- **图像格式**: JPG、PNG、WebP、TIFF等
- **标注格式**: TXT文本文件、JSON、CSV
- **存储格式**: 原始图像、潜在空间缓存(H5)

### 桶采样系统
项目实现了智能的桶采样系统，支持多分辨率训练：

```python
# 桶采样配置示例
dataset:
  name: data.bucket.AspectRatioDataset
  target_area: 1_048_576    # 目标像素数
  min_size: 512             # 最小尺寸
  max_size: 2048            # 最大尺寸
  img_path: "/path/to/images"
```

### 数据预处理工具

```bash
# 使用WD14 Tagger自动生成标签
python scripts/wd14_tagger.py --path /path/to/images --threshold 0.5

# 构建多分辨率数据索引
idk multireso -c dataset/yamls/config.yaml -t dataset/jsons/output.json

# CSV转Arrow格式（高效数据加载）
python data_loader/csv2arrow.py input.csv output_dir
```

## ⚙️ 高级配置

### 分布式训练

```yaml
lightning:
  accelerator: gpu
  devices: -1              # 使用所有可用GPU
  strategy: ddp            # 分布式数据并行
  precision: 16-mixed      # 混合精度训练
```

### DeepSpeed集成

```yaml
lightning:
  strategy: deepspeed
  strategy_params:
    stage: 2               # DeepSpeed ZeRO Stage 2
    offload_optimizer: true
    offload_parameters: true
```

### 内存优化

```yaml
advanced:
  use_checkpoint: true     # 梯度检查点
  vae_encode_batch_size: 1 # VAE编码批次大小
  
optimizer:
  name: bitsandbytes.optim.AdamW8bit  # 8-bit优化器
```

### 学习率调度

```yaml
scheduler:
  name: transformers.get_cosine_schedule_with_warmup
  params:
    num_warmup_steps: 1000
    num_training_steps: 10000
```

## 🔧 实用工具

### 自动标注工具

```bash
# WD14 Tagger - 动漫风格图像标注
python scripts/wd14_tagger.py \
  --path /path/to/images \
  --interrogator wd14-swinv2-v2 \
  --threshold 0.5

# DeepDanbooru - 另一种动漫标注工具
python scripts/deepdanbooru.py --input_dir /path/to/images
```

### 数据处理工具

```bash
# 构建训练数据索引
python data_loader/build_yaml.py --input_dir /path/to/data

# 数据清洗和验证
python data_loader/data_clean.py --data_dir /path/to/data
```

### 模型推理

```bash
# SDXL推理
python scripts/sdxl_inference.py \
  --model_path checkpoint/model.safetensors \
  --prompt "a beautiful landscape"

# LLaVA推理
python scripts/run_llava.py \
  --model_path checkpoint/llava_model \
  --image_path image.jpg \
  --question "What do you see in this image?"
```

## 📈 监控和日志

### Wandb集成

```yaml
trainer:
  wandb_id: "your-project-name"
  wandb_entity: "your-team"  # 可选
```

### 检查点管理

```yaml
trainer:
  checkpoint_dir: checkpoint
  checkpoint_freq: 1        # 每个epoch保存
  checkpoint_steps: 1000    # 每1000步保存
  save_weights_only: true   # 只保存权重
  save_format: safetensors  # 使用safetensors格式
```

## 🎯 训练技巧和最佳实践

### 1. 内存优化
- 使用梯度检查点减少内存使用
- 启用8-bit优化器
- 调整批次大小和累积梯度步数

### 2. 训练稳定性
- 使用混合精度训练
- 启用梯度裁剪
- 合理设置学习率和预热步数

### 3. 数据质量
- 使用高质量的训练数据
- 合理的标签和描述
- 数据增强和预处理

### 4. 超参数调优
- 从小批次开始测试
- 监控损失曲线和生成质量
- 使用验证集评估模型性能

## 🤝 贡献指南

我们欢迎社区贡献！请遵循以下步骤：

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [Stability AI](https://stability.ai/) - Stable Diffusion模型
- [Hugging Face](https://huggingface.co/) - Transformers和Diffusers库
- [PyTorch Lightning](https://lightning.ai/) - 训练框架
- [Microsoft](https://github.com/microsoft/DeepSpeed) - DeepSpeed优化

## 📞 支持

如果您遇到问题或有疑问，请：

1. 查看[Issues](https://github.com/your-repo/naifu-neta_noob/issues)
2. 阅读文档和配置示例
3. 在社区论坛寻求帮助

---

**注意**: 本项目仍在积极开发中，API和配置可能会发生变化。请关注更新日志和发布说明。
