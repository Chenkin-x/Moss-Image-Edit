# 使用指南

本仓库提供 MOSS-Image 的编辑 LoRA 训练与推理。默认入口为 **NOOB2 + ConceptEdit 在线编码、FA4 Varlen、DeepSpeed Muon / ZeRO-2**。另保留 JSONL 预编码入口。

## 环境与底模

`requirements.txt` 按源训练环境整理。主要版本为 PyTorch 2.11、Transformers 5.5.3、DeepSpeed 0.19.6、Flash Attention 4.0.0b35、Optimum Quanto 0.2.7；Diffusers 固定到源环境使用的提交。

所有命令均从仓库根目录运行。项目直接使用本地 `moss_edit` 包，无需安装项目本身。

模型目录需要包含官方 `pipeline_moss_image.py` 与完整权重：

```text
models/MOSS-Image-SFT/
├── pipeline_moss_image.py
├── transformer/
├── text_encoder/
├── tokenizer/
├── vae/
└── scheduler/
```

`tools/patch_deepspeed_muon.py` 修改**当前 Python 环境安装的 DeepSpeed 0.19.6**，备份原文件后，为 Muon 及其 ZeRO-2 调用链增加 `adjust_lr_fn`。默认使用 `match_rms_adamw`，正交化后的更新按 `0.2 * sqrt(max(rows, cols))` 缩放。只需在训练前执行一次，不会自动修改正在运行的训练。

## 在线数据

修改 `configs/train.json` 中的 `snapshot_dir`、`noob_root`、`conceptedit_root`、`data_index`、`noob_pair_index`、`output_dir`。所有相对路径均以配置文件所在目录为基准。

```text
data/
├── noob2/
│   ├── images.lance
│   └── pairs.lance
└── ConceptEdit-12M-webp90/
    ├── enhanced_prompt_random_resolution/
    │   └── *.tar
    └── .pipeline/completed/
        └── *.json
```

ConceptEdit 使用转换后的、未额外压缩的 TAR。每个完成标记 JSON 通过 `shard` 字段记录 TAR 文件名，例如 `{"shard": "00000.tar"}`。首次建索引时固定已完成的分片列表到 `data_index/selected_shards.json`。

TAR 内的样本 JSON 必须用 `images.source`、`images.edited` 指向实际图像成员名，指令取自 `instruction.detailed_en`。索引按这些明确的文件名配对，再记录字节偏移；不依赖图片出现顺序。

NOOB2 使用 `images.lance` 的 `img_id`、`data`，以及 `pairs.lance` 的 `pair_id`、`description`、`img_a_id`、`img_b_id`、`rev`。索引脚本从 `rev=0` 且指令非空的样本中抽取 `noob_subset_samples` 对，以 `noob_subset_seed` 固定抽样。已有子集文件会直接复用；修改数量或种子时，也请修改 `noob_pair_index` 文件名。

```bash
python prepare_data.py --config configs/train.json
```

图像读取使用 Lance 批量 `take` 与 TAR 偏移读取。参考图和目标图根据参考图尺寸共同缩放，长边不超过 `max_edge`、宽高为 16 的倍数，不裁剪、不补边。VAE 对同一对图像一起编码；不同样本保持各自尺寸。

数据索引记录 Lance 版本。更换数据集版本或分片集合时，应使用新的 `data_index` 和 `noob_pair_index`，保留旧索引用于对应训练的恢复。

## 训练、采样与恢复

```bash
torchrun --standalone --nproc_per_node=4 \
  train.py --config configs/train.json

# 从完整的 ZeRO-2 checkpoint 恢复
torchrun --standalone --nproc_per_node=4 \
  train.py --config configs/train.json \
  --resume output/moss-edit/checkpoint-00001000
```

`max_steps` 表示累计优化步数。global batch = 卡数 × `micro_batch_size` × `gradient_accumulation_steps`。更改卡数时同步修改配置，并为每个 rank 设置一条 `validation_keys`。

`validation_keys` 是 `[数据源, 样本索引]`：`0` 代表 NOOB2 子集，`1` 代表 ConceptEdit。默认四个 rank 各生成一张固定验证图，这些索引会从训练抽样中排除。采样目录必须能被所有 rank 共享访问。

```text
output/moss-edit/
├── resolved_config.json
├── metrics.jsonl
├── samples/step-00001000/           # source / generated / target 对比图与指令
└── checkpoint-00001000/
    ├── adapter/
    │   ├── adapter_config.json
    │   └── adapter_model.safetensors
    ├── rng-rank*.pt
    └── ...                        # DeepSpeed 训练状态
```

`sample_at_start` 控制是否在第 0 步采样；`sample_every`、`save_every` 分别控制后续采样和保存间隔。W&B 默认项目名为 `viola`；设 `wandb.enabled=false` 可关闭上传，本地指标、采样和 checkpoint 仍正常保存。

指标包含 target FM loss、采样时间均值、学习率、Muon 更新 RMS、在线编码和训练耗时、峰值已分配显存。`muon_update_norm` 是正交化后的更新范数，不是原始梯度范数。当前 ZeRO-2 入口关闭梯度裁剪。

## 参考图 KV cache

![参考图 KV cache 架构](../assets/kv-cache-architecture.png)

### 时间调制

ref 的 latent 保持 clean，主干中使用固定的 `t_ref=0`。时间网络在该输入下生成固定的 shift、scale 和 gate。`t=0` 是网络的输入值，得到的调制参数仍然是学习到的参数。

text 与 target 在主干中使用当前的 `t`。虽然文本编码器的输入文字不变，主干中的文本特征仍参与时间调制，并与不断变化的 target 交互，因此 text 与 target 都在每个去噪步重新计算。两条数据流使用同一套主干权重。

### 注意力可见关系

token 顺序为 `[text, ref, target]`。下表的行是 Query，列是 Key / Value；`1` 表示可见，`0` 表示屏蔽：

| Query \ Key / Value | text | ref | target |
|:---|:---:|:---:|:---:|
| text | 1 | 1 | 1 |
| ref | 0 | 1 | 0 |
| target | 1 | 1 | 1 |

每个可见块内部均为双向注意力。ref 的信息可以流向 text 与 target，而动态的 text / target 不会反过来更新 ref。结合固定时间调制，ref 的逐层隐藏状态在整个采样过程中保持一致。

参考图使用独立的图像位置标识和空间坐标。它的位置编码不随提示词长度和 target 状态变化，token 拼接时保留各自坐标。

### 构建与复用

`moss_edit/model.py` 的 `prepare_cache()` 将参考 tokens 独立送入全部 Transformer 层。在每层计算归一化、时间调制、Q/K/V 投影与位置编码后，保存该层的 K/V；ref 隐藏状态继续经过参考图内部的 attention 和 MLP，送入下一层。

每层保存的是该层的 `K_ref` 与 `V_ref`，其中 K 已经过 QK normalization 和 RoPE。去噪时，当前层的 text/target 计算自己的 Q/K/V，再按 `[text, ref, target]` 顺序拼入缓存的参考 K/V。只更新 text/target 隐藏状态，最后读取 target 的速度预测。

`moss_edit/sampling.py` 在采样循环前构建缓存，之后所有去噪步共用该缓存。CFG 的两个分支只改变文本条件，ref 不依赖文本，因此也共用同一份缓存。参考图或模型权重变化后重新构建。

[`configs/infer.json`](../configs/infer.json) 用于独立编辑推理，默认启用缓存。`--no-kv-cache` 可用于对比逐步重算参考图的结果。

### 训练目标

训练时参考分支参与反向传播。只对 target 加噪并计算速度 MSE，先在每张图内部取均值，再对样本取均值：

```python
t = (torch.randn(batch) * time_std + time_mean).sigmoid()
t = t.clamp(1e-5, 1 - 1e-5)
noisy_target = (1 - t) * target + t * noise
pred = model(noisy_target, t, text, clean_ref)
loss = mean_per_sample((pred - (noise - target)).square()).mean()
```

这是算法示意；实际的变长图像使用列表，计算见 `train.py::flow_loss`。

## JSONL 预编码入口

该入口使用 `configs/cached.json`，与默认训练配置独立：单卡 / DDP、AdamW、逐样本计算，支持 1–4 张参考图。

```json
{"id":"edit-001","instruction":"Change the background to a sunny beach.","source_images":["images/source.png"],"target_image":"images/target.png"}
```

图片路径相对 JSONL 所在目录，更多例子见 `configs/manifest.example.jsonl`。该入口分别缩放各张参考图和目标图；默认在线单图编辑入口使用同一缩放尺寸。

```bash
python -m tools.prepare_cache --config configs/cached.json
python -m tools.train_cached --config configs/cached.json

# 使用预编码配置推理，默认启用 ref KV cache
python infer.py --config configs/cached.json \
  --adapter output/moss-edit-cached/checkpoint-00000500 \
  --references reference.png --prompt "Change the background to a sunny beach." \
  --height 1024 --width 1024 --output edited.png
```

预编码 checkpoint 的 LoRA 文件直接位于 checkpoint 目录；ZeRO-2 入口则位于其中的 `adapter/`。两种完整训练状态格式不同，续训使用各自训练入口。

## 代码导航

| 文件 | 用途 |
|:---|:---|
| `train.py` | 多卡训练、在线预览与 checkpoint |
| `infer.py` | 编辑推理命令行入口 |
| `prepare_data.py` | TAR 索引与 NOOB2 固定子集 |
| `moss_edit/model.py` | 编辑模型、参考图 prefill 与逐层 KV cache |
| `moss_edit/attention.py` | 变长打包、注意力 mask 与反向 |
| `moss_edit/lora.py` | LoRA 注入、导出、加载 |
| `moss_edit/data.py` | NOOB2 / ConceptEdit 混合读取 |
| `moss_edit/encoding.py` | 冻结 VAE / Qwen 在线编码 |
| `moss_edit/processing.py` | 图像与 latent 处理、预编码数据读写 |
| `moss_edit/components.py` | 配置、底模加载与文本条件 |
| `moss_edit/sampling.py` | 训练预览和独立推理共用的去噪循环 |
| `moss_edit/quantization.py` | 冻结 Linear 的 INT8 量化 |
| `tools/patch_deepspeed_muon.py` | DeepSpeed Muon RMS matching |
| `tools/prepare_cache.py` | JSONL 数据的 VAE / Qwen 预编码 |
| `tools/train_cached.py` | 基于预编码数据的单卡 / DDP 训练 |

仓库代码采用 GPLv3，模型权重与第三方依赖使用各自许可证。
