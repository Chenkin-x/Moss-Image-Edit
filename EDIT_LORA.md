# 使用指南

本仓库提供 MOSS-Image 的编辑 LoRA 训练与推理。默认入口为 **NOOB2 + ConceptEdit 在线编码、FA4 Varlen、DeepSpeed Muon / ZeRO-2**。另保留 JSONL 预编码入口。

## 环境与底模

`requirements.txt` 按源训练环境整理。主要版本为 PyTorch 2.11、Transformers 5.5.3、DeepSpeed 0.19.6、Flash Attention 4.0.0b35、Optimum Quanto 0.2.7；Diffusers 固定到源环境使用的提交。

当前 FA4 实现来自 RTX 5090 训练环境，使用 FA4 的内部 backward 接口。更换 FA4 版本需要同步检查该接口。源环境使用 CUDA 12.8；本次本地整理未重新执行多卡训练。

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

`scripts/patch_deepspeed_muon.py` 修改**当前 Python 环境安装的 DeepSpeed 0.19.6**，备份原文件后，为 Muon 及其 ZeRO-2 调用链增加 `adjust_lr_fn`。默认使用 `match_rms_adamw`，正交化后的更新按 `0.2 * sqrt(max(rows, cols))` 缩放。只需在训练前执行一次，不会自动修改正在运行的训练。

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
python scripts/index_edit_sources.py --config configs/train.json
```

图像读取使用 Lance 批量 `take` 与 TAR 偏移读取。参考图和目标图根据参考图尺寸共同缩放，长边不超过 `max_edge`、宽高为 16 的倍数，不裁剪、不补边。VAE 对同一对图像一起编码；不同样本保持各自尺寸。

数据索引记录 Lance 版本。更换数据集版本或分片集合时，应使用新的 `data_index` 和 `noob_pair_index`，保留旧索引用于对应训练的恢复。

## 训练、采样与恢复

```bash
torchrun --standalone --nproc_per_node=4 \
  scripts/train_edit_zero2.py --config configs/train.json

# 从完整的 ZeRO-2 checkpoint 恢复
torchrun --standalone --nproc_per_node=4 \
  scripts/train_edit_zero2.py --config configs/train.json \
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

## 编辑机制

token 顺序为 `[text, ref, target]`：

| Query 来源 | 可读取的 Key / Value | 时间 |
|:---|:---|:---|
| text | text、ref、target | 当前 `t` |
| ref | ref | 固定 `0` |
| target | text、ref、target | 当前 `t` |

MOSS 的文本编码、text refiner、4D RoPE 与 target/text 时间调制保留。参考图增加独立位置标识，以 clean latent 参与每一步前向与反向。

训练仅计算 target 的速度损失，先对每张图的通道和空间维取均值，再对样本取均值，避免大图获得更高权重：

```python
t = (torch.randn(batch) * time_std + time_mean).sigmoid()
t = t.clamp(1e-5, 1 - 1e-5)
noisy_target = (1 - t) * target + t * noise
pred = model(noisy_target, t, text, clean_ref)
loss = mean_per_sample((pred - (noise - target)).square()).mean()
```

以上仅为算法示意；实际图像使用变长列表，见 `train_edit_zero2.py::flow_loss`。

FA4 前向通过 `mask_mod` 实现 mask，反向按 `ref→ref`、`text/target→all` 两个区域分别计算，并累加 ref 的 K/V 梯度。当前训练与 FA4 采样均重算 ref；这里没有跨去噪步的 KV cache。

`moss_edit_model.py` 的 dense 路径支持推理时 ref KV cache。只有该路径使用缓存，且要求 `eval()` 与 `no_grad()`。参考图只读取 ref，因此其缓存不依赖提示词或 target；更新模型或参考图后需重新构建缓存。

## JSONL 预编码入口

该入口使用 `configs/edit_lora.json`，与默认训练配置独立：单卡 / DDP、AdamW、逐样本计算，支持 1–4 张参考图。

```json
{"id":"edit-001","instruction":"Change the background to a sunny beach.","source_images":["images/source.png"],"target_image":"images/target.png"}
```

图片路径相对 JSONL 所在目录，更多例子见 `configs/edit_manifest.example.jsonl`。该入口分别缩放各张参考图和目标图；默认在线单图编辑入口使用同一缩放尺寸。

```bash
python scripts/prepare_edit_cache.py --config configs/edit_lora.json
python scripts/train_edit_lora.py --config configs/edit_lora.json

# 使用预编码配置执行 dense 推理，默认启用 ref KV cache
python scripts/sample_edit_lora.py --config configs/edit_lora.json \
  --adapter output/moss-edit-cached/checkpoint-00000500 \
  --references reference.png --prompt "Change the background to a sunny beach." \
  --height 1024 --width 1024 --output edited.png
```

预编码 checkpoint 的 LoRA 文件直接位于 checkpoint 目录；ZeRO-2 入口则位于其中的 `adapter/`。两种完整训练状态格式不同，续训使用各自训练入口。

## 代码导航

| 文件 | 用途 |
|:---|:---|
| `train_edit_zero2.py` | 默认多卡训练、采样、checkpoint |
| `moss_edit_varlen.py` | FA4 打包、mask 与对应反向 |
| `moss_edit_model.py` | 编辑模块与 dense ref KV cache |
| `moss_edit_lora.py` | LoRA 注入、导出、加载 |
| `moss_mixed_data.py` | NOOB2 / ConceptEdit 混合读取 |
| `index_edit_sources.py` | TAR 索引与 NOOB2 固定子集 |
| `moss_online_encode.py` | 冻结 VAE / Qwen 在线编码 |
| `moss_components.py` | 本地 pipeline 加载与文本条件 |
| `moss_quanto.py` | 冻结 Linear 的 INT8 量化 |
| `patch_deepspeed_muon.py` | DeepSpeed Muon RMS matching |
| `sample_edit_lora.py` | 独立编辑推理 |

代码文件均位于 `scripts/`。仓库代码采用 GPLv3，模型权重与第三方依赖使用各自许可证。
