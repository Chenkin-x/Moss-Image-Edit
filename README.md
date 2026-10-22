<p align="center">
  <strong>简体中文</strong> &nbsp; · &nbsp; <a href="README.en.md">English</a>
</p>

<p align="center">
  <img src="assets/moss-image-edit-banner-cases.png" alt="MOSS Image Edit — 用文字编辑图像" width="1280">
</p>

<p align="center">
  <strong>一句话，让图像有新的表达。</strong><br>
  MOSS Image Edit 是用于 MOSS-Image 的 <strong>图像编辑 LoRA</strong>，按文字指令修改表情、穿搭与场景，支持单图编辑和多参考图输入。<br>
  发布内容为 LoRA 权重及配置，需要搭配 MOSS-Image Base 或 SFT 底模使用。
</p>

<p align="center">
  <a href="#编辑展示">编辑展示</a>
  &nbsp; · &nbsp;
  <a href="#快速开始">快速开始</a>
  &nbsp; · &nbsp;
  <a href="#comfyui">ComfyUI</a>
  &nbsp; · &nbsp;
  <a href="#模型架构">模型架构</a>
</p>

## 编辑展示

从颜色、穿搭到光照、季节与画风。每组左侧为原图，其余为编辑结果。

<p align="center">
  <img src="assets/moss-image-edit-showcase.jpg" alt="MOSS Image Edit：65 个精选编辑结果，原图与编辑结果成组展示，涵盖服装、颜色、物体、光照、季节及画风变化。" width="1280">
</p>

| 文字描述变化 | 图像提供参考 | 多图共同引导 |
|:---|:---|:---|
| 用自然语言说明要修改的内容，例如表情、服装或环境。 | 将已有图像作为编辑起点，结合指令生成结果。 | 最多使用 4 张参考图，组合人物、服饰与场景。 |

## 快速开始

下载 [MOSS Image Edit LoRA 权重](https://huggingface.co/chinoll/Moss-Image-Edit)，底模可选择 [MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base) 或 [MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT)，**两者均可使用，任选其一即可**。按 [ComfyUI 使用教程](https://github.com/Chenkin-x/ComfyUI-Moss-Image-Edit) 放好底模和 LoRA、导入工作流，即可开始编辑。

上传一张图片，描述你想要的变化，例如：**“把背景改成晴天的海边，保留人物。”** 也可以上传多张参考图，在指令中说明各张图的用途。

## ComfyUI

**上传参考图，写下变化，点击生成。** 节点独立维护于 [ComfyUI-Moss-Image-Edit](https://github.com/Chenkin-x/ComfyUI-Moss-Image-Edit)。安装与模型放置方法见 [ComfyUI 使用教程](https://github.com/Chenkin-x/ComfyUI-Moss-Image-Edit)，下载以下工作流后拖入页面即可打开。

| 单图编辑 | 多参考图编辑 |
|:---|:---|
| [下载单图工作流](https://github.com/Chenkin-x/ComfyUI-Moss-Image-Edit/blob/main/workflows/moss_edit_single_reference.json) | [下载多图工作流](https://github.com/Chenkin-x/ComfyUI-Moss-Image-Edit/blob/main/workflows/moss_edit_multi_reference.json) |
| 一张参考图与文字指令。 | 默认连接两张参考图，可扩展到四张。 |

工作流包含模型加载、图像输入、编辑和保存。可调整输出尺寸、随机种子和采样参数。

## 模型架构

**让参考信息贯穿编辑过程，并复用已经完成的参考图计算。** 模型在 MOSS 主干中引入参考图分支，用分离的时间调制和注意力可见关系，将它变成可缓存的视觉上下文。

<p align="center">
  <img src="assets/kv-cache-architecture.png" alt="Token 与时间输入：文本、参考图、加噪目标图分别形成 text、ref、target token，以 t、0、t 调制共享的 MOSS Transformer。" width="1400">
</p>

**不同输入，不同时间。** token 按 `[text, ref, target]` 排列。参考图保持无噪声，使用固定时间 `0`；文本与加噪目标图使用当前去噪时间 `t`。两条分支共享主干权重，时间调制分别生成各自的 shift、scale 和 gate。

<p align="center">
  <img src="assets/reference-kv-cache.png" alt="注意力与缓存：参考图只读取参考图，多张参考图之间相互可见；文本和目标图读取全部 token。参考分支计算一次，为每层保存 K/V，后续去噪步复用。" width="1400">
</p>

**参考图相互可见，编辑分支读取全部信息。** ref 的 query 只访问 ref 的 key/value，多张参考图之间可以交互；text 与 target 的 query 可以访问全部 token。因此参考图的逐层特征只依赖这组参考图，而编辑分支同时利用文字和视觉信息。

**一次构建，逐步复用。** 采样前，参考分支经过全部 Transformer 层，分别保存每层的 `K_ref / V_ref`。后续每个去噪步只更新 text/target，并读取对应层的缓存。参考分支不依赖文本，同一份缓存也可供 CFG 的正、负条件共用，减少重复计算。

## 贡献

感谢 [@onglon114514](https://github.com/onglon114514) 贡献展示图的生成。

---

<p align="center">
  <sub>Built on MOSS-Image · 本仓库代码采用 <a href="LICENSE">GNU GPLv3</a><br>
  底模与第三方依赖的许可证见各自项目。</sub>
</p>
