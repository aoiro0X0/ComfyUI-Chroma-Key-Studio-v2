# ComfyUI Chroma Key Studio V2

一个仓库提供智能抠像背景选色与自适应 Keylight。安装一次即可获得两个主节点和五个参数节点。

V2 使用独立的仓库名、节点 ID、Python 命名空间、前端扩展名和颜色类型，因此可以与旧的 node1、node2、Smart Background、Smart RGB Keylight 同时安装，不会抢占旧节点。五类 Args 参数插口沿用旧版数据合同，已有参数节点可以直接连接 V2 Keylight。

## 包含节点

| 菜单显示名 | V2 节点 ID |
| --- | --- |
| Smart Chroma Background (Studio V2) | `ChromaKeyStudioSmartBackgroundV2` |
| Adaptive Chroma Keylight (Studio V2) | `ChromaKeyStudioKeylightV2` |
| Key Spill/Algo Args (Studio V2) | `ChromaKeyStudioSpillArgsV2` |
| Key Protect Highlights Args (Studio V2) | `ChromaKeyStudioProtectHighlightsArgsV2` |
| Key Edge Args (Studio V2) | `ChromaKeyStudioEdgeArgsV2` |
| Key Matte Math Args (Studio V2) | `ChromaKeyStudioMatteMathArgsV2` |
| Key Sampler Args (Studio V2) | `ChromaKeyStudioSamplerArgsV2` |

所有节点都在 `Chroma Key Studio V2` 分类下。

### Smart Chroma Background

- 普通黑背景 `IMAGE` 可以直接输入，不要求透明底或 Mask。
- 自动排除从画面边缘连通的黑色或近黑背景；主体内部未与边缘连通的黑色仍会参与分析。
- 优先选择纯绿、纯蓝、纯红，方便传统 Keylight 稳定抠像。
- 当三原色都会伤到主体时，才从 15° 间隔的高饱和色相中选择更安全的间色。
- 同时衡量全局颜色、局部色块和小面积高饱和显著色，避免红色小灯、Logo 等被面积统计忽略。
- 一个批次只输出一个键色，视频帧不会各选各的颜色。
- 红、黄、绿、青、蓝禁用开关继续可用。

### Adaptive Chroma Keylight

- 支持红、绿、蓝、青、黄、品红、紫色及任意高饱和中间色幕布。
- 使用去亮度的完整色相向量，避免把紫幕下的中性灰、白色、银色金属误判成背景。
- 黑位、低饱和彩色反光和主体内部黑色受到独立保护。
- `guided` 模式支持双通道间色与视频逐帧色偏。
- Despill 与 Defringe 沿完整键色色度向量工作。
- 默认启用批量自适应运动边缘清理：根据邻近干净前景与实际键色反推混合像素，同时修正 Alpha 与受污染 RGB；不需要新增节点、人工预设或逐素材调参。
- 通过可信前景色、方向一致的多级混合剖面和拟合残差共同确认运动边缘；证据不足时回退到温和处理，避免把主体内部颜色、高光以及不相关的火焰、烟雾统一粗暴缩边。
- 长视频 IMAGE 批次会在节点内部自动分块，保持帧顺序和结果一致，同时限制自适应分析的峰值显存。
- `image_rgba` 始终使用干净前景 RGB 与 Alpha，即使主输出选择了彩色合成背景。

五个 Args 节点都不是必连项；只有需要集中覆盖高级参数时再连接。旧版的 Key Edge、Protect Highlights、Spill/Algo、Matte Math 和 Sampler Args 也可直接连接新版主节点。

批量生产建议不连接任何 Edge 或 Protect Highlights Args，直接使用主节点默认值。自适应清理会逐帧判断清晰硬边、运动模糊、半透明边缘与高光，但整套工作流保持同一入口，无需中途人工切换参数。

## 安装

旧插件目录可以保留。只需克隆 V2 仓库：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/aoiro0X0/ComfyUI-Chroma-Key-Studio-v2.git
```

重启 ComfyUI。以后更新：

```bash
cd ComfyUI/custom_nodes/ComfyUI-Chroma-Key-Studio-v2
git pull
```

依赖只有 ComfyUI 已自带的 `torch` 与 `numpy`，不需要额外安装模型。

不要在 `custom_nodes` 中放置两份 V2 仓库副本；它们会注册相同的 V2 节点 ID。

## 推荐连接

```text
黑背景主体图
  -> Smart Chroma Background (Studio V2).image

Smart Chroma Background (Studio V2).color_hex
  -> Adaptive Chroma Keylight (Studio V2).key_color

使用该纯色生成或合成后的待抠图像/视频
  -> Adaptive Chroma Keylight (Studio V2).image
```

V2 的颜色输出和 Keylight 输入可以直接相连，不再需要中间的 AILab ColorInput。

## 推荐设置

- 纯色未发生变化：`key_mode = manual`，最确定。
- 视频模型让背景出现亮暗或轻微偏色：`key_mode = guided`。
- `shadow_recovery = 0.85`：恢复被压暗的彩色幕布，同时保护真黑主体。
- `edge_soft = 0.05`、`defringe = 0.07`：配合默认自适应清边，适合批量生产直接使用。

## 选色原则

1. 纯绿、纯蓝、纯红中有安全色时，继续使用三原色。
2. 三者都与主体的重要颜色冲突，并且某个间色明显更安全时，才启用间色兜底。
3. V2 Keylight 支持任意色相，因此紫色等兜底色也能按实际键色抠除。

## 从旧节点迁移

旧工作流继续使用旧节点，不会被 V2 静默接管。需要升级某个工作流时，请手动添加对应的 `(Studio V2)` 节点并重新连接；确认效果后，旧插件是否保留由你决定。

这是安全共存与旧工作流不被改变的必要取舍：不能同时让同一个旧节点 ID 自动切换到 V2，又保证两版独立运行。

## 已知限制

如果纯黑主体与纯黑背景完全融为一体，边界没有任何亮度或颜色差异，仅靠 RGB 像素无法恢复真实轮廓。上游最好保留微弱轮廓光。

单帧中若真实的不透明渐变恰好逐级复制“键色到主体色”的完整混合轨迹，数学上会与运动模糊产生相同像素，任何只看该帧 RGB 的算法都不能绝对区分。V2 会要求同方向的低、中、高 Alpha 证据来降低误判；Smart Chroma Background 同时应继续避开主体重要颜色，作为批量流程的第一层保护。

## 来源

V2 整合并迭代自以下项目的节点逻辑：

- [muriellee1x/ComfyUI-Mysterious-node1](https://github.com/muriellee1x/ComfyUI-Mysterious-node1)
- [muriellee1x/ComfyUI-Mysterious-node2](https://github.com/muriellee1x/ComfyUI-Mysterious-node2)

V2 保留主要输入输出语义，但不注册任何旧 mapping ID，以保证新旧插件可以共存。
