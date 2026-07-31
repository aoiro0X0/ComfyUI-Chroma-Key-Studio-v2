# ComfyUI Chroma Key Studio

一个仓库提供智能抠像背景选色与自适应 Keylight。安装一次即可获得两个主节点和原有五个参数节点。

## 包含节点

### Smart Chroma Background（智能抠像背景 V2.0）

节点 ID 保持为 `AutoChromaSmartBackground`，兼容旧工作流；同时保留 `KeylightSmartBackground` 别名。

- 只连接普通黑背景 `IMAGE` 即可，不要求透明底或 Mask。
- 自动排除从画面边缘连通的黑色/近黑背景，主体内部的黑色仍会保留。
- 优先选择纯绿、纯蓝、纯红，方便传统 Keylight 稳定抠像。
- 当三原色都会伤到主体时，才从 15° 间隔的高饱和色相中选择更安全的间色。
- 同时衡量全局颜色、局部色块和小面积高饱和显著色，避免红色小灯、Logo 等被面积统计忽略。
- 一个批次只输出一个 `#RRGGBB` 键色，视频帧不会各选各的颜色。
- 旧版的红、黄、绿、青、蓝禁用开关继续有效。

### Adaptive Chroma Keylight（V3.1.0）

节点 ID 保持为 `KeylightCoreHubV3`，旧工作流的输入位置和四个输出位置不变。

- 支持红、绿、蓝、青、黄、品红、紫色及任意高饱和中间色幕布。
- 使用去亮度的完整色相向量，不再把紫幕下的中性灰、白色、银色金属误判成背景。
- 黑位、低饱和彩色反光和主体内部黑色受到独立保护。
- `guided` 模式支持双通道间色与视频逐帧色偏，不再依赖单一最大 RGB 通道。
- Despill 与 Defringe 沿完整键色色度向量工作，中性灰不会被染色或误去色。
- `image_rgba` 始终使用干净前景 RGB 与 Alpha，即使主输出选择了彩色合成背景。

仓库还包含以下兼容参数节点：

- Key Sampler Args
- Key Edge Args
- Key Spill/Algo Args
- Key Protect Highlights Args
- Key Matte Math Args

## 安装

先把旧插件目录移出 `ComfyUI/custom_nodes`，不要与本仓库同时安装；重复的节点 ID 会导致加载顺序不确定。需要移走的旧目录可能包括：

- `ComfyUI-Mysterious-node1`
- `ComfyUI-Keylight-Smart-Background`
- `ComfyUI-Mysterious-node2`
- `ComfyUI-Smart-RGB-Keylight`

然后只克隆本仓库：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/aoiro0X0/ComfyUI-Chroma-Key-Studio.git
```

重启 ComfyUI。以后更新只需在该目录执行：

```bash
git pull
```

依赖只有 ComfyUI 已自带的 `torch` 与 `numpy`，不需要额外安装模型。

## 推荐连接

```text
黑背景主体图
  -> Smart Chroma Background.image

Smart Chroma Background.background_image
  -> 你的纯色背景合成/视频生成链路

Smart Chroma Background.color_hex (STRING)
  -> ColorInput
  -> Adaptive Chroma Keylight.key_color (COLORCODE)

待抠视频/图像
  -> Adaptive Chroma Keylight.image
```

你的原工作流中 `STRING -> AILab ColorInput -> COLORCODE` 的桥接可以原样保留。

## 推荐设置

- 纯色未发生变化：`key_mode = manual`，最确定。
- 视频模型让背景出现亮暗或轻微偏色：`key_mode = guided`。
- `shadow_recovery = 0.85`：恢复被压暗的彩色幕布，同时保护真黑主体。
- `edge_soft = 0.05`、`defringe = 0.07`：适合作为起点。
- `Key Edge Args` 等参数节点不是必连项；只有需要统一调参或覆盖主节点参数时再连接。

## 选色原则

不是固定只用三原色，也不是每次都用间色：

1. 纯绿、纯蓝、纯红中有安全色时，继续使用三原色。
2. 三者都与主体的重要颜色冲突，并且某个间色明显更安全时，才启用间色兜底。
3. 因为 Keylight V3.1 已同步支持任意色相，紫色等兜底色不会再误伤灰银金属。

## 已知限制

如果纯黑主体与纯黑背景完全融为一体，边界没有任何亮度或颜色差异，仅靠 RGB 像素无法恢复真实轮廓。上游最好保留微弱轮廓光。

## 来源与兼容

本仓库整合并兼容以下项目的节点合同：

- [muriellee1x/ComfyUI-Mysterious-node1](https://github.com/muriellee1x/ComfyUI-Mysterious-node1)
- [muriellee1x/ComfyUI-Mysterious-node2](https://github.com/muriellee1x/ComfyUI-Mysterious-node2)

统一版保留旧 mapping key、旧控件前缀、参数节点类型以及输出顺序，便于现有工作流直接迁移。
