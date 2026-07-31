# Smart RGB Keylight V3.0.0

面向“智能选择纯红/纯绿/纯蓝背景 → 视频生成 → Keylight 抠像”的 ComfyUI 节点。保留原节点和旧工作流兼容，并增加引导式逐帧校色、暗色幕布恢复、内置边缘处理以及稳定的 RGB 去溢色。

## 推荐接法

```text
[Smart RGB Background image] → [视频生成] → [Smart RGB Keylight image]
[Smart RGB Background color_hex] → [ColorInput] → [Smart RGB Keylight key_color]
```

`key_mode` 推荐使用默认的 `guided`。它不会盲目猜测背景，而是以上游传入的纯红、纯绿或纯蓝为锚点，仅从每帧边缘选取方向相近的像素进行校色。

## 安装

将 `KeylightChromaKeyHub/` 文件夹放入 `ComfyUI/custom_nodes/` 目录，然后重启 ComfyUI。

**依赖**：无需额外安装（仅使用 ComfyUI 自带的 PyTorch）

> 如果节点不显示，请检查 ComfyUI 启动控制台是否有报错，并尝试更新 ComfyUI 或浏览器硬刷新/清缓存。

---

## 节点概览

本节点组包含 **1 个核心节点** + **5 个参数节点**，采用模块化设计：

| 节点名称 | 类型 | 作用 |
|---------|------|------|
| Keylight Core Hub | 核心节点 | 执行色键抠像的主节点 |
| Key Sampler Args | 参数节点 | 配置自动键颜色采样方式 |
| Key Edge Args | 参数节点 | 控制蒙版边缘处理 |
| Key Spill/Algo Args | 参数节点 | 配置溢色抑制算法 |
| Key Protect Highlights Args | 参数节点 | 保护高光区域 |
| Key Matte Math Args | 参数节点 | 蒙版后处理数学运算 |

---

## 节点详细说明

### 1. Keylight Core Hub (核心节点)

**显示名称**: `Keylight Core (Hub V2.3.6fixE3_clean_final)`

色键抠像的核心处理节点，所有参数节点都连接到此节点。

#### 输入参数

| 参数 | 类型 | 默认值 | 范围 | 说明 |
|-----|------|-------|------|------|
| `image` | IMAGE | - | - | 输入的绿幕/蓝幕图像 |
| `key_mode` | 选择 | guided | guided/manual/auto | 引导校色、严格手动或完全自动 |
| `key_color` | 颜色 | #00FF00 | - | 手动模式下的键颜色（绿色） |
| `background_mode` | 选择 | alpha | alpha/color/soft_color | 背景处理模式 |
| `bg_color` | 颜色 | #000000 | - | 替换背景颜色 |
| `tolerance` | FLOAT | 1.0 | 0.0-2.0 | 色相偏离惩罚；越大越保护主体、越小越能清理偏色幕布 |
| `clip_black` | FLOAT | -0.02 | -1.0-1.0 | 黑场裁切，控制透明区域阈值 |
| `clip_white` | FLOAT | 0.30 | 0.0-2.0 | 白场裁切，控制不透明区域阈值 |
| `shadow_recovery` | FLOAT | 0.85 | 0.0-1.0 | 恢复被生成模型压暗的有色幕布；纯黑仍保留 |
| `edge_soft` | FLOAT | 0.05 | 0.0-1.0 | 主节点内置边缘柔化 |
| `defringe` | FLOAT | 0.07 | 0.0-1.0 | 主节点内置去红/绿/蓝边 |
| `shrink_expand` | FLOAT | 0.0 | -5.0-5.0 | 主节点内置蒙版收缩或扩展 |

#### 可选输入（连接参数节点）

| 输入 | 类型 | 说明 |
|-----|------|------|
| `sampler_args` | KEY_SAMPLER_ARGS | 采样器参数 |
| `edge_args` | KEY_EDGE_ARGS | 边缘处理参数 |
| `spill_algo_args` | KEY_SPILL_ALGO_ARGS | 溢色算法参数 |
| `ph_args` | KEY_PH_ARGS | 高光保护参数 |
| `mm_args` | KEY_MM_ARGS | 蒙版数学参数 |

#### 输出

| 输出 | 类型 | 说明 |
|-----|------|------|
| `image` | IMAGE | 合成后的图像（根据 background_mode 自动选择格式） |
| `mask` | MASK | Alpha 蒙版（单通道） |
| `mask_image` | IMAGE | 蒙版的 RGB 图像表示（用于预览） |
| `image_rgba` | IMAGE | 带 Alpha 通道的 RGBA 图像 |

#### 背景模式说明

- **alpha**: 输出带透明通道的 RGBA 图像
- **color**: 用纯色背景替换键颜色区域
- **soft_color**: 使用柔化边缘的纯色背景

---

### 2. Key Sampler Args (采样器参数)

**显示名称**: `Key Sampler Args (V2.3.6fixE3_clean_final)`

配置自动键颜色检测的采样方式。当 Core Hub 的 `key_mode` 设为 `auto` 时生效。

#### 参数

| 参数 | 类型 | 默认值 | 范围 | 说明 |
|-----|------|-------|------|------|
| `mode` | 选择 | auto_border | manual/auto_border | 采样模式 |
| `auto_border_frac` | FLOAT | 0.08 | 0.0-0.45 | 边框采样比例（auto_border 模式） |
| `rect_x` | FLOAT | 0.45 | 0.0-1.0 | 采样矩形中心 X 坐标（归一化） |
| `rect_y` | FLOAT | 0.45 | 0.0-1.0 | 采样矩形中心 Y 坐标（归一化） |
| `rect_w` | FLOAT | 0.1 | 0.0-1.0 | 采样矩形宽度（归一化） |
| `rect_h` | FLOAT | 0.1 | 0.0-1.0 | 采样矩形高度（归一化） |

#### 采样模式说明

- **auto_border**: 从图像边缘自动采样键颜色（适合绿幕填满整个背景的情况）
- **manual**: 使用指定矩形区域采样键颜色（适合需要精确指定采样区域的情况）

---

### 3. Key Edge Args (边缘参数)

**显示名称**: `Key Edge Args (V2.3.6fixE3_clean_final)`

控制蒙版的边缘处理，包括收缩/扩展、柔化和去边缘色。

#### 参数

| 参数 | 类型 | 默认值 | 范围 | 说明 |
|-----|------|-------|------|------|
| `shrink_expand` | FLOAT | 0.0 | -5.0-5.0 | 蒙版收缩（负值）或扩展（正值） |
| `edge_soft` | FLOAT | 0.08 | 0.0-1.0 | 边缘柔化程度 |
| `defringe` | FLOAT | 0.07 | 0.0-1.0 | 去边缘色强度，减少边缘色晕 |

#### 使用技巧

- **shrink_expand**: 正值扩大前景区域，负值缩小。用于修复边缘细节丢失或移除边缘噪点
- **edge_soft**: 增加此值可获得更柔和的边缘过渡
- **defringe**: 用于去除边缘残留的键颜色（绿边/蓝边）

---

### 4. Key Spill/Algo Args (溢色/算法参数)

**显示名称**: `Key Spill/Algo Args (V2.3.6fixE3_clean_final)`

配置溢色抑制算法，处理绿幕反射到主体上的颜色污染。

#### 参数

| 参数 | 类型 | 默认值 | 范围 | 说明 |
|-----|------|-------|------|------|
| `despill_mode` | 选择 | hybrid | hybrid/screen/geometric | 去溢色算法模式 |
| `algo` | 选择 | blend | blend/diff/proj | 算法类型（兼容旧版） |
| `diff_gain` | FLOAT | 1.0 | 0.0-5.0 | 差异增益 |
| `diff_balance` | FLOAT | 1.0 | 0.0-3.0 | 差异平衡 |
| `despill` | FLOAT | 2.0 | 0.0-8.0 | 去溢色强度 |
| `extra_lowalpha` | FLOAT | 1.5 | 0.0-8.0 | 低透明度区域额外去溢色 |
| `final_despill_strength` | FLOAT | 0.6 | 0.0-1.5 | 最终去溢色混合强度 |

#### 去溢色模式说明

- **hybrid**: 混合模式，综合多种算法效果（推荐）
- **screen**: 屏幕模式，基于颜色通道差异
- **geometric**: 几何模式，基于投影距离计算

#### 使用技巧

- 头发、皮肤等容易反射绿光的区域需要较强的去溢色处理
- `despill` 值过高可能导致颜色失真，需配合 `final_despill_strength` 调节
- `extra_lowalpha` 用于处理半透明区域（如发丝）的溢色

---

### 5. Key Protect Highlights Args (高光保护参数)

**显示名称**: `Key Protect Highlights Args (V2.3.6fixE3_clean_final)`

保护高光区域，防止被过度处理导致高光丢失或变色。

#### 参数

| 参数 | 类型 | 默认值 | 范围 | 说明 |
|-----|------|-------|------|------|
| `thr` | FLOAT | 0.75 | 0.0-1.0 | 高光阈值，亮度高于此值视为高光 |
| `strength` | FLOAT | 0.7 | 0.0-1.0 | 保护强度 |
| `soft_width` | FLOAT | 0.2 | 0.0-1.0 | 过渡柔化宽度 |
| `gamma` | FLOAT | 1.1 | 0.1-3.0 | 高光蒙版的伽马校正 |

#### 使用场景

- 主体上有高光反射（如金属、玻璃、湿润皮肤）
- 白色或浅色服装容易被误判为溢色的情况
- 需要保留光泽感的场景

---

### 6. Key Matte Math Args (蒙版数学参数)

**显示名称**: `Key Matte Math Args (V2.3.6fixE3_clean_final)`

对生成的蒙版进行后处理数学运算，用于精细调整最终蒙版。

#### 参数

| 参数 | 类型 | 默认值 | 范围 | 说明 |
|-----|------|-------|------|------|
| `extra_shrink_expand` | FLOAT | 0.0 | -3.0-3.0 | 额外的蒙版收缩/扩展 |
| `feather` | FLOAT | 0.15 | 0.0-2.0 | 羽化程度 |
| `gamma` | FLOAT | 1.0 | 0.2-4.0 | 蒙版伽马校正 |

#### 使用技巧

- **extra_shrink_expand**: 在 Edge Args 之后的二次调整
- **feather**: 增加蒙版边缘的柔和度，值越大边缘越模糊
- **gamma**: 小于 1 会扩大半透明区域，大于 1 会收缩半透明区域

---

## 典型工作流

### 基础用法

```
[Load Image] → [Keylight Core Hub] → [Preview/Save Image]
```

只需连接核心节点，使用默认参数即可完成基本抠像。

### 高级用法

```
[Load Image] ─────────────────────┐
                                  ↓
[Key Sampler Args] ──────→ [Keylight Core Hub] → [image]
[Key Edge Args] ─────────→        ↑                [mask]
[Key Spill/Algo Args] ───→        │                [mask_image]
[Key Protect Highlights Args] ────┘                [image_rgba]
[Key Matte Math Args] ────→
```

连接所需的参数节点进行精细调整。未连接 `Key Edge Args` 时，主节点自己的 `edge_soft`、`defringe` 和 `shrink_expand` 会直接生效；连接后参数节点会覆盖主节点值。

---

## 参数调节顺序建议

1. **首先**：调整 `tolerance`、`clip_black`、`clip_white` 获得基本的抠像效果
2. **其次**：使用 Edge Args 处理边缘问题
3. **然后**：使用 Spill/Algo Args 处理溢色
4. **接着**：如有高光问题，启用 Protect Highlights Args
5. **最后**：使用 Matte Math Args 进行最终微调

---

## 更新日志

### V3.0.0
- 新增 `guided` 模式：以上游 RGB 键控色为锚点，逐帧稳健校正视频生成造成的色偏
- 新增 `shadow_recovery`：清理暗红、暗绿、暗蓝背景，同时保护纯黑主体
- 将常用边缘参数内置到核心节点，无需额外连接 `Key Edge Args`
- 修复 `screen/hybrid` 去溢色只排除单个空间位置的问题
- 保留 `KeylightCoreHubV3` 节点标识和所有旧参数节点，兼容原工作流

### V2.3.6fixE3_clean_final
- 解决 Windows 路径式模块名导致的 `No module named ...nodes.core_hub` 问题
- 使用内部动态加载器，避免相对/绝对包名冲突
- 算法与接口同 V2.3.5

---

## 故障排除

**问题**：节点无法识别/不显示
- 检查 ComfyUI 启动控制台是否有报错
- 更新 ComfyUI 到最新版本
- 浏览器硬刷新/清除缓存
- 本节点使用前端 `COLORCODE` 颜色控件，需要正确加载 `web/` 扩展

**问题**：颜色选择器不工作
- 确保 `web/colorWidget.js` 文件存在
- 重启 ComfyUI 并刷新浏览器
