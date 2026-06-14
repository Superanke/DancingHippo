# AGENTS.md

> 这是给编码 agent（Codex）的项目操作手册。开工前必读，每次开始新任务前重读相关章节。

---

## 0. 一句话项目目标

把开源项目 **KupkaProd-Cinema-Pipeline** 改造成项目主人自用的本地 AI 影视制作工具「DancingHippo」：输入一句主意或一段剧本（后续还要支持一部小说），全本地生成剧本 → 分镜 → 关键帧 → 视频 → 成片。视频引擎保持 **LTX-2.3** 不变（这是当前路线）。

原始仓库：`https://github.com/Matticusnicholas/KupkaProd-Cinema-Pipeline`

---

## 1. 最高优先级：你该怎么对待项目主人（务必遵守）

项目主人**不是程序员，看不懂代码和技术术语**。这条决定你所有的沟通和工作方式：

- **用中文沟通，用大白话。** 不要丢术语和代码片段让他自己判断。要解释「我改了什么、为什么、他能看到什么变化」。
- **不要让他调试。** 出了问题你自己查。需要他做的，只能是「点这个按钮 / 看这个画面对不对 / 把报错截图发我」这种动作。
- **每完成一步，给他一份『请你手动验证』清单**，写成他能照做的简单步骤（见第 5 节）。
- **不要自作主张大改。** 不确定的设计选择，先用一句话中文问他要哪种，别猜。但能自己查清楚的（比如代码怎么写的）不要问，自己查。
- **保护好能跑的代码。** 不删、不覆盖现有可用功能，除非任务明确要求。任何可能破坏现有功能的改动，先说明再做。

---

## 2. 运行环境（假设，开工前跟主人确认一次）

- 操作系统：**Windows 11**。脚本、路径、命令用 Windows 风格。
- 机器：带 **RTX 5090（32GB 显存）**。
- **本项目、ComfyUI、Ollama 都跑在这同一台机器上**（这是刻意的设计，为了避免跨机读文件和远程连接的麻烦，详见第 4 节约束）。所以一切都按「本地 localhost」处理。
- ComfyUI 默认地址 `127.0.0.1:8188`，Ollama 默认 `localhost:11434`。
- 依赖见 `requirements.txt`：`websocket-client, ollama, opencv-python, Pillow, requests, sv_ttk`。

---

## 3. 代码地图（已逐文件读过，可信）

主体在 `video_director_agent/`：

- `agent.py` — 主编排器 + 命令行入口。核心函数 `run()`，流程分三阶段：Phase 1 拆场景、Phase 2 关键帧、Phase 3 视频；`preflight()` 会检查 ComfyUI 是否在跑，没跑就本地 `launch_comfyui()` 拉起。
- `director.py` — 用 Ollama LLM 做剧本拆解和写 prompt。关键函数：`breakdown()`（拆场景+角色描述）、`parse_script()` 和 `_is_script()`（自动识别剧本格式）、`write_prompt()`（为每个场景写自包含的视频 prompt）。**已有 JSON 修复逻辑**：`_fix_json()` / `_parse_json()`（带重试）/ `_chat_with_auto_tokens()`。
- `comfyui_client.py` — ComfyUI 的 API 客户端（WebSocket + REST）。提交工作流、等完成、取回结果。含 `build_workflow()`（只往模板里注入 prompt/负面词/帧数/seed/分辨率，其余保持模板原样）和节点自动识别 `_detect_video_nodes()`。
- `keyframe_gen.py` — 关键帧出图 + 评估。
- `evaluator.py` — 视频帧评估。
- `assembler.py` — 用 FFmpeg 把选中的片段无损拼接成片。
- `gui.py` — Tkinter 主界面 + 首次设置向导（只有三个填空：ComfyUI 根目录、启动脚本名、Ollama 模型名）。
- `storyboard.py` / `reviewer.py` — 关键帧审批界面 / 选 take 界面。
- `config.py` — 所有设置集中地。默认值在 `_DEFAULTS`，用户覆盖写在同目录 `user_settings.json`（首次运行自动生成）。
- `workflow_template.json` / `keyframe_template.json` — ComfyUI 的 API 格式工作流模板（程序实际用的）。
- `workflows/*.json` — 给人在 ComfyUI 里手动加载、验证用的工作流。
- 运行产物在 `video_director_agent/output/<项目名>/`，含 `state.json`（可断点续跑）、`keyframes/`、`scenes/`、`final.mp4`。

---

## 4. 硬约束与已知坑（改代码前必看，违反会出事）

这些是读源码确认出来的，不是猜的：

1. **取回生成文件靠本地磁盘路径，不是 API。** `comfyui_client.get_output_path()` 和 `keyframe_gen.get_image_output_path()` 都从 `COMFYUI_OUTPUT_DIR`（由 `comfyui_root` 推出的本地路径）读文件。所以全栈必须同机；**不要在没有处理文件取回的情况下，把架构改成连远程 ComfyUI**。

2. **`config.OLLAMA_HOST` 是失效的。** 它在 config.py 里有定义，但 `director.py` 调的是 `ollama.chat(model=...)`，用的是 ollama 官方库的默认地址，**没接 config 里那个值**。要连远程 Ollama 只能靠系统环境变量 `OLLAMA_HOST` 或显式 `ollama.Client(host=...)`。同机跑就不用管这条。

3. **整条流程严重依赖 LLM 吐干净 JSON。** config.py 里有注释说大模型「26B has JSON issues」，所以默认把 creative 和 fast 都指到 fast 模型。换模型（如换 Qwen）后，**务必验证 `breakdown()` 能稳定返回可解析的 JSON**。已有 `_fix_json`/重试兜底，但别依赖它兜所有锅。

4. **`build_workflow()` 只注入 5 样东西**：prompt、负面词、帧数、seed、分辨率。其余（采样器、sigmas、CFG、模型、LoRA）全部沿用模板。**换图像/视频模型 = 换 ComfyUI 工作流并重新导出 API 模板**，不是改 Python 参数。节点 ID 会自动识别，但换完要验证识别对了。

5. **默认负面提示词屏蔽了背景音乐。** `config.NEGATIVE_PROMPT` 里含 `background music, soundtrack, musical score`。要配乐就得动这里。

6. **LLM 的系统提示词是英文的**（在 director.py 里）。输出语言基本跟随输入，但可能飘回英文。中文化任务要处理这点。

7. **Windows 专属假设**：`setup.bat`、`start.bat`、`.bat` 启动器、反斜杠路径、在 comfyui_root 里找 `ffmpeg.exe`。别改成只在 Linux 能跑。

8. **默认是人工审批模式**：`LAZY_MODE=False`（人工挑关键帧和 take）、`SKIP_KF_EVAL=True`。主人要的就是这种可控，别擅自改成全自动。

---

## 5. 工作方式（流程纪律）

1. **先做基线，再改东西。** 任何修改前，先让未改动的原版在这台机器上跑通一条 1 分钟测试片（见 TASKS.md 阶段 0）。基线没跑通就改代码 = 给自己挖坑。
2. **用 Git 保命。** 一开始就 `git init`（或 fork）。每完成一个能跑通的小步就 commit，写清楚改了啥。这样随时能回滚——对看不懂代码的主人尤其重要。
3. **小步迭代，一次只动一件事。** 按 TASKS.md 的阶段顺序来，别几件事混在一起改。
4. **改前先读。** 动任何文件前先把它读完，遵守第 4 节约束。
5. **验证你能验证的。** 你能跑的：装依赖、import/语法检查、单点小逻辑测试、用假数据 dry-run。**你跑不动完整生成**（要 GPU + 模型 + ComfyUI 全开），所以完整出片这步，写成给主人的手动验证清单：
   - 例：「我把模型换成了 Qwen。请你打开程序，输入『做一个一分钟讲深秋庭院的短片』，点 Start Production。等它拆完场景后，看弹出来的场景描述**是不是中文、读起来通不通顺**，截图发我。」
6. **报错就自己查**，查不动再带着你的排查过程问主人要具体信息。

### 5.1 真实用户体验验收标准（务必遵守）

以后每次修改 WebUI、生成流程、项目状态、按钮、日志、重试、刷新恢复、失败处理，都必须按真实用户视角验收。**后台成功不等于完成；用户在页面上看得懂发生了什么，才算完成。**

- **改完必须自己实际走一遍产品流程。** 至少打开 WebUI，创建或进入项目，点击相关按钮，观察页面状态、日志、按钮可用性、结果是否真实出现。
- **任何超过 1 秒的动作都必须有可见反馈。** 按钮要进入 running/disabled 状态，页面要显示当前步骤，日志要持续更新，不能让用户以为程序卡死。
- **失败必须显示清楚原因和下一步。** 不允许只显示 `fail`、`AI FAIL`、`Retry Production` 这类用户无法判断的问题；要说明失败发生在哪一步、可能原因、用户现在能做什么。
- **长流程必须一步一步展示。** Script planning、storyboard/keyframe、take generation、review、assembly、final video 等阶段，都要有明确状态和完成/失败标记。
- **按钮行为必须收尾。** 例如 regenerate/start/assemble：点击后要变成处理中；完成后要消失、变灰或变成下一步；失败后要允许明确重试并显示错误。
- **刷新页面必须恢复正确状态。** 刷新后不能跳到错误项目，不能丢日志，不能把别的项目的 job/error 显示到当前项目。
- **项目切换必须隔离。** 当前项目只显示自己的 brief、storyboard、takes、final、logs、error；不要串到最新项目或后台其他 job。
- **完成项目必须锁住危险操作。** 已完成的项目不能默认显示会破坏结果的 regenerate/start/assemble 操作；这些操作要放到更谨慎的位置。
- **每次交付前至少做一次烟测清单。** 默认包括：新建项目、Start Production、Storyboard 状态、Regenerate keyframe、Assemble/final video、刷新页面、切换项目、失败状态显示。
- **交付说明要说用户能看到的变化。** 不要只说“后端已修复”；要说明页面上会出现什么、按钮如何变化、失败时怎么显示。

---

## 6. 任务清单

见同目录 **`TASKS.md`**，按阶段从上往下做。每个阶段都有「完成标准」和「请主人验证」两栏。做完一个阶段，更新 TASKS.md 的勾选状态，再进下一个。

---

## 7. 不确定时

- 涉及**设计取舍**（要 A 还是 B）→ 用一句中文问主人。
- 涉及**代码事实**（这函数怎么写的、这值哪来的）→ 自己读代码，别问。
- 涉及**外部不确定信息**（比如 Ollama 当前可用的 Qwen 版本标签）→ 让主人去 ollama.com 查一下告诉你，或你联网确认，别硬编一个可能不存在的标签。
