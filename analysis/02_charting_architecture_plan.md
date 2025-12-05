# 02 图表与练习架构规划（Lightweight Charts + 本地私有化）

本文件记录本项目未来 K 线可视化与“复盘练习/对照答案”相关的技术架构规划，作为后续实现的蓝图。

## 总体目标

- 完全本地、私有化：不依赖外部云服务或 CDN，在离线环境也能运行。
- 轻量化：一条命令启动后端 + 打开本地页面即可使用。
- 功能不降级：接近专业交易终端的 K 线体验（缩放、拖动、十字光标、分屏、副图等）。
- 超强标注能力：支持叠加多层标注（Paul 实盘、用户模拟交易、策略信号、教学高亮区域等）。
- 高可复用性：K 线模块与具体业务场景（练习/回测/教学）解耦，后续增加新场景时不用重写图表。

## 分层架构概览

从下到上分为四层：

1. **数据层（后端）**  
   - 已有：DuckDB 中的 1m K 线（`kline_1m`）与原始成交记录（`bitmex_trades.csv`）。  
   - 封装：  
     - `KlineData`：按 `symbol + interval + time range` 查询 K 线。  
     - 计划增加 `TradesData`：按 `symbol + time range` 查询 Paul 的真实成交记录。  

2. **场景层 / 练习引擎（后端）**  
   - 负责定义各种“练习/回放/教学”场景，例如：  
     - 随机抽取 `symbol + 时间段`，用于“盲练复盘”；  
     - 定义练习 session 的生命周期（开始时间、结束时间、进度、用户模拟交易记录）。  
   - 对外提供 REST API：  
     - `POST /api/practice/session`：创建一个练习 session，返回 `sessionId, symbol, interval, startTime, endTime`。  
     - `GET /api/practice/bars?sessionId=&from=&to=`：按时间窗口拿回放用 K 线。  
     - `GET /api/practice/paul_trades?sessionId=`：在“揭晓答案”时返回 Paul 的真实成交（作为标注层）。  

3. **图形与标注 Schema（后端 + 前端共享）**  
   - 与具体图表库无关的一套标准化描述，用 JSON 表达：  
     - 主图与副图的 K 线/线图配置（`KlineSeriesConfig`, `LineSeriesConfig` 等）；  
     - 指标配置（如 EMA、BOLL 等 overlay 的声明）；  
     - 标注（`Marker`）：  
       - `{ time, price, shape, color, text, layer }`，例如：  
         - Paul 的实盘交易：`layer='paul_trades'`；  
         - 用户模拟交易：`layer='user_trades'`；  
         - 教学提示/信号：`layer='signals'`。  
   - 后端负责生成这些 schema 和数据，前端通过适配层渲染到具体图表库上。

4. **渲染与 UI 层（前端，本地静态文件）**  
   - 图表库选型：**TradingView Lightweight Charts**（开源 MIT 协议，适合本地私有化使用）。  
   - 模块划分：  
     - `chart_core.js`（Chart Core）：对 Lightweight Charts 的统一封装，提供：  
       - `createChart(container, options) -> chartHandle`  
       - `destroyChart(chartHandle)`  
       - `addKlineSeries`, `addVolumeSeries`, `addLineSeries` 等接口；  
       - `setMarkers(layerId, markers)`：管理不同标注层；  
       - 若干交互事件回调（点击、移动、视野变化）。  
     - 场景 UI 模块（例如 `practice_ui.js`）：  
       - 面向具体业务（复盘练习、对照答案），调用 REST API，驱动 Chart Core。  
       - 不直接依赖 Lightweight Charts，只通过 Chart Core 操作。

## 后端技术栈与职责

- **语言/运行环境**：Python 3.x（沿用现有环境）。  
- **Web 框架**：FastAPI（推荐）  
  - 轻量、类型友好、易于扩展 WebSocket。  
- **核心职责**：  
  1. 暴露通用数据 API（可被任意前端复用）：  
     - `/api/kline`：按 `symbol + interval + [from, to]` 返回 K 线数据。  
     - `/api/trades`：按 `symbol + [from, to]` 返回 Paul 的成交记录。  
  2. 场景/练习 API：  
     - `/api/practice/session` / `/api/practice/bars` / `/api/practice/paul_trades` 等。  
  3. 输出统一的 Chart Schema（JSON），供前端渲染适配层消费。  
- **存储**：DuckDB 作为本地事实数据源，1m 为底层时间粒度，高周期在查询时聚合。

## 前端技术栈与组织形式

- **图表库**：TradingView Lightweight Charts  
  - 通过 npm 在本地安装/构建，将打包后的 JS 文件放入 `web/` 目录，避免外部 CDN。  
- **框架**：前期保持“轻量”，使用原生 JS 模块 + 极简组织；未来若有复杂 UI，可逐步引入 React/Svelte 等。  
- **目录结构（规划示例）**：  
  - `web/index.html`：入口页面，包含一个用于承载 K 线的容器 `<div id="chart">` 以及简单的控制按钮。  
  - `web/chart_core.js`：封装 Lightweight Charts 的核心模块。  
  - `web/practice_ui.js`：复盘练习场景的前端逻辑（调用后端 API，驱动 Chart Core）。  
  - 后续可以在 `web/` 下增加更多场景模块（例如 `compare_ui.js`, `study_ui.js` 等），共用同一 Chart Core。

## 典型场景：复盘练习 + 对照 Paul 实盘

以“随机品种 + 随机时间段练习，然后对照 Paul 的真实交易”作为例子说明各层如何协作：

1. **后端（练习引擎）**：  
   - `POST /api/practice/session`：  
     - 在所有品种/时间段中随机/按规则选取某个 `symbol + interval + [start, end]`，返回 `sessionId` 和基本信息。  
   - `GET /api/practice/bars?sessionId=&from=&to=`：  
     - 调用 `KlineData.get_klines(...)` 获取该会话中的 K 线数据（例如 1m/5m/15m）。  
   - `GET /api/practice/paul_trades?sessionId=`：  
     - 从 Paul 的成交记录中抽取该时间段的真实交易，转换为标准 Marker Schema（`layer='paul_trades'`）。  

2. **前端（复盘练习 UI）**：  
   - 初始化：  
     - 调用 `/api/practice/session` 获得一个 session。  
     - 使用 Chart Core 实例化 K 线图（主图 K 线 + 成交量，先不显示 Paul 的标注）。  
   - 练习过程：  
     - 按时间逐步请求 `/api/practice/bars` 并调用 Chart Core 的 `update` 接口播放；  
     - 用户在图上通过交互（点击/快捷键）记录模拟交易，添加到 `layer='user_trades'`。  
   - 揭晓答案：  
     - 调用 `/api/practice/paul_trades`，获取 Paul 的真实交易；  
     - 调用 `setMarkers('paul_trades', markers)` 在图上叠加 Paul 的买卖点，并可额外展示 Paul 的权益曲线。

## 本地化与打包策略

- 运行形态（开发阶段）：  
  - 一条命令启动后端（FastAPI + 静态文件）：`python -m tv_backend.main` 或 `uvicorn tv_backend.main:app`。  
  - 浏览器打开 `http://localhost:8000` 即可访问主界面。  
- 完全本地化：  
  - 所有 JS/CSS/HTML 和 DuckDB 数据文件都存储在本地仓库中，无需外部网络。  
  - 若需要进一步“免安装”，可以考虑使用 PyInstaller 或类似工具将后端 + 依赖打包为一个可执行文件，配合本地 web 静态文件。

## 下一步实施建议

1. 在后端新增一个最小可用的 FastAPI 应用：  
   - 提供 `/api/kline`（基于 `KlineData`）、`/api/trades` 两个基础接口。  
   - 挂载 `web/` 目录为静态文件根路径，验证前后端基本连通性。  
2. 定义初版 Chart Schema（仅包含主图 K 线 + 成交量 + markers 三类元素）并文档化。  
3. 在 `web/` 下实现首版 `chart_core.js` + 简单 `index.html`：  
   - 与 `/api/kline` 对接，成功渲染第一张可交互的 K 线图。  
4. 在此基础上逐步扩展：  
   - 增加复盘练习相关 API 与 `practice_ui.js`，实现“练习 + 对照答案”的最简闭环。

