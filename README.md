# FedRBTVis — 本地联邦数据异质性实验与可视分析工作台

FedRBTVis 是一个本地优先的联邦学习数据异质性实验工作台：控制标签噪声、
类别分布距离、客户端样本量与 LID 邻域参数，执行小规模 FedAvg 训练，并通过
原创 Web 界面回放和分析**真实训练产生**的事件与指标。所有可视化数据来自
实际训练过程，不提供任何伪造或插值数据。

## 项目简介

联邦学习研究中的客户端数据异质性（Non-IID）难以直接观察：分布距离、噪声、
样本量等抽象指标与训练过程脱节。本项目把"实验"与"观察"合到一个工作台：

- **实验核心**：参数化生成异质性实验（标签噪声、类别分布距离、客户端样本量、
  LID 邻域参数），执行 FedAvg 训练并落盘完整事件流。
- **可视分析**：React 前端回放训练事件，对比客户端训练轨迹、参数与指标，
  支持断线重连后按序号补齐，不依赖任何"补画"数据。
- **本地优先**：全部数据落在本地 `runs/` 产物树，无外部服务依赖。

## 核心能力

- 标签噪声：target / actual noise 两种注入方式。
- 类别分布距离：`categorical_emd_01`——0/1 类别代价距离，**不是**像素空间 EMD。
- 客户端样本量：sample count。
- 局部内在维度：LID，邻域参数 k。
- 实验事件流：先持久化再广播；前端断线后按序号补齐，避免状态伪造。

## 技术栈

| 层 | 技术 |
|---|---|
| 实验核心 | Python 3.11+ / PyTorch ≥2.5（自研训练核心，不依赖 Web 框架） |
| 后端服务 | FastAPI / uvicorn[standard] / pydantic |
| 前端 | React 18 / TypeScript / Vite 8 / Vitest / D3.js |
| 产物 | JSON / JSONL / CSV / PT（checkpoint） |

## 目录结构

```text
app/
  backend/fedrbtvis/       # FastAPI 应用与实验核心（main/api/training/engine/events/metrics…）
  backend/tests/           # 后端单元测试
  frontend/                # React 前端（src/features、src/api、tests）
data/cifar-10-batches-py/  # 标准 CIFAR-10 数据
docs/                      # architecture.md、local-development.md、provenance.md
evidence/legacy/           # 哈希锚定的历史观测（只读）
runs/                      # 实验产物树
scripts/                   # import_legacy.py、verify_owned_release.py
```

架构与事件 schema 详见 [docs/architecture.md](docs/architecture.md)，本地开发
指引见 [docs/local-development.md](docs/local-development.md)。

## 预设

| 预设 | 说明 |
|---|---|
| `test-fixture` | synthetic + TinyCNN，只用于自动化测试，不得作为研究结论 |
| `research-lite` | CIFAR-10 + ResNet18 的本地演示预设 |
| `historical-compatible` | 100+25 客户端结构兼容性预设，不等于精确复现 |

## 本地启动

后端（Windows PowerShell）：

```powershell
$env:PYTHONPATH = (Resolve-Path 'app/backend').Path
python -m uvicorn fedrbtvis.main:app --app-dir app/backend --host 127.0.0.1 --port 8000
```

前端：

```powershell
cd app/frontend
npm run dev
```

## 验证

后端：

```powershell
$env:PYTHONPATH = (Resolve-Path 'app/backend').Path
python -W error::RuntimeWarning -m unittest discover -s app/backend/tests -v
python -m compileall -q app/backend/fedrbtvis app/backend/tests
```

前端：

```powershell
cd app/frontend
npm test -- --run
npm run build
```

发布归属审计：

```powershell
python scripts/verify_owned_release.py
```

当前本地验收状态：后端 93 个测试、前端 36 个测试全部通过，`npm run build`
成功。

## 数据与产物

- 实验数据：本地 `data/cifar-10-batches-py/` 标准 CIFAR-10 批次。
- 实验产物：`runs/` 下按 Run/Study 组织的 JSON/JSONL/CSV/PT 产物树。
- 历史观测：`evidence/legacy/` 保存哈希锚定的 4,500 行课程阶段观测
  （180 块 × 25 个探测客户端），来源 SHA-256 见
  [evidence/legacy/manifest.json](evidence/legacy/manifest.json)。这些是
  **历史观测**，不是 2026 新实验，也不能作为新性能数字。

## 历史与来源

项目由三层来源演进而来，完整边界见 [docs/provenance.md](docs/provenance.md)：

1. 2023–2024 三人课程前身：CIFAR-10/ResNet18、标签噪声、类别分布与 LID 探索。
2. 用户个人毕设扩展：联邦学习异质性可视化毕业设计。
3. 2026 原创重构：新训练核心、运行服务、React 前端与历史证据导入
   （当前版本即此层）。

## 已知限制

- 不包含 FedCorr 标签纠正、真实设备联邦、隐私安全或生产级调度。
- `categorical_emd_01` 是 0/1 类别代价距离，不是像素空间 EMD。
- 实验基于合成/公开基准数据（CIFAR-10），结论仅用于方法演示与观察分析，
  不声称真实联邦场景的有效性。
- 本地结果、课程结果和公开状态分开描述；公开仓库：
  https://github.com/K-Gonzalez1204/FedRBTVis。

## 论文

`report.pdf` 为本地个人毕设材料，不进入公开仓库；如需公开须先完成
个人信息复核与来源边界确认。
