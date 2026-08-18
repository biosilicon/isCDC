# Cell type 空间可视化实施与完成记录

## 状态

本功能已于 2026-08-18 完成实现、全量注释和最终审计。本文由原始实施计划更新为当前架构与
验收记录；科学迭代的详细数据见
[`assets/cell_type_annotation/iteration_history.yaml`](assets/cell_type_annotation/iteration_history.yaml)，
可复用经验见
[`annotation/细胞类型注释经验总结.md`](annotation/细胞类型注释经验总结.md)。

最终结果：

- 35/35 个二维 `full` Database 均有当前成功 sidecar；
- 3 个使用经来源核验的正式 `mdata.obs["cell_type"]`；
- 1 个 cell-resolution Xenium 使用 SingleR；
- 31 个 bin/spot Database 使用 RCTD `full` mode；
- 最终全量审计为 35 success、0 scientific failure、0 framework failure；
- 正式 `.h5mu`、catalogue schema 和公共 `/api/databases*` 字段均未改变。

## 最终架构

```text
只读 .h5mu + 固定版本参考数据
                │
                ▼
独立 iscdc-cell-annotation 环境
Python 稀疏 I/O + R/SingleR/RCTD
                │
                ▼
data/cell_type_visualizations/
references + work + immutable generations + failures + status.json
                │
                ▼
FastAPI 启动时严格校验并建立轻量只读索引
                │
                ▼
内部压缩点位端点 + deck.gl/WebGL2 可视化
```

网站运行时继续使用 `iscdc` 环境，不导入注释模块或 R 依赖。参考下载、注释、校准、产物生成
和审计只允许通过 `iscdc-cell-annotation` 环境执行。

该流程不需要 GPU、CUDA、PyTorch、scVI、CellTypist 或 cell2location。SingleR 与 RCTD 均为
CPU 任务；Python 和 R 通过受控的稀疏 Matrix Market/TSV 文件及 `Rscript` 子进程交换数据。

## 注释方法

### 来源标签

正式 `mdata.obs["cell_type"]` 已通过来源核验且完整覆盖时，直接生成 source annotation：

- 保留来源语义和拼写；
- 不重新推断；
- 不制造数值 confidence；
- 来源标签可选映射 CL ID，但映射不阻断发布。

当前共有 3 个此类 Database，来源证据见
[`cell_type来源核验记录.md`](cell_type来源核验记录.md)。

### Cell-resolution

Xenium 使用 SingleR：

- 保存全部候选类型分数、最佳/次佳分数、delta、原始标签和 `pruned.labels`；
- 使用独立 donor holdout 拟合并评估可靠性；
- RCC reference 限制为目标 panel 共享基因，并按 Xenium 稀疏 counts 深度匹配 holdout；
- `Uncertain` 不赋予 CL ID；
- protein 只参与独立 QC，不覆盖 RNA 标签。

### Bin/spot-resolution

其余 31 个目标使用 RCTD `rctd_mode="full"`：

- 保存完整 cell-type weights；
- 保存 top-2、margin、normalized entropy、effective types 和收敛诊断；
- 通过独立 pseudo-bin/pseudo-spot 校准 dominant-type reliability；
- `Mixed` 表示可靠的复杂组成状态，`Uncertain` 表示证据不足；两者均无 CL ID；
- protein、ATAC 和 histone 只做独立 QC，不覆盖 RNA 产生的标签。

## Reference 与校准

Reference recipe 固定以下信息：

- Census release、collection ID、dataset ID 和 dataset version ID；
- license、citation、物种、组织、疾病和发育阶段；
- donor-level train/holdout 划分；
- raw counts、gene key、目标 panel checksum 和共享基因门槛；
- 标签层级裁剪规则、sampling seed 和校准设计。

RCTD calibration 使用互不重叠的 logistic-fit、isotonic-recalibration 和 final-evaluation 分区。
balanced accuracy 与 macro-F1 在 pure pseudo-spots 上计算；dominant-reliability ECE 在与实际
calibrator 用途一致的独立 pure-plus-mixed 总体上计算，pure-only ECE 另存为诊断。

SingleR RCC reference v2 使用有标签 donor holdout 的目标深度匹配，不使用 Xenium target 标签。
最终 reference 指标为 BA 0.5580、macro-F1 0.5585、ECE 0.0143。

## 产物契约

默认根目录为 `data/cell_type_visualizations/`，该目录被 Git 忽略。主要结构为：

```text
data/cell_type_visualizations/
├── references/<reference_id>/
├── work/
└── <dataset_id>/
    ├── status.json
    ├── failures/<failure_id>/report.json
    └── generations/<generation_id>/
        ├── manifest.json
        ├── report.json
        ├── inference.h5
        └── points/
            ├── <sample_key>.bin
            ├── <sample_key>.bin.gz
            └── <sample_key>.bin.br
```

每个 generation 保存：

- source SHA-256、observation count、obs 顺序摘要、sample 和二维坐标方向；
- 注释方法、完整参数、环境锁摘要；
- reference ID、版本、checksum 和校准 ID；
- QC gate、指标、警告和生成时间；
- SingleR scores 或 RCTD weights 等完整推断诊断；
- 每个 inference/point 文件的大小和 SHA-256。

二进制点位格式为 little-endian SoA：header 后依次存储 `Float32 x`、`Float32 y`、推断结果的
`Float32 confidence` 和 `Uint16 type code`。点位按 sample 独立编码，并预生成 identity、gzip
和 Brotli 三种 representation。

`status.json` 只指向最新成功 generation 或最新失败报告。新 scientific/framework failure 会
撤下旧成功结果；staging 未完成、中断或校验失败时不会发布半成品。

## 网站与前端

应用通过 `ISCDC_CELL_TYPE_VISUALIZATION_ROOT` 选择 sidecar 根目录，默认使用
`data/cell_type_visualizations/`。启动时逐项校验：

- Database 类型、dataset ID 与 source SHA；
- 二维坐标、observation 数量与 sample；
- status、manifest、report 和 inference；
- identity/gzip/Brotli 点位文件的大小与 checksum。

缺失、失败、损坏或过期项完全不渲染 cell type 区域，也不会影响页面其余部分。

内部端点为：

```text
GET|HEAD /databases/{dataset_id}/cell-type-visualization/{generation_id}/{sample_key}
```

端点不进入 OpenAPI、不记录 analytics、不接受用户文件路径，并按 `Accept-Encoding` 协商
`br`、`gzip` 或 `identity`。公共 `/api/databases*` 响应保持不变。

前端位于 `frontend/`，固定 Node 24 LTS、esbuild 和按需导入的 deck.gl 模块。功能包括：

- 正交等比例视图、pan/zoom/reset；
- sample 切换、lazy load 和竞态请求取消；
- hover、图例计数和类别筛选；
- binary attributes 与 GPU buffer 释放；
- WebGL2/request/decode/context-loss 失败时只关闭可视化区域。

v1 只支持二维坐标；未来三维 Database 不会被静默投影到 XY。

## 离线命令

```bash
conda run -n iscdc-cell-annotation env PYTHONPATH=src \
  python -m iscdc.cell_type_annotation build-cell-type-reference REFERENCE_ID

conda run -n iscdc-cell-annotation env PYTHONPATH=src \
  python -m iscdc.cell_type_annotation generate-cell-type-visualization DATASET_ID [--force]

conda run -n iscdc-cell-annotation env PYTHONPATH=src \
  python -m iscdc.cell_type_annotation audit-cell-type-visualizations \
  [--all | DATASET_ID ...] [--jobs N]
```

`--force` 只允许重新构建对应 reference 或重新生成对应 dataset；原子 staging 和完整校验仍然
不可绕过。当前 scheduler 最多接受 20 个 job，并按配置的 per-task cores 与 40 个逻辑核总额
进行 deterministic batching。BLAS/OMP 线程在启动 R 前固定为 1，防止隐藏线程池超额占用。

实际利用率必须通过进程 `%CPU/100` 和 RSS 监测，不能根据声明 worker 数推断。本次 Xenium
即使声明 30 workers，采样峰值也仅约 6.36 个逻辑核。

## 最终验收

2026-08-18 的完成证据：

- 全量 annotation audit：35 success、0 scientific failure、0 framework failure；
- provenance audit：35/35 参数与计划精确相等，32/32 推断 reference 记录完整；
- 后端完整 pytest：206 passed；
- annotation Python tests：3 passed；
- annotation R contracts：2 passed；
- Ruff：passed；
- YAML parse：passed；
- 正式 `.h5mu` checksum 未变化。

最终结果不是对推断标签的 canonical metadata 背书。页面必须继续将推断标为
`Computationally inferred`，并展示 confidence、`Mixed` 和 `Uncertain`；来源标签与推断
sidecar 的语义边界不得混淆。
