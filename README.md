# isCDC

isCDC 是一个轻量级、公开只读的空间多组学科研数据目录。它在导入时验证 `.h5mu`
结构及人工维护的元数据；网页目录请求只读取 `catalog.db`，不读取大型表达矩阵，访客会话
和行为事件则写入完全独立的 `analytics.db`。网页将 `full` 文件作为 Database 展示，并将
相同 `split_id` 的 `train`/`test` 文件聚合为一个 Challenge。每个 Challenge 通过
`derivation.challenge_type` 标记为同切片、同个体跨切片或跨个体。

## 目录约定

`temp/` 是不会纳入版本控制的本地暂存目录，用于保存尚未整理成目录可导入形式的
数据文件。暂存数据在导入前需要补充符合 schema 1.1 的 `metadata.yaml` 等必要内容；
完成整理后再通过导入命令写入正式的 `data/` 目录。`exp/` 则继续用于真实数据实验、
手工测试配置和实验输出。批量整理 `temp/` 数据时遵循
[原始数据处理工作流](原始数据处理规范.md)，最终产物必须符合
[数据库存储规范 1.1](数据库存储规范_v1.1.md)。

`.codex/` 用于存放本地 Codex 的 `dataset_planner` 和 `dataset_worker` 配置。该目录
不纳入版本控制，因此新工作区需在运行批量处理流程前准备相应的本地配置。

## 快速开始

项目使用 `conda iscdc` 环境。激活环境后：

完整测试依赖以下不会纳入版本控制的真实数据文件，并会在临时目录重跑一次空间划分：

```text
exp/xenium_human_rcc_ffpe_rna_protein.h5mu
exp/xenium_human_rcc_ffpe_rna_protein_vertical_split.yaml
```

请在运行测试前准备这两个文件，并为生成的 train/test 产物预留约 300 MB 临时空间。
缺失文件会使测试失败。

```bash
conda activate iscdc
make setup
make test
make lint
make run
```

网页和 API 的自动化测试统一使用 `httpx.AsyncClient` 与 `httpx.ASGITransport`，不使用
同步的 `fastapi.testclient.TestClient` 或 `starlette.testclient.TestClient`。仅导入数据文件且
没有修改源代码、schema、模板或 API 行为时，不需要额外启动 HTTP/ASGI 层测试；应检查导入
命令结果、`validation_report.json`、checksum、manifest 以及 catalogue/repository 读取结果。
只有网页或 API 行为发生变化时才增加或执行相应的网络层测试。

默认网页地址为 <http://127.0.0.1:8000>，API 文档为
<http://127.0.0.1:8000/docs>。

目录数据库默认保存在 `data/catalog.db`，正式数据文件保存在 `data/datasets/`。可通过
`ISCDC_DATABASE_PATH` 和 `ISCDC_DATA_ROOT` 修改这两个位置。

### 访客统计

网页页脚显示累计浏览器访问会话数。应用使用没有持久有效期的第一方 `iscdc_session`
Cookie 区分会话，并在独立的 `data/analytics.db` 中记录成功的页面浏览、筛选搜索、详情
查看和下载事件。JSON API、静态资源、健康检查及失败响应不会产生访客事件。已识别的
自动流量会记录并标记，但不增加公开访问次数。

事件明细包含直接连接的原始 IP、User-Agent、Referer、UTC 时间、路由、状态码和耗时；
应用不信任 `X-Forwarded-For` 等转发头。明细默认保留 30 天，不包含这些字段的每日汇总
永久保留。分析库发生故障时，目录页面、API 和下载继续服务，页脚计数显示为 unavailable。

可以使用 CLI 查看汇总或导出仍在保留期内的明细：

```bash
PYTHONPATH=src python -m iscdc.cli analytics summary --from 2026-07-01 --to 2026-07-31
PYTHONPATH=src python -m iscdc.cli analytics export --format csv --output events.csv
PYTHONPATH=src python -m iscdc.cli analytics export --format jsonl --output events.jsonl
```

导出文件包含原始访问信息，需按敏感运维数据管理。已有目标文件默认不会覆盖；确需覆盖时
显式添加 `--force`。相关环境变量如下：

- `ISCDC_ANALYTICS_ENABLED`：是否启用统计，默认 `true`。
- `ISCDC_ANALYTICS_DATABASE_PATH`：分析数据库路径，默认 `data/analytics.db`。
- `ISCDC_ANALYTICS_RETENTION_DAYS`：事件明细保留天数，默认 `30`。
- `ISCDC_ANALYTICS_COOKIE_SECURE`：是否为 Cookie 添加 Secure，当前 HTTP 测试部署默认
  `false`，HTTPS 部署应设为 `true`。

### 服务器测试部署

在当前服务器上，可以使用根目录的脚本将网页运行在后台 tmux 会话中：

```bash
./deploy_test.sh start
```

脚本默认使用 `/home1/shezixi/miniconda3/envs/iscdc/bin/python`，监听
`0.0.0.0:5000`，网页入口为 <http://10.138.46.171:5000>，日志保存在
`data/iscdc-server.log`。无需预先激活 Conda 环境。其他常用操作为：

```bash
./deploy_test.sh status
./deploy_test.sh logs
./deploy_test.sh restart
./deploy_test.sh stop
```

如环境位置或测试端口发生变化，可使用 `ISCDC_PYTHON` 和 `ISCDC_DEPLOY_PORT` 覆盖默认值；
运行 `./deploy_test.sh --help` 可查看全部选项。该方式只用于测试，不会开机自启，也不会
修改防火墙。若服务器本机检查正常但其他机器无法访问，需要管理员放行对应端口。

## 导入数据

每个待导入数据集由一个 `.h5mu` 文件和一个 `metadata.yaml` 组成。示例元数据位于
`assets/examples/`：

```bash
make import-example
```

也可以直接使用 CLI：

```bash
PYTHONPATH=src python -m iscdc.cli import-dataset dataset.h5mu metadata.yaml
```

导入会在临时目录完成复制、SHA-256 计算、结构验证和元数据一致性检查。全部成功后，
正式目录包含：

```text
dataset.h5mu
metadata.yaml
manifest.json
validation_report.json
checksum.sha256
```

重复的 `dataset_id` 会被拒绝。初版不提供覆盖、在线编辑或删除功能。

目录严格只接受 `schema_version: "1.1"`，不再兼容 1.0。`full` 可直接导入；导入
`train` 或 `test` 前，必须先导入其 `derivation.source_dataset_ids` 引用的全部 `full`
数据集。导入器会对照来源文件验证每个来源观测对象和特征合并策略；若目录中已经存在
相同 `split_id` 的另一侧，还会检查 train/test 的 `challenge_type` 一致且不包含重复来源
观测对象。

升级前若发现非空的旧版 SQLite 目录，应用会停止并提示备份和重新导入，不会自动删除
已有记录；空的旧版目录会自动重建为当前目录结构。
`manifest_version` 独立描述导入清单格式，与 `.h5mu` 的 `schema_version` 无关。

### 批量整理原始数据

`temp/` 下彼此独立的数据集使用两种 agent 角色和五个受控阶段处理：

1. `dataset_planner` 只读调查每个数据集，主 agent 汇总处理方案和待决定问题。
2. 用户回答后，主 agent 将全局决定传播到受影响的数据集，并提交最终计划待批准。
3. 只有获得明确批准后，`dataset_worker` 才在各数据集的隔离目录中转换和验证。
4. 主 agent 独立复核文件内容、元数据和验证报告。
5. 验收通过的数据集由主 agent 串行入库；导入后复核通过前不得删除原始数据。

角色配置默认位于 `.codex/agents/`，并受 `.codex/config.toml` 的本地并发限制。
完整的并发数、路径隔离、批准门槛、验收和清理规则见
[原始数据处理工作流](原始数据处理规范.md)。

## metadata.yaml

YAML 是网站元数据的主记录。`.h5mu` 内部已有的数据库、样本和 assay 元数据必须在
YAML 中出现且值一致；YAML 可以包含文件内部没有的附加数据库元数据。

```yaml
database:
  schema_version: "1.1"
  dataset_id: example_rna_protein
  dataset_type: full
  source: GSE000000
  organism: Homo sapiens
  tissue: kidney
  spatial_unit: cell
  coordinate_unit: micrometer
  pairing_type: same_unit
sample_ids:
  - sample_01
modalities:
  rna:
    technology: Xenium
    value_type: counts
  protein:
    technology: Xenium
    value_type: intensity
title: Example spatial RNA and protein dataset
description: A concise scientific description.
keywords:
  - spatial transcriptomics
  - spatial proteomics
license: null
publication: null
```

`license` 和 `publication` 键必须存在，但在信息无法确认时可以为 `null`。
`coordinate_unit` 可使用 `array_index` 表示平台原生的离散阵列索引，例如 10x
Visium 的 `[array_col, array_row]`。如果没有执行明确的像素或物理坐标换算，应保留
`array_index`，不应将这些值重新标记为 `pixel` 或 `micrometer`。
多全集衍生数据的 `source`、`organism`、`tissue` 以及模态 `technology` 可以使用去重后的
字符串列表；目录 API 会原样保留字符串或列表形式。

## 划分训练集和测试集

`iscdc.splitter` 是独立的 `.h5mu` 划分工具，不自动修改 SQLite，也不生成网站使用的
`metadata.yaml`。需要发布产物时，为 train/test 分别准备匹配的 metadata YAML，再使用
现有 `import-dataset` 命令逐个导入。来源文件必须符合根目录
[`数据库存储规范_v1.1.md`](数据库存储规范_v1.1.md)，并明确包含：

```yaml
schema_version: "1.1"
dataset_type: full
```

所有划分参数均写入 YAML。配置内的来源和输出相对路径以配置文件所在目录为基准。
输出目录必须尚不存在；成功时其中只包含 `<train_id>.h5mu` 和
`<test_id>.h5mu`。工具先在同级临时目录写入并重新验证两个文件，全部通过后才原子提交。

### 查看坐标范围

`range` 是只读检查命令，可先用它确定空间划分边界：

```bash
PYTHONPATH=src python -m iscdc.splitter range full.h5mu
PYTHONPATH=src python -m iscdc.splitter range full.h5mu --sample-id sample_01
PYTHONPATH=src python -m iscdc.splitter range full.h5mu --json
```

输出包含全局及各样本的 x/y 最小值、最大值、观测数、坐标单位和坐标维数。对于
三维坐标仍只统计 x/y，但会报告维数为 3。

### 按空间区域划分

`spatial` 从一个全集产生互不重叠且完整覆盖来源观测的训练集和测试集。配置示例：

```yaml
schema_version: "1.1"
split_id: spatial_v1
challenge_type: same_slice
feature_merge_policy: preserve
source: full.h5mu
output_dir: outputs/spatial_v1
train:
  dataset_id: spatial_train_v1
test:
  dataset_id: spatial_test_v1
  regions:
    - sample_id: sample_01
      x_min: 0
      x_max: 100
      y_min: 0
      y_max: 100
    - sample_id: sample_02
      x_min: 20
      x_max: 80
      y_min: 10
      y_max: 90
```

运行：

```bash
PYTHONPATH=src python -m iscdc.splitter spatial spatial.yaml
```

矩形边界闭合，多个矩形取并集。矩形内的观测进入测试集，其余全部进入训练集；未在
`regions` 中出现的样本完整进入训练集。两侧必须都非空，并且每个来源模态在两侧均须
有观测。空间划分固定使用 `feature_merge_policy: preserve`，保留来源特征顺序、矩阵值、
坐标、观测 ID 和模态成员关系。

### 按完整全集组合

`compose` 不切分全集内部观测，而是将完整来源分配给 train 或 test：

```yaml
schema_version: "1.1"
split_id: benchmark_v1
challenge_type: cross_subject
feature_merge_policy: intersection
output_dir: outputs/benchmark_v1
train:
  dataset_id: benchmark_train_v1
  sources:
    - full_a.h5mu
    - full_b.h5mu
  reference_dataset_id: null
test:
  dataset_id: benchmark_test_v1
  sources:
    - full_c.h5mu
  reference_dataset_id: null
```

运行：

```bash
PYTHONPATH=src python -m iscdc.splitter compose compose.yaml
```

同一个全集不能分配给两侧。每侧只有一个来源时记录为 `subset`，多个来源时记录为
`composite`。所有来源必须使用相同的空间单位和坐标单位，同名模态必须使用相同的
`value_type`，最终 train/test 模态集合必须一致且至少包含两个模态。

`spatial` 和 `compose` 配置都必须显式声明 `challenge_type`，可选值为：

- `same_slice`：同一物理切片内部划分。
- `cross_slice_same_subject`：同一个体的不同切片之间划分。
- `cross_subject`：跨个体划分，涵盖生物学重复及不同条件或发育阶段。

该字段会写入两侧 `uns["database"]["derivation"]`。发布产物时，对应
`metadata.yaml` 的 `database.derivation.challenge_type` 必须与文件内部取值一致。同一
`split_id` 的 train/test 必须使用相同值；缺少该字段的旧 schema 1.1 衍生文件需要补齐后
重新校验或导入。

支持以下特征策略：

- `preserve`：每个输出侧内的相关来源必须具有完全一致的特征 ID 和顺序。
- `intersection`：每个输出侧分别按本侧第一个相关来源的顺序保留本侧共同特征。
- `union`：每个输出侧分别按本侧来源顺序保留首次出现的全部特征。
- `reference`：两侧分别通过 `reference_dataset_id` 指定本侧参考全集；两个参考全集的
  模态和特征顺序必须一致。

`union` 以及存在缺失特征的 `reference` 会以零作为存储占位，并在
`varm["feature_measured_by_source"]` 保存“特征 × 来源全集”布尔掩码。模态元数据会说明
`False` 表示来源未测量该特征，而不是真实测量值为零。来源完全缺少某个模态时不会创建
伪造矩阵。

`intersection` 和 `union` 由 train/test 各自声明的来源分别计算，因此两侧特征空间可能
不同；发生差异时，两个产物的 `processing_description` 会明确记录。`reference` 仍要求
两侧参考全集使用完全相同的模态和特征顺序。

组合产物的顶层观测 ID 和样本 ID 分别编码为
`<source_dataset_id>::<source_obs_id>` 和
`<source_dataset_id>::<original_sample_id>`。每个观测的原始来源仍保存在
`source_dataset_id`、`source_obs_id` 中，产物的 `uns["database"]["derivation"]` 会记录
划分 ID、Challenge 类型、来源全集、选择规则、特征策略、处理说明和
`random_seed: null`。

## 网页和 API

- `/databases`：浏览和筛选 `full` Database 文件。
- `/databases/{dataset_id}`：Database 元数据、模态规模、校验信息和下载。
- `/challenges`：以 `split_id` 为单位浏览和筛选 Challenge。
- `/challenges/{split_id}`：同页查看 Challenge 对应的 train/test 文件；尚未配对时会明确
  标记缺失侧。
- `/api/databases`、`/api/databases/{dataset_id}`：Database JSON 列表和详情。
- `/api/challenges`、`/api/challenges/{split_id}`：按 Challenge 聚合的 JSON 列表和详情。
- `/downloads/{dataset_id}/{kind}`：下载 h5mu、metadata、manifest、validation 或 checksum。
- `/healthz`：供部署脚本和监控使用的健康检查，不创建访客会话或行为事件。

网页列表使用 `q`、`organism`、`tissue`、`modality`、`technology` 和 `spatial_unit`
筛选；Challenge 列表还支持 `challenge_type`，API 列表另外支持 `limit`、`offset`。
Challenge 的任一侧满足全部筛选条件时，响应
仍会返回该 Challenge 已导入的完整两侧。若同一个 `split_id` 下存在多份 train 或多份
test，目录会报告完整性错误，不会静默选择其中一份。

Challenge JSON 使用顶层 `challenge_type` 返回分类，使用 `status` 表示 `complete`、
`missing_train` 或 `missing_test`，并通过 `train`、`test` 字段返回对应的完整文件记录或
`null`。旧版 `/datasets`、
`/datasets/{dataset_id}` 和 `/api/datasets*` 路由已经移除，不提供兼容重定向。

生产部署时可以在下载路由前增加 Nginx；当前开发版本由 FastAPI 直接传输文件。

## 目录

```text
src/iscdc/       应用、校验、导入和数据划分代码
tests/           自动化测试（包括 splitter 合成数据与必需的真实数据测试）
assets/templates 网页模板
assets/static    页面样式
assets/examples  示例人工元数据
data/            catalog.db、analytics.db 和已导入文件（不纳入版本控制）
temp/            待整理数据及其隔离的转换产物（不纳入版本控制）
exp/             本地真实输入、手工测试配置和实验产物（不纳入版本控制）
.codex/          本地 agent 角色和并发配置（不纳入版本控制）
```
