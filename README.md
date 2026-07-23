# isCDC

isCDC 是一个轻量级、公开只读的空间多组学科研数据目录。它在导入时验证 `.h5mu`
结构及人工维护的元数据，网页请求只读取 SQLite，不读取大型表达矩阵。

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
make setup
make test
make lint
make run
```

默认网页地址为 <http://127.0.0.1:8000>，API 文档为
<http://127.0.0.1:8000/docs>。

数据库默认保存在 `data/catalog.db`，正式数据文件保存在 `data/datasets/`。可通过
`ISCDC_DATABASE_PATH` 和 `ISCDC_DATA_ROOT` 修改这两个位置。

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
相同 `split_id` 的另一侧，还会检查 train/test 不包含重复来源观测对象。

升级前若发现非空的旧版 SQLite 目录，应用会停止并提示备份和重新导入，不会自动删除
已有记录；空的旧版目录会自动重建为当前目录结构。
`manifest_version` 独立描述导入清单格式，与 `.h5mu` 的 `schema_version` 无关。

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
划分 ID、来源全集、选择规则、特征策略、处理说明和 `random_seed: null`。

## 网页和 API

- `/datasets`：关键词搜索以及物种、组织、模态、技术和空间单位筛选。
- `/datasets/{dataset_id}`：元数据、模态规模、校验信息和文件下载。
- `/api/datasets`：JSON 列表，支持相同筛选条件及 `limit`、`offset`。
- `/api/datasets/{dataset_id}`：单个数据集的完整 JSON 记录。

生产部署时可以在下载路由前增加 Nginx；当前开发版本由 FastAPI 直接传输文件。

## 目录

```text
src/iscdc/       应用、校验、导入和数据划分代码
tests/           自动化测试（包括 splitter 合成数据与必需的真实数据测试）
assets/templates 网页模板
assets/static    页面样式
assets/examples  示例人工元数据
data/            本地 SQLite 和已导入文件（不纳入版本控制）
exp/             本地真实输入、手工测试配置和实验产物（不纳入版本控制）
```
