# isCDC

isCDC 是一个轻量级、公开只读的空间多组学科研数据目录。它在导入时验证 `.h5mu`
结构及人工维护的元数据，网页请求只读取 SQLite，不读取大型表达矩阵。

## 快速开始

项目使用 `conda iscdc` 环境。激活环境后：

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

## metadata.yaml

YAML 是网站元数据的主记录。`.h5mu` 内部已有的数据库、样本和 assay 元数据必须在
YAML 中出现且值一致；YAML 可以包含文件内部没有的附加数据库元数据。

```yaml
database:
  schema_version: "1.0"
  dataset_id: example_rna_protein
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

## 网页和 API

- `/datasets`：关键词搜索以及物种、组织、模态、技术和空间单位筛选。
- `/datasets/{dataset_id}`：元数据、模态规模、校验信息和文件下载。
- `/api/datasets`：JSON 列表，支持相同筛选条件及 `limit`、`offset`。
- `/api/datasets/{dataset_id}`：单个数据集的完整 JSON 记录。

生产部署时可以在下载路由前增加 Nginx；当前开发版本由 FastAPI 直接传输文件。

## 目录

```text
src/iscdc/       应用、校验和导入代码
tests/           自动化测试
assets/templates 网页模板
assets/static    页面样式
assets/examples  示例人工元数据
data/            本地 SQLite 和已导入文件（不纳入版本控制）
```
