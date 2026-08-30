# isCDC

isCDC 是一个面向跨组学翻译、轻量级且公开只读的空间多组学科研数据目录。它在导入时验证
`.h5mu` 结构及人工维护的元数据；网页目录请求读取 `catalog.db`、启动时建立的辅助文件索引和
经过校验的 `challenge_difficulty.json` 快照，不读取大型表达矩阵，访客会话和行为事件则写入
完全独立的 `analytics.db`。网页将 `full`
文件作为 Database 展示，并将
相同 `split_id` 的 `train`/`test` 文件聚合为一个 Challenge。每个 Challenge 通过
`derivation.challenge_type` 标记为同切片、同个体跨切片或跨个体。

面向数据访问者的下载、`.h5mu` 结构、元数据字段和分析注意事项见
[数据使用说明](doc/数据使用说明.md)。
Schema 1.2 还允许在可靠来源标签完整或经核验为部分覆盖时保存统一格式的可选
`mdata.obs["cell_type"]`；当前 catalogue 的逐数据集来源结论见
[cell type 来源核验记录](doc/cell_type来源核验记录.md)。离线推断与空间可视化的最终架构和
运行方式见下文 [Cell type 空间可视化](#cell-type-空间可视化)，35 数据集的方法学与运行经验见
[细胞类型注释经验总结](doc/annotation/细胞类型注释经验总结.md)。

## 目录约定

`temp/` 是不会纳入版本控制的本地暂存目录，用于保存尚未整理成目录可导入形式的
数据文件。暂存数据在导入前需要补充符合 schema 1.2 的 `metadata.yaml` 等必要内容；
完成整理后再通过导入命令写入正式的 `data/` 目录。`exp/` 则继续用于真实数据实验、
手工测试配置和实验输出。批量整理 `temp/` 数据时遵循
[原始数据处理工作流](doc/原始数据处理规范.md)，最终产物必须符合
[数据库存储规范 1.2](doc/数据库存储规范_v1.2.md)。

`.codex/` 用于存放本地 Codex 的 `dataset_planner` 和 `dataset_worker` 配置。该目录
不纳入版本控制，因此新工作区需在运行批量处理流程前准备相应的本地配置。

Database 的网页缩略图以 `<dataset_id>.webp` 命名，保存在被忽略的
`assets/static/database_thumbnails/`，不纳入版本控制；新工作区和部署环境需要单独准备这些本地
文件。仅用于制作缩略图的 PNG、JPEG 或 TIFF 来源文件保存在被忽略的
`assets/he_wsi_thumbnails/`；正式提供下载的 WSI 则注册到所属
数据集的 `data/datasets/<dataset_id>/auxiliary/`。已注册 `he_wsi` 的 Database 应直接由该 WSI
生成缩略图，不再使用另一张预览图代替。并非每个 Database 都有缩略图或 WSI；缺图
时页面不显示占位图或空图片区域。具体图像来源与当前可下载状态以 Database 详情页的辅助
文件清单、source URL 和 SHA-256 为准。

## 快速开始

项目使用 `conda iscdc` 环境。激活环境后：

日常开发和 agent 修改默认只运行与本次改动直接相关的测试，优先指定精确 pytest node ID 或
最窄的相关测试文件，不自动运行完整套件。例如：

```bash
conda activate iscdc
make setup
PYTHONPATH=src python -m pytest tests/test_app.py::test_multisource_challenge_shows_every_sample_source_in_page_and_api
make lint
make run
```

只有用户明确要求完整测试时才运行 `make test`。完整测试依赖以下不会纳入版本控制的真实数据
文件，并会在临时目录重跑一次空间划分：

```text
exp/xenium_human_rcc_ffpe_rna_protein.h5mu
exp/xenium_human_rcc_ffpe_rna_protein_vertical_split.yaml
```

请在运行完整测试前准备这两个文件，并为生成的 train/test 产物预留约 300 MB 临时空间。
缺失文件会使测试失败。

显式完整验收使用 `make test`。

完整测试包含使用多进程 `DataLoader` worker 的 PyTorch 用例，因此运行环境必须允许本地
IPC socket。在受限沙箱中，运行 `make test` 前应申请仅覆盖本机进程间通信的最小权限；
该要求不需要也不授权外部网络访问。若本地 socket 被禁止，worker 可能在
`multiprocessing.resource_sharer` 中报出 `PermissionError`，而 pytest 主进程继续等待。

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
默认 `data/` 可能位于网络文件系统，因此 `analytics.db` 使用 DELETE journal；不要改用依赖
共享内存协调的 WAL 模式。应用启动时会将旧版创建的 WAL 数据库安全切换回 DELETE 模式。

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

如环境位置或测试端口发生变化，可使用 `ISCDC_PYTHON` 和 `ISCDC_DEPLOY_PORT` 覆盖默认值。
脚本默认等待应用就绪 60 秒；可通过 `ISCDC_DEPLOY_START_TIMEOUT` 设置 1 至 600 秒的等待
时间。运行 `./deploy_test.sh --help` 可查看全部选项。该方式只用于测试，不会开机自启，
也不会修改防火墙。若服务器本机检查正常但其他机器无法访问，需要管理员放行对应端口。
应用在启动时扫描 Database 缩略图和辅助文件 manifest、校验 Challenge difficulty 快照，并计算
`styles.css` 的内容版本；新增或删除缩略图、注册或移除辅助文件、重新生成 difficulty 快照或
修改页面样式后，应执行 `./deploy_test.sh restart`。
样式表 URL 会携带内容哈希，重启后浏览器会自动获取新版本，无需用户手动清除缓存。

### Cell type 空间可视化

Database 详情页可以读取独立的 cell type 可视化 sidecar。正式 `.h5mu`、catalogue schema、
公共 `/api/databases*` 响应和筛选字段均不因此改变。应用默认在
`data/cell_type_visualizations/` 查找产物，也可设置
`ISCDC_CELL_TYPE_VISUALIZATION_ROOT`。启动时会校验最新 `status.json`、源文件 SHA-256、
二维坐标声明、manifest 与各压缩点位文件；缺失、失败、过期或损坏的项目不会在页面产生
占位区。替换产物后必须重启应用。

可视化标题旁的 `?` 按钮用于查看当前数据集的注释方法。来源标签 sidecar 只显示具体方法，
并明确说明标签来自既有注释文件、没有执行计算推断，不显示 reference、运行参数、阈值或推断
QC。计算推断 sidecar 则从启动时已校验的 manifest/report 展示方法、reference ID 与版本、
运行参数、QC 发布阈值和实际 QC 结果；未配置阈值显示为 `Not configured`。该说明弹窗不重复
展示逐点 confidence，confidence 仍只在现有点位 hover 中呈现，公共 Database JSON 保持不变。
若 sidecar 包含项目保留类别 `Unannotated`，图例仍提供该复选框，但首次加载时默认不勾选，
对应点位以零半径隐藏；用户可单独勾选或通过 `Select all` 恢复显示。该规则只匹配精确标签
`Unannotated`，不会隐藏来源自身定义的 `Unlabeled` 等类别。

2026-08-18 批次的全量结果为 35/35 Database success：3 个使用来源标签，1 个 Xenium
使用 SingleR，31 个 bin/spot 使用 RCTD `full`。这是当时的 sidecar 快照；后续新增 Database
可以暂时没有 sidecar，缺失时页面按上述规则不显示占位区。任何推断标签都不会提升为 canonical
dataset metadata。完整 provenance/QC 结果见
[`iteration_history.yaml`](assets/cell_type_annotation/iteration_history.yaml)，方法学与失败修复经验见
[`doc/annotation/细胞类型注释经验总结.md`](doc/annotation/细胞类型注释经验总结.md)。

2026-08-28 新增的 `xenium_human_ccrcc_ffpe_rna_protein` 已另行发布来源型 sidecar：690,322
个点位、19 个展示类别，其中 331,237 个 `Unannotated` 按上述规则默认隐藏，且没有执行计算
推断。该增量发布不改写 2026-08-18 的 35/35 历史批次统计。

参考构建、SingleR、RCTD、校准、生成和审计必须在隔离环境中运行，不能向网站使用的
`iscdc` 环境安装 R 或注释依赖：

```bash
conda env create -f annotation/environment.yml
conda run -n iscdc-cell-annotation Rscript -e \
  'options(timeout=600); install.packages("https://cloud.r-project.org/src/contrib/renv_1.2.4.tar.gz", repos=NULL, type="source")'
conda run -n iscdc-cell-annotation Rscript -e \
  'renv::restore(lockfile="annotation/renv.lock", library=.libPaths()[1], prompt=FALSE)'

conda run -n iscdc-cell-annotation env PYTHONPATH=src \
  python -m iscdc.cell_type_annotation build-cell-type-reference REFERENCE_ID
conda run -n iscdc-cell-annotation env PYTHONPATH=src \
  python -m iscdc.cell_type_annotation generate-cell-type-visualization DATASET_ID [--force]
conda run -n iscdc-cell-annotation env PYTHONPATH=src \
  python -m iscdc.cell_type_annotation audit-cell-type-visualizations --all --jobs 20
```

来源已有可靠标签的 Database 直接保留 `mdata.obs["cell_type"]` 的来源拼写及经验证的
`Unannotated` 状态；Xenium 推断使用
SingleR，bin/spot 使用 RCTD `full` mode。推断生物类型映射到稳定 CL ID，`Mixed` 和
`Uncertain` 仅作为无 CL ID 的预测状态。完整分数或 weights、校准结果和 QC 报告只保存于
sidecar generation。科学质量门槛失败会发布完整失败状态并立即撤下旧成功结果。

浏览器端源码与锁文件位于 `frontend/`，生产 bundle 会提交到 `assets/static/`。构建机固定
使用 Node 24 LTS；Node 不属于 `iscdc` Conda 环境：

```bash
cd frontend
npm ci
npm test
npm run build
```

## 使用 PyTorch 训练

PyTorch 接口是可选功能，不会增加网页服务的基础依赖。需要训练时安装：

```bash
conda activate iscdc
python -m pip install -r requirements-pytorch.txt
```

`H5MuDataset` 以顶层 observation 为样本，按需从 `.h5mu` 读取各模态的 `X`，不会把完整
表达矩阵载入内存。默认样本包含模态张量、模态存在性 mask、特征测量 mask、空间坐标和
观测 ID。两模态 schema 1.2 文件始终完全配对；三个及以上模态的 `partially_shared`
文件仍以零向量表示缺失模态，训练时必须同时使用 mask，不能将占位零解释为真实测量值。
`cell_type` 不是每个文件都有；只有确认列存在后才能将它加入 `obs_columns`。

```python
import torch
from torch.utils.data import DataLoader

from iscdc.pytorch import H5MuDataset


class LogNormalizeRNA:
    def __call__(self, sample):
        sample["modalities"]["rna"] = torch.log1p(sample["modalities"]["rna"])
        return sample


dataset = H5MuDataset(
    "dataset.h5mu",
    modalities=("rna", "protein"),
    obs_columns=("cell_type",),  # 仅在文件包含该列时使用
    transform=LogNormalizeRNA(),
)
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

for batch in loader:
    rna = batch["modalities"]["rna"]
    protein = batch["modalities"]["protein"]
    protein_present = batch["modality_masks"]["protein"]
    protein_measured = batch["feature_masks"]["protein"]
```

使用 `num_workers > 0` 时，运行环境必须允许 PyTorch worker 通过本地 IPC socket 传递
队列和张量；受限沙箱或容器应显式允许本机进程间通信，无需开放外部网络。transform 应是
可 pickle 的顶层函数或类。Dataset 会为每个 worker 惰性打开独立的只读 HDF5 句柄；直接
使用 Dataset 时可调用 `close()`，也可用上下文管理器。若需要改变默认样本结构，可继承并
重写 `build_sample()`；该方法同样用于批量读取，不会被 `DataLoader` 的批量索引路径绕过。

对于每个 observation 同时具有输入和目标模态、且两侧所有特征均真实测量的跨模态监督
任务，可以使用返回纯 `(x, y)` 的包装器：

```python
from iscdc.pytorch import H5MuPredictionDataset

dataset = H5MuPredictionDataset(
    "dataset.h5mu",
    input_modality="rna",
    target_modality="protein",
    input_transform=torch.log1p,
)
loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

for x, y in loader:
    prediction = model(x)
    loss = loss_fn(prediction, y)
```

包装器只保留两个模态均存在的 observation；若没有配对 observation，或 union/reference
特征空间中仍有未测量特征，则拒绝初始化并提示改用 `H5MuDataset` 及其 mask。

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

替换已经入库的同一数据集必须显式执行：

```bash
PYTHONPATH=src python -m iscdc.cli import-dataset dataset.h5mu metadata.yaml --replace
```

导入会在临时目录完成复制、SHA-256 计算、结构验证和元数据一致性检查。全部成功后，
正式目录包含：

```text
dataset.h5mu
metadata.yaml
manifest.json
validation_report.json
checksum.sha256
auxiliary/              # 可选；仅在注册辅助文件后创建
```

重复的 `dataset_id` 默认会被拒绝。管理员可显式使用 `import-dataset --replace` 原子替换
同一数据集；替换必须保持 `dataset_type`，衍生数据还须保持
`derivation.construction_type`、按顺序排列的 `source_dataset_ids`、`split_id` 和
`challenge_type` 不变，并会校验和保留已注册的辅助文件。校验或事务失败时原目录与
catalogue 记录保持不变。该命令不提供在线编辑或删除功能。

目录严格只接受 `schema_version: "1.2"`，不兼容 1.1 或更早版本。恰好两个模态时必须为
`same_unit`；三个及以上模态可为 `same_unit` 或 `partially_shared`，但不接受 `unpaired`。
导入器不会隐式裁剪 observation，不符合规则的数据应先在 `temp/` 中整理。`full` 可直接导入；导入
`train` 或 `test` 前，必须先导入其 `derivation.source_dataset_ids` 引用的全部 `full`
数据集。导入器会对照来源文件验证每个来源观测对象和特征合并策略；若目录中已经存在
相同 `split_id` 的另一侧，还会检查 train/test 的 `challenge_type` 一致且不包含重复来源
观测对象。

### 可选 cell type 注释

`cell_type` 只存放在顶层 `mdata.obs["cell_type"]`，不写入 `metadata.yaml`，也不增加
catalogue 列、网页/API 字段或筛选项。原始公开来源提供可与全部或经核验子集 observation
逐行对齐的离散标签时可以加入。部分覆盖时，只对来源中确实没有记录的 observation 使用项目
保留类别 `Unannotated`；重复、外来、冲突、空白或无法唯一对齐的来源行仍是错误，不能借此
补齐。不得用聚类或模型推断结果填充 canonical `cell_type`。

Database 页面可以另外显示独立、版本化并经启动校验的 cell type sidecar；其中的推断标签、
confidence、`Mixed` 和 `Uncertain` 不属于本字段，也不会写回 `.h5mu` 或公共 JSON。该边界见
[Cell type 空间可视化](#cell-type-空间可视化)。

存在时必须是无序 pandas categorical，所有 category 都是非空、无首尾空白的字符串，并且
不得保留未使用 category。`Unannotated` 必须是最后一个 category；使用它时必须在
`mdata.uns["cell_type_provenance"]` 记录 1.0 版来源契约：以直接来源 dataset ID 为键，保存
原始文件名、绝对 HTTP(S) URL、SHA-256、observation/label 对齐列，以及当前文件内实际的
annotated/unannotated 数量。完整来源列不需要该对象；不含 `cell_type` 时也不得残留该对象。
保留来源的分类层级、语义与拼写，不强行把不同数据集映射到统一 ontology；来源明确使用
`Unlabeled` 等类别时可以原样保留，它不等同于项目补充值 `Unannotated`。完整规则见
[数据库存储规范 1.2 的顶层观测章节](doc/数据库存储规范_v1.2.md)。

升级前若发现非空的旧版 SQLite 目录，应用会停止并提示备份和重新导入，不会自动删除
已有记录；空的旧版目录会自动重建为当前目录结构。
`manifest_version` 独立描述导入清单格式，与 `.h5mu` 的 `schema_version` 无关。新导入使用
manifest 1.1；它在保持原有主文件字段不变的基础上增加 `auxiliary_files` 列表。应用仍兼容
没有辅助文件字段的 manifest 1.0。

### 辅助文件

WSI 等不属于组学模态、但与某个已导入数据文件直接关联的大文件，可通过 CLI 注册为辅助
文件：

```bash
PYTHONPATH=src python -m iscdc.cli add-auxiliary-file DATASET_ID FILE \
  --id he_wsi \
  --label "H&E whole-slide image" \
  --source-url https://example.org/source.ome.tif \
  --media-type image/tiff
```

命令只接受常规文件，将源文件复制到数据集目录内的 `auxiliary/` 子目录，在复制过程中计算
SHA-256，并原子更新 `manifest.json`。辅助文件 ID 在单个数据集内唯一且只使用小写字母、
数字、下划线或连字符；命令拒绝符号链接、路径穿越、未知数据集和已有 ID/文件，不执行覆盖。
主 `checksum.sha256` 和 `validation_report.json` 继续只描述 `.h5mu`，辅助文件大小、哈希和
来源记录在 manifest 中。注册后需重启应用，使启动时的只读辅助文件索引重新加载。

已注册辅助文件 ID `he_wsi` 的 Database 可直接生成本地 WebP 缩略图：

```bash
PYTHONPATH=src python -m iscdc.cli generate-wsi-thumbnails DATASET_ID
PYTHONPATH=src python -m iscdc.cli generate-wsi-thumbnails DATASET_ID --force
PYTHONPATH=src python -m iscdc.cli generate-wsi-thumbnails --all --force
```

命令只处理 `full` Database。有金字塔的 TIFF 会读取最小且最长边不低于 640 px 的层；
没有金字塔时读取完整图像。最终保持整张切片和原始纵横比，使用 Lanczos 缩小到
最长边 640 px，不做组织区域裁剪，并以 RGB WebP `quality=85`、`method=6` 编码。默认拒绝
覆盖；`--force` 会在校验新 WebP 后原子替换旧图。`--all` 跳过没有 `he_wsi` 的 Database，
并以 JSON 汇总成功、跳过和失败结果。
批量中单个条目失败不回滚已成功输出；任一失败或没有生成任何缩略图时命令返回非零状态。
生成后必须重启应用，使启动时缩略图索引重新加载。

现有 schema 1.1 catalogue 升级时，先停止应用并执行只读预检，再运行正式迁移：

```bash
PYTHONPATH=src python -m iscdc.cli migrate-schema-1-2 --dry-run
PYTHONPATH=src python -m iscdc.cli migrate-schema-1-2
```

正式迁移会在隔离目录生成和验证完整的 1.2 副本，然后保留 1.1 备份并切换 active
catalogue。网站/API 验收通过后，使用迁移结果中的报告路径删除精确备份：

```bash
PYTHONPATH=src python -m iscdc.cli finalize-schema-1-2 data/migrations/schema_1_2_<UTC>.json
```

迁移报告永久保留旧/新 checksum、shape、配对类型和被裁剪的 observation ID；
`analytics.db` 不参与 schema 数据迁移。

Catalogue v3 删除 dataset License 元数据并升级到 v4 时，同样先停止应用并执行显式离线
迁移。命令会清理正式 metadata，以及存在时的 `temp/`、`exp/` metadata；`.h5mu` 和
manifest 不含该字段且不会重写：

```bash
PYTHONPATH=src python -m iscdc.cli migrate-catalogue-v4 --dry-run
PYTHONPATH=src python -m iscdc.cli migrate-catalogue-v4
PYTHONPATH=src python -m iscdc.cli finalize-catalogue-v4 \
  data/migrations/catalogue_v4_<UTC>.json
```

正式迁移在删除 `datasets.license` 物理列前备份 catalogue 和全部被修改的 metadata；失败
会自动恢复。只有 active catalogue、metadata 和所有 `.h5mu` checksum 均通过复核后，
finalize 才会删除 v3 备份。普通应用启动不会自动迁移旧 catalogue。

### 批量整理原始数据

`temp/` 下彼此独立的数据集使用两种 agent 角色和五个受控阶段处理：

1. `dataset_planner` 只读调查每个数据集，主 agent 汇总处理方案和待决定问题。
2. 用户回答后，主 agent 将全局决定传播到受影响的数据集，并提交最终计划待批准。
3. 只有获得明确批准后，`dataset_worker` 才在各数据集的隔离目录中转换和验证。
4. 主 agent 独立复核文件内容、元数据和验证报告。
5. 验收通过的数据集由主 agent 串行入库；导入后复核通过前不得删除原始数据。

角色配置默认位于 `.codex/agents/`，并受 `.codex/config.toml` 的本地并发限制。
完整的并发数、路径隔离、批准门槛、验收和清理规则见
[原始数据处理工作流](doc/原始数据处理规范.md)。

## metadata.yaml

YAML 是网站元数据的主记录。`.h5mu` 内部已有的数据库、样本和 assay 元数据必须在
YAML 中出现且值一致；YAML 可以包含文件内部没有的附加数据库元数据。

```yaml
database:
  schema_version: "1.2"
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
publication: null
```

Dataset metadata 不接受 `license`；数据使用条件应从 `source` 指向的原始发布页面确认。
`publication` 键必须存在，但在论文信息无法确认时可以为 `null`。
`coordinate_unit` 可使用 `array_index` 表示平台原生的离散阵列索引，例如 10x
Visium 的 `[array_col, array_row]`。如果没有执行明确的像素或物理坐标换算，应保留
`array_index`，不应将这些值重新标记为 `pixel` 或 `micrometer`。
多全集衍生数据的 `source`、`organism`、`tissue` 以及模态 `technology` 可以使用去重后的
字符串列表；目录 API 会原样保留字符串或列表形式。

`technology` 使用平台/方法级受控词表：`Immunofluorescence`、`MISAR-seq`、`SPOTS`、
`STARmap PLUS`、`Spatial ATAC-RNA-seq`、`Spatial CUT&Tag-RNA-seq`、
`Spatial-CITE-seq`、`Stereo-CITE-seq`、`Visium CytAssist` 和 `Xenium`。不要在该字段中
加入厂商前缀、模态、试剂版本、组蛋白标记或处理参数；新增技术须先扩展项目词表。

组蛋白修饰数据在 schema 1.2 中统一使用 `histone` 模态名。例如
H3K27me3 数据的 `modalities.histone.technology` 写为 `Spatial CUT&Tag-RNA-seq`，具体标记
只保存在 `database.histone_mark`。`h3k27me3` 等具体标记不直接作为模态名。

## 划分训练集和测试集

`iscdc.splitter` 是独立的 `.h5mu` 划分工具，不自动修改 SQLite，也不生成网站使用的
`metadata.yaml`。需要发布产物时，为 train/test 分别准备匹配的 metadata YAML，再使用
现有 `import-dataset` 命令逐个导入。来源文件必须符合根目录
[`doc/数据库存储规范_v1.2.md`](doc/数据库存储规范_v1.2.md)，并明确包含：

```yaml
schema_version: "1.2"
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
schema_version: "1.2"
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
坐标、观测 ID、模态成员关系以及 `histone_mark`、`genome_assembly` 等附加数据库元数据。
若来源包含有效 `cell_type`，两侧会按各自 observation 子集传播该列并移除未使用 category；
若仍使用 `Unannotated`，还会保留来源文件身份并重算两侧 annotated/unannotated 数量。
来源没有该列时，产物也不创建该列。

### 按完整全集组合

`compose` 不切分全集内部观测，而是将完整来源分配给 train 或 test：

```yaml
schema_version: "1.2"
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
每个输出侧仅在分配给该侧的所有来源都包含有效 `cell_type` 时保留该列，并按来源顺序及标签
首次出现顺序合并 category，`Unannotated` 固定置于最后；所有部分来源的 provenance 保持原始
文件身份并按输出 observation 重算数量。任一来源缺少该列时，该输出侧省略整列。

`spatial` 和 `compose` 配置都必须显式声明 `challenge_type`，可选值为：

- `same_slice`：同一物理切片内部划分。
- `cross_slice_same_subject`：同一个体的不同切片之间划分。
- `cross_subject`：跨个体划分，涵盖生物学重复及不同条件或发育阶段。

该字段会写入两侧 `uns["database"]["derivation"]`。发布产物时，对应
`metadata.yaml` 的 `database.derivation.challenge_type` 必须与文件内部取值一致。同一
`split_id` 的 train/test 必须使用相同值；缺少该字段的旧衍生文件需要补齐并升级到 schema 1.2 后
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

跨技术来源使用不同 feature ID 或空间坐标单位时，`compose` 还支持显式、可审计的
harmonization。该模式只与 `feature_merge_policy: intersection` 配合，先在全部 Challenge
来源上建立 canonical feature ID，再按首个 train 来源的顺序保留全局交集；同一 canonical
feature 对应多个 raw count feature 时使用 `sum` 聚合。配置示例：

```yaml
feature_harmonization:
  version: "1.0"
  scope: all_challenge_sources
  aggregation: sum
  modalities:
    rna:
      namespace: gene_symbol
      sources:
        full_a: {kind: var_column, column: gene_symbol}
        full_b: {kind: identity}
    protein:
      namespace: protein_marker
      sources:
        full_a: {kind: mapping_file, path: full_a_protein.yaml}
        full_b: {kind: mapping_file, path: full_b_protein.yaml}
coordinate_harmonization:
  version: "1.0"
  spatial_unit: region
  coordinate_unit: array_index
  sources:
    full_a: {kind: obs_columns, x: array_col, y: array_row}
    full_b: {kind: obsm, key: spatial}
```

feature 来源规则只能为 `identity`、`var_column` 或 `mapping_file`；mapping YAML 是 raw
feature ID 到 canonical ID 的非空映射，路径相对 compose 配置解析。所有 Challenge 来源和
最终模态必须完整声明，当前 `sum` 聚合只接受 `counts`。坐标规则只能从一个 `obsm` 矩阵或
两个顶层 `obs` 列读取。输出在 derivation 中记录全局摘要，在各 modality 的 `var`/`uns` 和
顶层 `uns["coordinate_harmonization"]` 中嵌入逐来源映射、hash 与坐标 provenance；导入器会
对正式 full 文件重算交集并核对矩阵和坐标。没有这些配置的旧 compose 行为保持不变。

组合产物的顶层观测 ID 和样本 ID 分别编码为
`<source_dataset_id>::<source_obs_id>` 和
`<source_dataset_id>::<original_sample_id>`。每个观测的原始来源仍保存在
`source_dataset_id`、`source_obs_id` 中，产物的 `uns["database"]["derivation"]` 会记录
划分 ID、Challenge 类型、来源全集、选择规则、特征策略、处理说明和
`random_seed: null`。多来源 train/test 必须使用上述 sample ID 编码；Challenge 详情页会在
File metadata 的 Source 和 Derivation 的 Source databases 中逐 sample 展示对应关系。
Challenge JSON 中每侧的 `sample_sources` 提供相同的结构化映射，包含 derived/original sample、
来源 database ID/标题及该 database 自身的 Source；单来源文件返回空列表。

## Challenge distribution-shift 难度参考

可以通过固定的 domain classifier 离线评估全部完整 Challenge。该功能使用 RNA 输入判断每个
observation 来自 train 还是 test；held-out AUROC 越高，只表示在统一 representation 下两侧越
容易区分、跨分布泛化要求可能越强。它不是绝对难度、biological shift 或下游模型性能估计。

先安装可选分析依赖：

```bash
conda activate iscdc
python -m pip install -r requirements-difficulty.txt
```

然后运行：

```bash
PYTHONPATH=src python -m iscdc.cli evaluate-challenge-difficulty
```

若默认输出已经存在，应在确认需要发布新快照后执行：

```bash
PYTHONPATH=src python -m iscdc.cli evaluate-challenge-difficulty --force
./deploy_test.sh restart
```

`--seed` 可覆盖默认随机种子，`--input-modality` 可为整个榜单选择另一输入模态，`--output`
可写入不供网站读取的独立实验快照。网站只读取 `catalog.db` 同目录下固定名称
`challenge_difficulty.json`；不同输入模态生成的结果不得混入同一个排名。

默认以 seed 42 对每侧最多抽取 5,000 个 observation，进行 5 次重采样和每次 5-fold
held-out evaluation。raw counts 先按 observation 归一化至总量 10,000 并 `log1p`；已经声明为
`normalized` 的输入不再二次归一化。每个 fold 只使用其 classifier training 部分选择最多
2,000 个高方差共同 feature、拟合 whitened 50 维 PCA 和固定 L2 logistic regression，避免
representation 或 classifier 泄漏 held-out 数据。train/test feature 仅取稳定 ID 交集，标记为
未测量的 feature 会被排除，不会以人工补零制造 domain 信号。

结果默认原子写入 `data/challenge_difficulty.json`；已有文件必须显式使用 `--force` 替换。
也可用 `--output`、`--seed` 和 `--input-modality` 生成独立快照，但不同输入模态的结果不应混在
同一个榜单比较。JSON 保留每折和每次重复 AUROC、mean/std AUROC、派生 shift score、global
及同 `challenge_type` percentile、实际样本/feature 数、文件校验和、随机种子与采样 ID hash、
稳定性统计和 warning。单个 Challenge 失败时仍会出现在报告中，但 rank/percentile 为 `null`，
且 CLI 返回非零状态。

网站在应用启动时读取并校验该快照，确认报告版本、Challenge 集合、类型、train/test 数据集 ID
和 SHA-256 均与当前 catalogue 一致后，才会发布其中的指标。重新生成报告后需要重启应用。
报告缺失、损坏、过期或单个 Challenge 评估失败时，目录和 API 仍可用，对应 difficulty 显示为
`Unavailable` 或 `null`。

Challenge 目录卡片和详情页只发布 mean AUROC、shift score 与 global percentile；标题旁的
`?` 会打开方法和解释限制说明。更完整的标准差、重复评估、采样和诊断信息只保留在离线报告
中，不作为网页核心指标。

sample、source dataset 及可用的 donor/slice 等层级只用于诊断潜在混杂，不参与 AUROC 修正。
domain classifier 无法区分 biological 与 technical shift；normalization、平台、batch 或样本边界
均可能造成高 separability，解释排名时必须结合 `challenge_type`、metadata hierarchy 和 warning。

## 网页和 API

- `/databases`：浏览和筛选 `full` Database 文件；有图条目显示紧凑缩略图。
- `/databases/{dataset_id}`：Database 元数据、模态规模、缩略图（如有）、校验信息和下载。
- `/challenges`：以 `split_id` 为单位浏览和筛选 Challenge，并可按 difficulty 从低到高或
  从高到低排序。
- `/challenges/{split_id}`：同页查看 Challenge 对应的 train/test 文件和 difficulty 核心指标；
  尚未配对时会明确标记缺失侧。
- `/api/databases`、`/api/databases/{dataset_id}`：Database JSON 列表和详情。
- `/api/challenges`、`/api/challenges/{split_id}`：按 Challenge 聚合的 JSON 列表和详情。
- `/downloads/{dataset_id}/{kind}`：下载 h5mu、metadata、manifest、validation 或 checksum。
- `/downloads/{dataset_id}/auxiliary/{auxiliary_id}`：下载所属数据文件的辅助文件，支持 HTTP
  Range 请求和断点续传。
- `/healthz`：供部署脚本和监控使用的健康检查，不创建访客会话或行为事件。

网页列表使用 `q`、`organism`、`tissue`、`modality`、`technology` 和 `spatial_unit`
筛选；Challenge 列表还支持 `challenge_type` 和
`sort=newest|difficulty_asc|difficulty_desc`，API 列表另外支持 `limit`、`offset`。difficulty
排序先作用于完整筛选结果再分页；没有可用指标的 Challenge 在两个 difficulty 方向中都排在
末尾。
Challenge 的任一侧满足全部筛选条件时，响应
仍会返回该 Challenge 已导入的完整两侧。若同一个 `split_id` 下存在多份 train 或多份
test，目录会报告完整性错误，不会静默选择其中一份。

当 train/test 由多个 source Databases 组成时，详情页 File metadata 的 Source 和 Derivation
的 Source databases 均按 sample 逐行显示，不对重复 Source 合并。Challenge JSON 的每侧通过
`sample_sources` 返回 derived sample、original sample、source Database ID/标题及该 Database
自身的 Source；full 和单来源文件返回空列表。原有聚合 `source` 与
`derivation.source_dataset_ids` 字段保持不变。

Database 缩略图是样本或切片的辅助预览，并不都属于 H&E，也不能替代原始 WSI。页面按
`dataset_id` 与 `<dataset_id>.webp` 精确匹配；没有匹配文件时不渲染图片。该展示信息仅存在
于 HTML 页面，不改变 catalogue schema、导入流程或 Database JSON API。已注册 `he_wsi` 的
Database 是例外：其 WebP 必须直接由对应 WSI 副本生成，但仍只是有损的展示预览。

已注册的辅助文件只在所属 Database 或 Challenge 文件的详情下载面板中显示，不形成独立的
目录条目或详情页。Database/Challenge JSON 响应通过 `auxiliary_files` 返回稳定 ID、标签、
文件名、媒体类型、大小、SHA-256、来源和本站下载 URL；某个数据文件的辅助清单无效、文件
缺失或大小不符时，该清单会被隐藏，不影响主目录、API 或主文件下载。

Challenge JSON 使用顶层 `challenge_type` 返回分类，使用 `status` 表示 `complete`、
`missing_train` 或 `missing_test`，并通过 `train`、`test` 字段返回对应的完整文件记录或
`null`。`difficulty` 返回可空的 `mean_auroc`、`domain_shift_score` 和全局
`difficulty_percentile`；这些字段仅是 train/test separability proxy。旧版 `/datasets`、
`/datasets/{dataset_id}` 和 `/api/datasets*` 路由已经移除，不提供兼容重定向。

生产部署时可以在下载路由前增加 Nginx；代理必须保留 HEAD、`Range`、`If-Range`、
`Content-Range` 和 `Accept-Ranges` 语义。当前开发版本由 FastAPI 直接传输文件。

## 目录

```text
src/iscdc/       应用、校验、导入和数据划分代码
tests/           自动化测试（包括 splitter 合成数据与必需的真实数据测试）
doc/             项目文档；annotation 专题文档位于 doc/annotation/
assets/templates 网页模板
assets/static    页面样式；本地 `<dataset_id>.webp` 缩略图被忽略
assets/he_wsi_thumbnails  仅用于缩略图采集的本地来源图像（忽略，不纳入版本控制）
assets/examples  示例人工元数据
data/            catalog.db、analytics.db、difficulty 快照、已导入文件及辅助文件（忽略）
temp/            待整理数据及其隔离的转换产物（不纳入版本控制）
exp/             本地真实输入、手工测试配置和实验产物（不纳入版本控制）
.codex/          本地 agent 角色和并发配置（不纳入版本控制）
```
