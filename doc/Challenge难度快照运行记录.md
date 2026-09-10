# Challenge 难度快照运行记录

## 2026-09-10：空间分辨率元数据迁移后的复用发布

用户批准将空间分辨率改为 Single-cell、Near-cellular、Spot-level 后，289 份正式文件
完成元数据迁移，其中 58 份 train/test 的文件校验和发生变化。使用现有
`evaluate_catalogue` 流程核验实际文件和评估输入指纹，并在本次一次性脚本中禁止调用
分类器；29 个 Challenge 全部复用，`reused_count=29`、`evaluated_count=0`、
`failure_count=0`。AUROC、shift score、全局及类别百分位与迁移前精确一致。

2026-09-10 03:32:52 UTC 完成切换后重启服务，04:12:26 UTC 线上验收通过：29 项 API
难度值与快照一致，升降序排序及分页正确。此次迁移共核验 60 个 HTTP 响应，另覆盖全部
289 份文件的分类、筛选、代表性下载及 32 份有效可视化的保留展示。

旧快照、原目录和文件备份保留在 `data/backups/spatial_resolution_20260910/`；一次性
脚本、批准清单、执行报告及 `live_verification.json` 保留在
`temp/spatial_resolution_20260910/`。分类及既有可视化问题见
[空间分辨率分类](空间分辨率分类.md#批准执行与上线结果)。未重新训练或增加新的校验和体系。

本文件记录本地目录的日期化发布结果。评估方法和使用命令见
[README](../README.md#challenge-distribution-shift-难度参考)，快照契约见
[数据库存储规范 1.2](数据库存储规范_v1.2.md)。运行快照、输入数据、备份和详细审计均不纳入
版本控制；新 checkout 需为自己的目录生成快照。

## 2026-09-05：恢复 entry-ID 修正后的难度展示

历史 entry-ID 修正重写了 train/test H5MU，旧快照保存的 SHA-256 与当前 catalogue 不一致，
应用因而将难度信息显示为 `Unavailable` / `null`。本次重新运行固定 domain classifier，
对全部 29 个 Challenge 重新计算指标与全局排名，严格校验后原子替换正式快照，并重启服务。
截至本次验收，该问题已解决。

### 评估与目录基线

| 项目 | 本次记录 |
| --- | --- |
| Catalogue | v5；242 个正式文件：184 full、29 train、29 test；30 个不同 entry_id |
| 评估结果 | 29 个 Challenge，29 成功、0 失败 |
| 输入文件 | 58 个 train/test H5MU，实际 SHA-256 全部与 catalogue 一致 |
| 输入与采样 | RNA，seed 42，每侧最多 5,000 个有效 observation |
| 重复与交叉验证 | 5 次重采样 × 5 折，共 725 折 |
| Representation / classifier | 每折训练部分选择最多 2,000 个高方差共同特征；whitened 50 维 PCA；固定 L2 logistic regression |
| Mean AUROC 范围 | 0.5490564166179068–1.0 |
| 与旧结果比较 | 29 项 mean AUROC 全部相同，最大绝对差为 0 |
| 正式发布时间 | 2026-09-05 04:31:32 UTC |
| 线上验收完成时间 | 2026-09-05 04:32:09 UTC |

本次是完整重新评估，结果与旧指标相同；发布的新快照记录了当前输入文件的 SHA-256。
AUROC 相同不免除文件身份校验，也不能据此将以后过期的快照直接视为有效。

报告保留以下方法诊断，均未导致评估失败：

| Warning code | 出现次数 |
| --- | ---: |
| `low_sample_size` | 3 |
| `hierarchy_domain_aligned` | 14 |
| `small_category_pool` | 2 |
| `preprocessed_input` | 1 |

出现次数按 warning 记录统计，同一 Challenge 可以有多条诊断。解释指标时仍需考虑样本量、
层级混杂、类别规模和输入预处理；这些指标仅表示 train/test 分布可分性。

### 发布验收与制品身份

- 候选快照通过应用使用的 `load_difficulty_snapshot` 严格校验，覆盖全部 29 个 Challenge，
  核对类型、输入模态、两侧 ID/SHA-256、指标和全局百分位一致性；另核验默认参数及每项 25 折结果。
- 正式快照通过 `write_report_atomically(..., force=True)` 发布，发布后重新加载校验通过。
- 通过 `deploy_test.sh restart` 重启原有 iscdc 服务，健康检查通过。
- 69 个线上 HTTP 响应全部通过：29 个详情 API、29 个详情页面，以及健康检查、列表和分页
  请求；29 项 API 指标与新快照精确一致，页面按规定精度展示，升降序和分页结果均正确。
- 评估和发布前后 catalogue SHA-256 不变。未修改项目代码，未运行完整测试套件。

| 制品 | SHA-256 |
| --- | --- |
| `data/catalog.db` | `4c9ada1f3cf0ebe8a5758b481573ddadea6cee55f09e19a278e58ec6f2f2b5ae` |
| `data/challenge_difficulty.json` | `12cd0e40070bd7214ffff05012a81b52352de999d128403ad9a7a31d06f1d275` |

本地审计位于 `temp/difficulty_recompute_20260905T042101Z/`：

- `baseline.json`：目录哈希、旧快照失效原因及 58 个输入文件身份。
- `challenge_difficulty.before.json`：旧快照备份。
- `evaluation.log`：输入哈希检查、逐 Challenge 进度及 CLI 成功结果。
- `challenge_difficulty.candidate.json`：完成评估的候选快照。
- `publication_audit.json`、`live_verification.json`：发布与线上验收证据。
- `publish.py`、`verify_live.py`、`completion.md`：本次发布/核验脚本及本地完成记录。

此前 `temp/approved_entry_import_handoff_20260904.md` 中“旧 difficulty 快照仍不可用”的描述
属于当时状态；本次发布已解决该项，旧交接文件保留历史内容。

## 2026-09-05：区域级收录范围调整后的复核

S020/S022 的31份full下架，当前187个Database、29项Challenge。被撤下的full没有Challenge依赖，
29项Challenge的58份输入实际SHA-256重新计算后均与catalogue一致，快照与本页前述2026-09-05重算结果
逐字节相同，因此本次没有重新评估或替换快照。服务重启后全部29项指标、升降序与分页验证通过。
四份历史`region`标签的派生文件经直接来源obs逐行核验，实际是spot/bin，保留原文件。
详见[空间观测层级收录审计](空间观测层级收录审计_2026-09-05.md)。

## 2026-09-05：两项 Challenge 空间标签更正后的完整重算

四份跨平台 train/test 将错误 `region` 标签更正为 `spot/bin`。所有观测、特征、矩阵、坐标和
来源身份保持不变，H5MU SHA-256 已变化；旧快照保留，58份实际输入校验通过后完整重算29项、725折。
成功29项、失败0项，29项mean AUROC相对旧结果最大绝对差为0.0。
新快照SHA-256为 `e6e27cb7bc1a5c74f42cc93b0cac5af1570d7588e457ed7d0ab3d895c857d88d`，发布时间为2026-09-05T10:02:15.003653+00:00。
启动严格校验与重启后全部详情指标、升降序和分页通过；region筛选为0、spot/bin筛选为2。
上一节“保留原文件、快照未变”为首次范围审计的历史状态，本节更正已替代该状态。
详情与审计路径见[后续复核与标签更正](待核实观测层级复核与Challenge标签更正_2026-09-05.md)。

## 2026-09-05：修正无条件全量重算，启用经验证的结果复用

此前命令每次都训练完整目录的分类器，操作规范又将文件 SHA-256 变化直接视为全量重算条件。
这导致 entry-ID 和空间标签修正后，即使评估输入完全相同，也重复执行了 725 折训练。
以上运行记录保留历史事实，后续维护以本节及 README 的增量刷新规则为准。

现在 `evaluate-challenge-difficulty --force` 默认核验实际文件并复用已有成功结果。文件身份变化
时比较评估输入指纹，指纹一致则同步文件 SHA-256 并保留原指标；新增、真实输入变化、失败或
无法验证的条目才评估。增删条目更新全局相对排名，`challenge_type` 更正仅更新分类排名。
`--force` 仅允许覆盖；`--recompute` 才显式要求重新训练全部分类器。网站启动时的完整集合、
类型、ID、SHA-256 和指标一致性检查保持不变。

本次为当前有效旧快照补建版本化输入指纹，运行中显式禁止调用分类器，任何尝试训练都会失败，
因此本次验收确认的是实际复用，不能将它记作 29 次重新评估。

| 项目 | 本次记录 |
| --- | --- |
| Catalogue | 245 个正式文件：187 full、29 train、29 test；目录 SHA-256 保持不变 |
| 文件核验 | 58 个实际 train/test H5MU 与目录 SHA-256 一致 |
| 结果复用 | `reused_count=29`、`evaluated_count=0`、`failure_count=0` |
| 首次核验与补建指纹 | 63.63 秒；未调用分类器 |
| 指标保留 | 29 项全部原有逐折结果、AUROC、排名和诊断精确不变 |
| 真实元数据变更验证 | 四份跨平台文件旧 `region` 与新 `spot/bin` 的文件 SHA-256 不同，评估输入指纹全部相同 |
| 发布时间 | 2026-09-05 10:36:03 UTC |
| 线上验收 | 2026-09-05 10:37:03 UTC；69 个 HTTP 响应通过，含 29 项详情页面/API 和升降序分页 |

| 制品 | SHA-256 |
| --- | --- |
| `data/catalog.db` | `6e88108888a16e0ab6728b4ec9e60182eaff96abbd5771dde9fa9e8d2b6dd679` |
| 原快照备份 | `e6e27cb7bc1a5c74f42cc93b0cac5af1570d7588e457ed7d0ab3d895c857d88d` |
| 新 `data/challenge_difficulty.json` | `3fb808e8c763b0121868112a3663514dea02ec690692a441279484ffde28ada5` |

本地审计保存在 `temp/difficulty_incremental_20260905/`：旧快照与目录备份、`bootstrap.py`、
`bootstrap.log`、候选快照、`publication_audit.json`、`verify_live.py` 和 `live_verification.json`。
快照保持报告 1.0，通过可选离线字段增加指纹和复用/评估计数；无需迁移 catalogue 或 H5MU。

验证覆盖 42 项相关测试：难度评估、快照校验与 entry-ID 快照同步；包括不训练断言、元数据和
其他模态修改、实际输入变化、增删/分类排名、失败与损坏缓存、旧格式补建、实际文件身份不符、
评估期间文件变动、参数/方法/软件变化及显式重算。相关 Ruff 检查通过，未运行完整测试套件。
