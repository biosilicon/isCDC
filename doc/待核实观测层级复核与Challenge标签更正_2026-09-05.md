# 待核实观测层级复核与 Challenge 标签更正（2026-09-05）

本轮承接[首次空间层级审计](空间观测层级收录审计_2026-09-05.md)，按用户指令进一步核实
S055、U005、U007、U010、U012，并更正两项 Challenge 的错误 `region` 标签。
没有新增数据文件；目录仍为 **187 个 Database、29 项 Challenge，共 245 份文件**。

## 五个条目的最终处置

| Entry | 观测层级与身份 | 当前处理 |
| --- | --- | --- |
| S055 | 已定位 Fig.6 正常人肝脏，真实共同单位为 cell | 保留候选；改为缺公开逐细胞配对产物的 hold，纠正 RNA/物种/旧 Fig.3 错绑 |
| U005 | 高置信定位 Hendriks 2024 蛋白/脂质研究，约1 mm² ROI | 纠正 RNA 模态误记，按 region level 排除 |
| U007 | 高置信匹配 2023 TBI 论文，作者按 cluster/ROI 整合 | 年份从 2024 更正为 2023，按 region level 排除 |
| U010 | 泛化队列描述，无唯一来源、无已验证共同观测轴 | 保留身份待核实，不绑定或导入 |
| U012 | 高置信重复于 U008，同切片 Xenium cell 级 RNA/MSI | 记录 duplicate_of_entry_id=U008，保留原行，不另建下载/入库任务 |

当前 88 行 intake 中，区域级排除 **16 行**；另有 U012 一行重复记录，S055 一行缺公开配对数据，
U010 一行身份待核实。其他候选原有缺源、访问和配对验证限制继续有效。

### S055：单位明确，缺的是公开细胞矩阵

REDCAT 论文 Fig.6 为正常人新鲜冷冻肝脏，同片 TPEF–SRS 与 47-plex CODEX，按映射细胞质心
一一比较。共同单位是细胞，ROI/圈层统计为下游汇总。实际模态为蛋白与光学代谢/生化特征，
没有原清单所写的 RNA。[作者论文](https://www.nature.com/articles/s41592-026-03180-0)

Zenodo 21938834 的八个文件全属 Fig.3 淋巴瘤，不适用于肝脏；本地四 TIFF、四 PNG 原样保留并
标记错绑。Fig.6 source-data XLSX 只给区域数值/比例，作者代码预期的肝脏 cell 表、细胞映射和
输入尚未公开；SRS hyperspectral 数据需申请。因此状态为
`hold_missing_public_cell_pairing_artifacts`。
[官方仓库](https://zenodo.org/api/records/21938834)、
[作者固定代码版本](https://github.com/yal026/REDCAT/tree/b79f00fdf865b789575da56cf47597df92beacce)

解除暂停需要肝脏 CODEX 和 SRS/TPEF 逐细胞表、共享 ID/映射、质心、分割与配准 QC，以及真实
sample/section 身份；不从区域统计、论文图或错绑组织重建原始配对矩阵。

### U005：按作者时间线纠正模态，排除区域级产物

2024 Hendriks GBM 论文涉及 MALDI 与 LMD–LC-MS 蛋白/脂质，联合定量为约 1 mm² 的肿瘤/坏死
ROI，没有 RNA。2025 “One section, two worlds” 是同片 Xenium RNA 与逐细胞 MSI，符合模态，
但不符合原条目的 2024 年；其已单列为 U008。
[2024 论文](https://doi.org/10.1021/acs.analchem.3c05850)、
[2025 论文](https://doi.org/10.1038/s41598-025-26735-1)

进一步查作者官方发表记录和 ASMS 2023 前身报告后，2024 同片人 GBM 流程可高置信定位到上述
蛋白/脂质研究。2025 RNA/MSI 论文有独立时间线，并将2024工作列作独立参考文献。综合作者、
年份、组织、同片设计及分列的U008，原RNA字段按模态误录纠正，U005改为区域级排除。
[作者官方记录](https://cris.maastrichtuniversity.nl/en/publications/maldi-msi-lc-msms-workflow-for-single-section-single-step-combine/)、
[ASMS 2023 官方程序](https://www.asms.org/docs/default-source/past-annual-conference-programs/71st-asms_2023_conference-program.pdf)

原工作簿没有直接DOI，编者意图仍是汇聚证据支持的高置信推断；记录推断性质及所有旧字段，
不声称恢复了原始引用，也不把该条目替换为2025细胞研究。

### U007：年份纠正后按区域级排除

疾病、人脑外伤、RNA/代谢组合与“代谢组需申请”的访问描述一致匹配
*Integrated spatial transcriptome and metabolism study reveals metabolic heterogeneity in human injured brain*，
DOI `10.1016/j.xcrm.2023.101057`。原表无直接 DOI；这是结合 PubMed、Crossref 出版记录与
其他候选排除后的高置信身份推断，历史字段与推断依据均保留。
[原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10313933/)

作者将 ST clusters 与区域性 MSI profile 对应，并人工绘制/转移 ROI；补充比较 high/low
metabolite areas。55 µm Visium spot 和独立 100 µm MSI 扫描步长不是共同观测轴。
没有找到作者发布的逐 spot 特征矩阵。论文的 GSE223245 实际为 16 份全血表达芯片，不能作为
该空间研究来源。状态从 hold 改为 `excluded_region_level`。
[官方补充](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC10313933/supplementaryFiles)、
[GEO 记录](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE223245)

### U010：候选单位可以说明，但无法唯一绑定

| 具体候选 | 原生测量及共同轴问题 |
| --- | --- |
| Ji 2020 cSCC | 17,064 ST spots 与 55,832 MIBI 分割细胞是独立观测集合。[论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC7391009/) |
| Hwang 2022 PDAC | MIBI FOV 内细胞、GeoMx 形态 AOI 与独立 snRNA 细胞核；整合在标本/模型层面。[论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10290535/) |
| Rahim 2023 HNSCC | 解离 scRNA/TCR 与独立 MIBI 视野；患者/组织配对不代表同一空间细胞的 RNA/蛋白。[论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC10348701/) |

原条目仅有泛化标题、2023、人肿瘤和检索链接，没有唯一来源。三个候选均不能作为同单位成品
导入，也不能把其中一项当作 U010 的确定身份。继续 hold，待唯一题名/DOI/仓库线索。

### U012：作为 U008 的高置信重复项保留追溯

2025、人 GBM、靶向 RNA、脂质 MSI、same-section 和无公开 accession 的组合一致匹配 U008。
同一切片先做 5 µm MALDI，再做 Xenium，按细胞边界的像素重叠面积加权得到逐细胞谱，作者报告
8,318 个共同细胞。因此单位为 cell。GBM 数据因患者同意范围限制不能直接公开。
[作者论文](https://www.nature.com/articles/s41598-025-26735-1)

Ma 2025 为相邻 TMA 切片，Tsyben 2025 为 Visium 与同位素代谢物、相邻切片，Godfrey 2025
为乳腺/肺癌，MALDI-ISH 为小鼠脑；这些候选不同时满足原行的身份字段。原 U012 无 DOI，所以
duplicate 是高置信推断，保留原始行和其他候选证据；若出现不同权威来源，可恢复独立复核。
U008 保留为 canonical 候选且不入库，U012 不再形成独立任务。

## 两项 Challenge 的标签更正

四份文件将以下三个嵌入位置及对应 YAML/catalogue/manifest 从 `region` 改为 **`spot/bin`**：
`uns/database/spatial_unit`、`uns/database/derivation/coordinate_harmonization/spatial_unit`、
`uns/coordinate_harmonization/spatial_unit`。两侧必须有相同的 harmonization 摘要，故使用共同的
复合标签；逐来源原始 `input_spatial_unit` 保留，各文件说明明确实际组成。

| split_id | train | test |
| --- | --- | --- |
| `human_tonsil_visium_to_spatial_citeseq_v1` | 4,194 Visium spots | 2,492 Spatial-CITE-seq bins |
| `mouse_spleen_spots_rep2_spatial_citeseq_to_spots_rep1_v1` | 2,768 SPOTS spots + 1,303 Spatial-CITE-seq bins | 2,653 SPOTS spots |

`spot/bin` 不表示区域聚合、相同物理分辨率或新的 observation 类型。现有 schema 允许字符串，
验证器每文件如实给出一个 `nonstandard_spatial_unit` 提示，共四个；来源与成对验证错误均为零。
没有修改验证器来隐藏这些提示。

先在隔离候选目录修改文件，对全部 HDF5 datasets/attributes 逐项比较，确认除三个单位字符串外
所有内容一致，包括矩阵、细胞类型、观测/特征/样本身份、空间坐标和来源 provenance。每份文件
均经正式 `validate_h5mu` 验证器核对真实 full 来源及更正后的 peer。由于单文件 replacement
会遇到尚未更正的 peer，本次将四份候选全部验证后，在一个 SQLite 事务中同时切换目录和更新
四条记录；异常路径可恢复原目录和数据库事务，原文件永久保留在本地审计归档。

entry ID、dataset ID、train/test 类型、source IDs、split ID 与 challenge type 均保持不变。
241 条无关 catalogue 记录逐字段一致；当前正式目录的 `region` 标签数量为 **0**。
本地 compose 配置和 README 示例也已同步，旧生成产物作为历史材料保留，不能再次发布旧标签。

## 难度快照与发布验收

四份 H5MU 的 SHA-256 已改变，因此按固定域分类器流程完整重算全部 **29 项 Challenge、725 折**。
重算前实际读取并核验 58 份 train/test 文件 SHA-256；旧快照保留，未手改任何 snapshot checksum。

- 评估成功 **29**，失败 **0**；完整启动校验通过。
- 与旧快照的 29 项 mean AUROC 比较，最大绝对差为 **0.0**。
- 新快照 SHA-256：`e6e27cb7bc1a5c74f42cc93b0cac5af1570d7588e457ed7d0ab3d895c857d88d`。
- 发布于 **2026-09-05T10:02:15.003653+00:00**；随后重启服务。
- 难度验收 69 个 HTTP 响应通过，覆盖全部 29 项 API/详情指标、升降序和分页。
- 标签验收 14 个 HTTP 响应通过；四份数据的 JSON 与 metadata 下载一致，
  `region` 筛选返回 0 项，`spot/bin` 筛选返回 2 项，页面显示更正后的标签。

本轮没有修改生产算法、schema 或 API 行为，没有运行完整 pytest 套件。四份数据经过实际文件
语义比较、正式来源/peer 验证、四处元数据核对、SHA-256 及线上发布验证。

## 清单、历史与本地制品

五个条目的最新结论写入 acquisition manifest/row JSON/summary；S055 同步 staged manifest、
planner 和暂存 inventory。身份更正逐字段记录前值、后值、主来源和推断性质，原始工作簿不改。
U008 增加可追溯重复关系，既有公开性状态不变。历史 88 行初审结果和前次撤下的31份成品继续保留。

本轮详细材料位于被忽略的 `temp/hold_units_challenge_correction_20260905/`：
`followup_findings.json`、`intake_update.json`、`original_datasets/`、`correction.json`、
`*.candidate_validation.json`、`baseline.json`、`evaluation.log`、`challenge_difficulty.before.json`、
`challenge_difficulty.candidate.json`、`publication_audit.json`、`live_verification.json`、
`correction_verification.json`。运行数据和归档不随 Git 分发。
