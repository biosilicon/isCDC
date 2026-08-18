# 已入库数据集 cell type 来源核验记录

核验日期：2026-08-15

## 判定口径

本记录以当前 catalogue 中 35 个 `full` 数据集为范围。只有原始公开来源提供可与当前文件
全部顶层 observation 一一对齐的离散细胞类型标签，才判定为“包含”。论文中仅展示聚类、
label transfer 图或空间区域命名，但未公开逐 observation 对齐标签的，不计为包含。标签覆盖
不完整时也不写入 `cell_type`。

## 确认包含 cell type 的数据集

| dataset_id | 来源证据与对齐结果 | 采用字段 | 规范化结果 |
|---|---|---|---|
| `2023_nc_10x_breast_cancer_HBC_rep1` | [GEO GSM7780153](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM7780153) 与 [10x Genomics companion repository](https://github.com/10XGenomics/janesick_nature_comms_2023_companion)；当前 167,780 个 observation 已完整携带标签 | 来源 cell type | 20 个来源类别；现有列已符合无序 categorical、无空值、无空白和无未使用 category 的要求 |
| `starmap_plus_ad_13m_disease_rep1` | [Zenodo 7332091](https://zenodo.org/records/7332091) 的 `metadata.csv`；筛选 `13months-disease-replicate_1` 后 10,372 行，去除公开 `NAME` 的数据集前缀后与当前 obs_names 精确一一对应 | `top_level_cell_type` | 13 个类别，完整覆盖；`sub_level_cell_type` 另有 33 类但不作为本库统一字段 |
| `starmap_plus_ad_13m_disease_rep2` | [Zenodo 7332091](https://zenodo.org/records/7332091) 的 `metadata.csv`；筛选 `13months-disease-replicate_2` 后 9,634 行，去除公开 `NAME` 的数据集前缀后与当前 obs_names 精确一一对应；[Broad SCP1375](https://singlecell.broadinstitute.org/single_cell/study/SCP1375/integrative-in-situ-mapping-of-single-cell-transcriptional-states-and-tissue-histopathology-in-an-alzheimer-disease-model) 亦为该研究的公开入口 | `top_level_cell_type` | 13 个类别，完整覆盖；类别为 `Astro`, `CA1`, `CA2`, `CA3`, `CTX-Ex`, `DG`, `Endo`, `Inh`, `LHb`, `Micro`, `OPC`, `Oligo`, `SMC` |

## 实施状态

截至核验日期，catalogue 共 89 个文件（35 `full`、27 `train`、27 `test`）。上述三个 `full`
是仅有的 `cell_type` 数据来源；乳腺癌 full 保留原有合规列，两份 STARmap full 已完成来源
回填。以下四个依赖文件已从 full 重新生成并逐 observation 传播标签：

- `2023_nc_10x_breast_cancer_HBC_rep1_train`：98,816 observations，20 类；
- `2023_nc_10x_breast_cancer_HBC_rep1_test`：68,964 observations，20 类；
- `starmap_plus_ad_13m_disease_train`：10,372 observations，13 类；
- `starmap_plus_ad_13m_disease_test`：9,634 observations，13 类。

六个替换文件的 catalogue 记录、manifest、validation report、文件大小与 SHA-256 已复核一致；
Challenge difficulty 随后按完整 27 个 Challenge 重算，27 项均成功并通过应用启动校验。

## 未确认包含可直接入库标签的数据集

主要公开核验入口包括 [GSE205055](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205055)、
[GSE213264](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE213264)、
[GSE198353](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE198353)、
[Zenodo 7480069](https://zenodo.org/records/7480069)、
[Zenodo 10362607](https://zenodo.org/records/10362607)、
[10x Visium tonsil](https://www.10xgenomics.com/datasets/gene-protein-expression-library-of-human-tonsil-cytassist-ffpe-2-standard)
和 [10x Xenium RCC](https://www.10xgenomics.com/cn/datasets/xenium-protein-ffpe-human-renal-carcinoma)。

| dataset_id | 核验结论 |
|---|---|
| `2024_nm_human_lymph_nodes_A1` | 来源的 `final_annot` 表示解剖/空间区域，不作为逐细胞 cell type。 |
| `2024_nm_human_lymph_nodes_D1` | 未发现完整、逐 observation 对齐的公开 cell type 字段。 |
| `GSE205055_human_hippocampus_50um_atac_rna` | 公开逐 observation 元数据仅提供 RNA/ATAC 聚类；论文中的 label transfer 展示不足以构成可对齐来源标签。 |
| `GSE205055_mouse_brain_p21_20um_atac_rna` | 同上。 |
| `GSE205055_mouse_brain_p21_20um_atac_rna_rep2` | 同上。 |
| `GSE205055_mouse_brain_p21_20um_h3k27ac_rna` | 同上。 |
| `GSE205055_mouse_brain_p21_20um_h3k27ac_rna_rep2` | 同上。 |
| `GSE205055_mouse_brain_p22_20um_atac_rna` | 同上。 |
| `GSE205055_mouse_brain_p22_20um_h3k27ac_rna` | 同上。 |
| `GSE205055_mouse_brain_p22_20um_h3k27me3_rna` | 同上。 |
| `GSE205055_mouse_brain_p22_20um_h3k4me3_rna` | 同上。 |
| `GSE205055_mouse_embryo_e13_25um_atac_rna` | 同上。 |
| `GSE205055_mouse_embryo_e13_50um_atac_rna` | 同上。 |
| `GSE213264_human_gbm_spatial_citeseq` | GEO 公开内容为矩阵/空间文件，未发现完整逐 observation 细胞类型注释。 |
| `GSE213264_human_skin_spatial_citeseq` | 同上。 |
| `GSE213264_human_spleen_spatial_citeseq` | 同上。 |
| `GSE213264_human_thymus_spatial_citeseq` | 同上。 |
| `GSE213264_human_tonsil_spatial_citeseq` | 同上。 |
| `GSE213264_mouse_colon_spatial_citeseq` | 同上。 |
| `GSE213264_mouse_intestine_spatial_citeseq` | 同上。 |
| `GSE213264_mouse_kidney_spatial_citeseq` | 同上。 |
| `GSE213264_mouse_spleen_spatial_citeseq` | 同上。 |
| `MISAR_seq_mouse_brain_E13_5_S1` | Zenodo 7480069 未提供可与该入库样本完整对齐的细胞类型文件。 |
| `MISAR_seq_mouse_brain_E15_5_S1` | Zenodo 文件清单仅见 counts、barcode、position 与脚本，未见逐 observation cell type。 |
| `MISAR_seq_mouse_brain_E18_5_S1` | Zenodo 7480069 未提供可与该入库样本完整对齐的细胞类型文件。 |
| `gse198353_spots_mouse_spleen_rep1` | GEO 公开内容为表达/蛋白矩阵和空间信息，未发现完整逐 observation 标签。 |
| `gse198353_spots_mouse_spleen_rep2` | 同上。 |
| `visium_human_tonsil_rna_protein` | 10x Genomics 来源未提供可直接对齐的人工整理 cell type。 |
| `xenium_human_rcc_ffpe_rna_protein` | 10x Genomics 来源未提供可直接对齐的人工整理 cell type。 |
| `zenodo_10362607_stereo_cite_seq_mouse_thymus_19adt_sample_01` | 教程中的人工聚类命名为混合细胞类型的空间区域，不作为逐 observation cell type。 |
| `zenodo_10362607_stereo_cite_seq_mouse_thymus_19adt_sample_02` | 同上。 |
| `zenodo_10362607_stereo_cite_seq_mouse_thymus_19adt_sample_03` | 同上。 |

## 与推断可视化 sidecar 的关系

本记录只判断公开来源是否提供可写入正式 `mdata.obs["cell_type"]` 的完整标签，不判断离线
计算注释是否可行。2026-08-18 完成的 cell type 空间可视化为全部 35 个 Database 建立了
独立 sidecar：上述 3 个确认包含来源标签的数据集直接使用原始标签，其余 32 个分别使用
SingleR 或 RCTD `full` 推断。

推断 sidecar 不改变本表的“未发现来源标签”结论，也不把计算结果写回 `.h5mu`、catalogue
字段或公共 JSON。页面中的 `Computationally inferred`、confidence、`Mixed` 和 `Uncertain`
必须与来源 `cell_type` 明确区分。最终方法、QC 和 provenance 见
[实施与完成记录](plan.md)；科学迭代经验见
[细胞类型注释经验总结](annotation/细胞类型注释经验总结.md)。

## 入库规则

统一字段为顶层 `mdata.obs["cell_type"]`，且为可选项。存在时必须是完整覆盖的无序 pandas
categorical，category 为非空、无首尾空白的字符串且没有未使用类别；保留来源语义和拼写。
空间切分按 observation 传播，组合切分仅在所有来源均有完整有效标签时保留，否则省略该列。
canonical `cell_type` 不增加 catalogue 列、网页/API 字段或筛选项；Database 页面可另行读取
经严格校验的独立 sidecar，但该展示不改变来源标签的入库规则。
