  ## 目标总结

  在 Database 详情页加入英文的 cell type 空间可视化：

  - 覆盖所有具备空间坐标的 full Database；
  - 按 cell type 着色，支持缩放、平移、悬停、图例筛选；
  - cell/bin/spot 均支持，bin/spot 展示预测主导类型；
  - 来源标签直接读取正式 mdata.obs["cell_type"]；
  - 推断标签、置信度和报告独立保存，绝不写回 .h5mu；
  - 先用 4 个代表集完成框架验证，再分批处理完整 35 个数据集；
  - 无法通过质量门槛的数据集保持 Visualization unavailable，并保留失败报告。

  ## 当前项目状态

  只读检查结果：

  - 网站采用 FastAPI、Jinja、Bootstrap 和少量 CSS，没有前端构建流程或空间可视化接口。
  - 现有 Database 详情页和 JSON API 均不读取大型 .h5mu。
  - 当前有 35 个 full Database，均为二维空间坐标，共 767,523 个 observation：
      - 4 个 cell；
      - 26 个 bin；
      - 5 个 spot。

  - 最大数据集 xenium_human_rcc_ffpe_rna_protein 有 465,534 个点。
  - 3 个 Database 有经来源核验的正式 cell type，另外 32 个需要计算推断。
  - 当前所有待推断数据集的 RNA 模态均保存 counts，可支持参考映射或去卷积。
  - Git 工作区当前干净。

  ## 总体架构

  .h5mu + 公共参考数据
            │
            ▼
  独立 Conda 注释环境（离线、可使用 A100）
            │
            ▼
  严格校验的忽略目录产物
    manifest + binary points + report
            │
            ▼
  FastAPI 启动时建立只读索引
            │
            ▼
  页面内部专用数据端点
            │
            ▼
  浏览器中的 deck.gl / WebGL 2

  网站运行时只使用现有 Python 服务；Node 只用于开发和发布前构建静态资源，注释模型只离线运行。

  ## 主要实施步骤

  ### 1. 定义产物与目录契约

  建议默认使用：

  data/cell_type_visualizations/
  ├── references/                 # 下载的参考数据和模型，忽略
  ├── results/<dataset_id>/       # 点位、manifest、报告，忽略
  └── work/                       # 中间结果，忽略

  版本追踪以下内容：

  - 独立注释环境声明文件；
  - 注释脚本和每个数据集的配置；
  - 版本化宽泛 cell type 词表；
  - 二进制格式和报告 schema；
  - 前端源代码、依赖锁文件及生产静态 bundle；
  - 文档和测试。

  每个成功产物记录：

  - dataset ID、.h5mu SHA-256、observation 数量及顺序摘要；
  - sample IDs、坐标范围和显示方向；
  - 来源标签或计算推断状态；
  - Cell Ontology 名称和 ID；
  - 软件及版本、参考数据 ID/版本/校验和；
  - 参数、置信度定义、Mixed/Uncertain 阈值；
  - QC 指标、警告和生成时间；
  - 点位文件大小及 SHA-256。

  Mixed 和 Uncertain 是预测状态，不伪装成 Cell Ontology 类型。

  ### 2. 建立独立注释环境与离线命令

  新建 iscdc-cell-annotation Conda 环境，固定 CUDA/PyTorch、MuData、Scanpy、scvi-tools、CellTypist、cell2location 等兼容版本。

  生成命令必须：

  - 只读正式 .h5mu；
  - 默认拒绝覆盖已有产物，显式 --force 才允许替换；
  - 临时生成、完整校验后原子替换；
  - 支持单数据集、配置驱动和批量审计；
  - 失败时保留诊断报告，不留下可被网站误发布的半成品；
  - 不依赖第三方云端或商业 API。

  参考数据优先从版本化、带本体注释的 CZ CELLxGENE Census 中筛选，再补充论文或来源项目的组织匹配参考；Census 本身提供版本化发布、dataset version ID 和 ontology
  字段，适合作为可追溯参考入口。CELLxGENE Census schema (cellxgene-census/docs/cellxgene_census_schema.md at main · chanzuckerberg/cellxgene-census)

  ### 3. 建立版本化宽泛词表

  预测标签映射到 Cell Ontology，并保存稳定 CL ID。Cell Ontology 是跨物种动物 cell type 的结构化受控词表，适合建立可扩展的宽泛标签层级。OBO Cell Ontology
  (https://obofoundry.org/ontology/cl.html)

  词表采用：

  - 全局复用的宽泛核心类型；
  - 允许新增有正式 CL ID 的组织特异宽泛类型；
  - 每次变更提升词表版本；
  - 细分类证据不足时回退到父级；
  - 来源提供的正式标签保持原始语义和拼写，不被预测词表覆盖。

  ### 4. 实现按数据类型选择方法的注释流程

  不强制所有数据集使用同一方法。

  - 来源标签：直接生成可视化产物，不生成虚假置信度。
  - cell 数据：
      - 以组织/物种匹配参考的 scANVI/scArches 映射为主；
      - scANVI 支持返回逐类别软预测概率，可用于后续校准。scANVI API (https://docs.scvi-tools.org/en/1.4.1/api/reference/scvi.model.SCANVI.html)
      - CellTypist 作为独立交叉检查或免疫类型辅助，不把未经校准的概率直接当作最终置信度；其官方说明也提示概率并非对所有查询数据都能形成有意义范围。CellTypist
        (https://github.com/Teichlab/celltypist)

  - bin/spot 数据：
      - 使用匹配 scRNA-seq 参考进行去卷积；
      - 首选评估 cell2location，因为它提供 cell abundance 的 posterior mean、标准差和分位数，可支持主导类型及不确定性判断。cell2location
        (https://cell2location.readthedocs.io/en/latest/cell2location.html)

      - 通过参考数据构造 pseudo-bin/pseudo-spot 校准方法和阈值；
      - 若 cell2location 的先验或拟合质量不足，再比较其他完全本地的开源方法。

  - 多模态：
      - RNA 作为主要标签证据；
      - protein、ATAC 或 histone 用于独立 marker/模态一致性检查；
      - 只有匹配参考充分且验证表现更好时才采用 MULTIVI、totalVI 等联合模型。

  置信度按方法和数据集分别校准：

  - cell：校准后的类别 posterior；
  - bin/spot：主导类型 posterior、可信区间、占比差距和组成熵；
  - 高混合度标记 Mixed；
  - OOD、拟合差或不确定性过高标记 Uncertain。

  ### 5. 建立质量门槛

  每个数据集至少检查：

  - 点数、sample、坐标与源 observation 完整对齐；
  - 源 .h5mu SHA-256 一致；
  - 标签全部属于当前词表或允许的状态；
  - 置信度有限且位于 [0,1]；
  - 参考与目标物种、组织、发育阶段及疾病背景适配；
  - 共享基因和 marker 覆盖率；
  - 参考留出集的 balanced accuracy、macro-F1 和校准误差；
  - marker enrichment、独立方法一致性；
  - protein/ATAC/histone 的跨模态一致性；
  - 空间分布合理性、异常碎片和单一类别塌缩；
  - Mixed、Uncertain 比例及类别分布。

  阈值写入各数据集配置，不设不科学的全局统一阈值。任何硬性门槛失败时不发布点位产物。

  ### 6. 实现网站启动时的严格、失败开放加载

  新增独立可视化快照加载模块，沿用 Challenge Difficulty 的模式：

  - 应用启动时扫描产物 manifest；
  - 校验版本、dataset type、ID、SHA-256、sample、点数和点位文件校验和；
  - 仅加载轻量摘要和文件索引，不把全部点位载入服务器内存；
  - 单个产物缺失、损坏或过期时只使该 Database unavailable；
  - 不影响详情页、现有 JSON API、下载和其他 Database；
  - 数据导入或替换后，旧产物因 SHA-256 不一致自动失效；
  - 重新生成产物后需重启应用。

  不增加 catalogue 列，不修改现有公共 JSON API。

  ### 7. 实现紧凑数据端点和 WebGL 页面

  点位按 sample 单独编码：

  - x/y：Float32；
  - cell type：Uint16 编码；
  - confidence：推断标签使用 Float32，来源标签省略；
  - 图例和分段偏移保存在小型 manifest；
  - gzip/Brotli 预压缩；
  - 最大预测数据集预计原始约 6.22 MiB，网络约 3–5 MiB。

  页面内部端点使用受控 dataset/sample 查找，不接受文件路径，也不进入公共 OpenAPI/Database JSON。

  前端采用最小 Node + esbuild + 本地 deck.gl bundle。deck.gl 原生支持二进制属性、GPU picking、缩放和平移；其官方性能说明给出的基准是基础 ScatterplotLayer
  在较旧笔记本上约一百万点仍可流畅操作，因此 46 万点是合理目标，但仍需在本项目数据上实测。deck.gl 性能与二进制数据 (https://deck.gl/docs/developer-guide/performance)、交互与 picking
  (https://deck.gl/docs/developer-guide/interactivity)

  UI 放在 Database 标题/简介之后、文件元数据之前，使用全宽布局：

  - 左侧/主体：等比例空间画布；
  - 右侧：可滚动图例；
  - 图例显示类型、数量、占比和颜色；
  - 支持 Select all、Clear all、逐类开关；
  - hover 只显示 cell type；推断标签额外显示 confidence；
  - sample selector 一次只显示一个独立坐标系；
  - 来源标签显示 Source annotation，不显示数值置信度；
  - 推断结果显示明确的 Computationally inferred；
  - 方法摘要卡片和 ? modal 复用 Difficulty UI 语言；
  - modal 展示参考、版本、参数、置信度、阈值、QC 和限制；
  - 默认 x 向右、y 向上；有明确来源依据时按配置翻转；
  - 不叠加 H&E 或其他背景图。

  接近可视区域时自动加载。WebGL 2 不可用时只显示英文兼容性提示，页面其他部分继续工作。WebGL 2 已在主要现代浏览器中广泛支持，但仍依赖浏览器硬件加速状态。MDN WebGL 2
  (https://developer.mozilla.org/en-US/docs/Web/API/WebGL2RenderingContext)

  ### 8. 测试与验收

  Python 测试覆盖：

  - manifest 和二进制格式严格解析；
  - 缺失、损坏、过期、校验和不符时 fail open；
  - source/inferred 两类产物；
  - 原子生成、覆盖保护和失败回滚；
  - 不修改 .h5mu；
  - sample 隔离；
  - 专用端点的成功、404、缓存及内容头；
  - 现有公共 JSON API 保持不变；
  - 页面摘要、modal、confidence 展示规则；
  - 所有 ASGI 测试继续使用 httpx.AsyncClient。

  前端测试覆盖：

  - 二进制解码；
  - sample 切换；
  - 图例筛选及计数；
  - hover 文本；
  - 坐标方向和 reset view；
  - lazy loading、retry、WebGL 不可用状态；
  - Chrome、Edge、Firefox、Safari 最新两个主版本的发布前 smoke test。

  最大数据集验收：

  - 稳定宽带下 10 秒内可交互；
  - 缩放和平移约 30 FPS；
  - 图例筛选 1 秒内生效；
  - hover 100 ms 内反馈；
  - 完整视野、等比例、方向正确，无异常裁切。

  ## 分批路线图

  首批已确认：

  1. starmap_plus_ad_13m_disease_rep2：来源标签；
  2. xenium_human_rcc_ffpe_rna_protein：预测 cell + 最大规模；
  3. GSE205055_mouse_brain_p22_20um_atac_rna：预测 bin + RNA/ATAC；
  4. visium_human_tonsil_rna_protein：预测 spot + RNA/protein。

  首批通过后：

  1. 补齐另外两个来源标签 Database；
  2. 按共享参考处理免疫组织 RNA/protein spot 和 bin；
  3. 批量处理 GSE205055 脑/胚胎 RNA+ATAC/histone 系列；
  4. 处理 MISAR、Stereo-CITE-seq 和其余组织；
  5. 对完整 35 个 Database 运行最终审计。

  最终验收是：35 个全部完成注释尝试和审计；通过质量门槛的全部上线，无法通过者以 Visualization unavailable 加完整失败报告验收。

  ## 主要风险

  - 参考数据与目标组织、疾病或测量 panel 不匹配，可能产生系统性误注释。
  - bin/spot 的“主导类型”会压缩真实混合组成，必须明确标为推断。
  - 不同方法的 confidence 不可直接横向比较。
  - Cell Ontology 宽泛映射可能丢失来源细粒度信息。
  - 46 万点的 hover/picking 在低端或软件渲染设备上可能不达标。
  - 前端依赖和构建产物会增加维护面。
  - 当前工具沙箱看不到 GPU 驱动；实际执行时必须在能访问用户确认的 A100 环境中验证 CUDA。
  - 分批期间未处理或失效的数据集会暂时显示 unavailable。

 ## 实现基本要求总结：
  1. 使用 deck.gl + esbuild、本地静态 bundle，生产运行时仅启动 Python。
  2. 使用版本化自定义二进制点位格式和独立忽略目录。
  3. 使用“scANVI/CellTypist 参考映射 + cell2location/备选去卷积”的按数据集选择策略。
  4. 使用 Cell Ontology 的版本化宽泛词表。
  5. 采用严格质量门槛，允许有完整失败报告的 unavailable 结果。
  6. 按上述 4 个代表集启动，再扩展到完整 35 个数据集。
