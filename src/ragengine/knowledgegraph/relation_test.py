from FlagEmbedding import BGEM3FlagModel

from knowledgegraph.relation_align import RelationAlign

emdedding_model = BGEM3FlagModel(
    model_name_or_path=r".\bge_m3",
    normalize_embeddings=True,
    use_fp16=True,
)

aligner = RelationAlign(
    embedding_model=emdedding_model,
    threshold=0.55,  # 建议0.55
    relation_direction=r".\test",
)

DEFAULT_EXEMPLAR_BANK: dict[str, list[str]] = {
    # 动态。加剧
    "INDUCES_DEFECT": [
        "causes molding defects",
        "induces defect formation",
        "leads to severe defects",
        "aggravates product defects",
        "results in flashing or void defects",
        "increases defect occurrence rate",
    ],
    # 动态。抑制
    "ELIMINATES_DEFECT": [
        "eliminates molding defects",
        "suppresses defect formation",
        "mitigates defect severity",
        "reduces product defects",
        "prevents defects effectively",
        "solves blushing and flashing issues",
    ],
    # 动态。提升
    "ADJUST_INCREASE": [
        "increase process parameter",
        "raise machine setting",
        "higher parameter value",
        "elevate mold temperature or pressure",
        "extend processing time",
    ],
    # 动态。降低
    "ADJUST_DECREASE": [
        "decrease process parameter",
        "lower machine setting",
        "reduce parameter value",
        "lower injection velocity or pressure",
        "shorten processing time",
    ],
    # 物理机理归因
    "PHYSICAL_MECHANISM": [
        "physical mechanism attribution",
        "attributed to underlying physics",
        "root cause mechanism",
        "microstructural orientation and crystallization",
    ],
    # 外观特征表现
    "MANIFESTS_AS": [
        "manifests as visible defect",
        "characterized by morphological features",
        "defect appearance pattern",
        "macroscopic visual observation",
    ],
    # 参数状态关联
    "PARAM_CORRELATION": [
        "parameter correlation and interaction",
        "process parameters affect each other",
        "directly or inversely proportional relationship",
    ],
    # 材料适用
    "APPLIED_TO_MATERIAL": [
        "applicable to polymer material",
        "applied to specific plastic grade",
        "targeted for thermoplastic resin",
    ],
    # 构件从属 (Part-Of)
    "PART_OF": [
        "part of mold or machine assembly",
        "belongs to injection system component",
        "installed inside mold cavity",
    ],
}

# aligner.canonical_relation_embed(DEFAULT_EXEMPLAR_BANK)
aligner.index_read()


TEST_RELATIONS: list[tuple[str, str]] = [
    # ================= 1. 标准关系直接命中 (Canonical Hits, 9条) =================
    ("INDUCES_DEFECT", "INDUCES_DEFECT"),
    ("ELIMINATES_DEFECT", "ELIMINATES_DEFECT"),
    ("ADJUST_INCREASE", "ADJUST_INCREASE"),
    ("ADJUST_DECREASE", "ADJUST_DECREASE"),
    ("PHYSICAL_MECHANISM", "PHYSICAL_MECHANISM"),
    ("MANIFESTS_AS", "MANIFESTS_AS"),
    ("PARAM_CORRELATION", "PARAM_CORRELATION"),
    ("APPLIED_TO_MATERIAL", "APPLIED_TO_MATERIAL"),
    ("PART_OF", "PART_OF"),

    # ================= 2. 英文学术与工业表达 (English Academic & Industrial Mentions, 150条) =================
    # --- INDUCES_DEFECT (25条) ---
    ("causes severe flash", "INDUCES_DEFECT"),
    ("leads to void formation", "INDUCES_DEFECT"),
    ("induces warpage in parts", "INDUCES_DEFECT"),
    ("aggravates silver streaks", "INDUCES_DEFECT"),
    ("results in surface sink marks", "INDUCES_DEFECT"),
    ("provokes weld line formation", "INDUCES_DEFECT"),
    ("triggers flow marks", "INDUCES_DEFECT"),
    ("promotes shrinkage porosity", "INDUCES_DEFECT"),
    ("generates gas entrapment", "INDUCES_DEFECT"),
    ("contributes to part cracking", "INDUCES_DEFECT"),
    ("exacerbates gate blushing", "INDUCES_DEFECT"),
    ("renders molding defective", "INDUCES_DEFECT"),
    ("provokes short shot defect", "INDUCES_DEFECT"),
    ("induces blistering on surface", "INDUCES_DEFECT"),
    ("increases cold shut occurrence", "INDUCES_DEFECT"),
    ("tends to cause flashing", "INDUCES_DEFECT"),
    ("promotes air traps in cavity", "INDUCES_DEFECT"),
    ("aggravates thermal degradation", "INDUCES_DEFECT"),
    ("initiates micro-cracks", "INDUCES_DEFECT"),
    ("accelerates defect propagation", "INDUCES_DEFECT"),
    ("induces residual stress concentration", "INDUCES_DEFECT"),
    ("causes severe burn marks", "INDUCES_DEFECT"),
    ("leads to excessive deformation", "INDUCES_DEFECT"),
    ("results in incomplete filling", "INDUCES_DEFECT"),
    ("triggers surface waviness", "INDUCES_DEFECT"),

    # --- ELIMINATES_DEFECT (25条) ---
    ("prevents void defects", "ELIMINATES_DEFECT"),
    ("mitigates part warpage", "ELIMINATES_DEFECT"),
    ("suppresses silver streaks", "ELIMINATES_DEFECT"),
    ("eliminates flashing along parting line", "ELIMINATES_DEFECT"),
    ("reduces shrinkage marks", "ELIMINATES_DEFECT"),
    ("avoids gate blushing", "ELIMINATES_DEFECT"),
    ("resolves short shot issue", "ELIMINATES_DEFECT"),
    ("cures surface blistering", "ELIMINATES_DEFECT"),
    ("minimizes weld line visibility", "ELIMINATES_DEFECT"),
    ("inhibits porosity formation", "ELIMINATES_DEFECT"),
    ("remedies cold shut flaws", "ELIMINATES_DEFECT"),
    ("alleviates part deformation", "ELIMINATES_DEFECT"),
    ("eradicates mold flashing", "ELIMINATES_DEFECT"),
    ("counteracts thermal degradation", "ELIMINATES_DEFECT"),
    ("prevents micro-crack initiation", "ELIMINATES_DEFECT"),
    ("relieves residual stress", "ELIMINATES_DEFECT"),
    ("improves surface defect quality", "ELIMINATES_DEFECT"),
    ("overcomes gas trap problem", "ELIMINATES_DEFECT"),
    ("effectively eliminates burn marks", "ELIMINATES_DEFECT"),
    ("avoids sink mark defects", "ELIMINATES_DEFECT"),
    ("suppresses cavity air entrapment", "ELIMINATES_DEFECT"),
    ("fixes incomplete filling problem", "ELIMINATES_DEFECT"),
    ("dramatically reduces defects", "ELIMINATES_DEFECT"),
    ("prevents flashing defect", "ELIMINATES_DEFECT"),
    ("solves warpage distortion", "ELIMINATES_DEFECT"),

    # --- ADJUST_INCREASE (20条) ---
    ("raise the mold temperature", "ADJUST_INCREASE"),
    ("increase injection speed", "ADJUST_INCREASE"),
    ("elevate packing pressure", "ADJUST_INCREASE"),
    ("extend cooling duration", "ADJUST_INCREASE"),
    ("boost back pressure setting", "ADJUST_INCREASE"),
    ("increase screw rotation speed", "ADJUST_INCREASE"),
    ("enhance intensification pressure", "ADJUST_INCREASE"),
    ("raise holding pressure", "ADJUST_INCREASE"),
    ("prolong holding time", "ADJUST_INCREASE"),
    ("amplify injection velocity", "ADJUST_INCREASE"),
    ("elevate barrel melt temperature", "ADJUST_INCREASE"),
    ("step up pouring temperature", "ADJUST_INCREASE"),
    ("increase fast shot velocity", "ADJUST_INCREASE"),
    ("maximize clamping force", "ADJUST_INCREASE"),
    ("raise melt temp", "ADJUST_INCREASE"),
    ("increase cavity vacuum level", "ADJUST_INCREASE"),
    ("prolong plunger stroke", "ADJUST_INCREASE"),
    ("elevate nozzle temperature", "ADJUST_INCREASE"),
    ("increase pressure rise speed", "ADJUST_INCREASE"),
    ("heighten injection pressure", "ADJUST_INCREASE"),

    # --- ADJUST_DECREASE (20条) ---
    ("lower injection velocity", "ADJUST_DECREASE"),
    ("reduce melt temperature", "ADJUST_DECREASE"),
    ("decrease packing pressure", "ADJUST_DECREASE"),
    ("shorten cycle time", "ADJUST_DECREASE"),
    ("drop back pressure", "ADJUST_DECREASE"),
    ("curtail cooling duration", "ADJUST_DECREASE"),
    ("lower screw rpm", "ADJUST_DECREASE"),
    ("diminish injection pressure", "ADJUST_DECREASE"),
    ("shorten pressure rise time", "ADJUST_DECREASE"),
    ("decrease plunger stroke distance", "ADJUST_DECREASE"),
    ("lessen slow shot velocity", "ADJUST_DECREASE"),
    ("reduce holding duration", "ADJUST_DECREASE"),
    ("minimize mold temperature", "ADJUST_DECREASE"),
    ("cut down injection rate", "ADJUST_DECREASE"),
    ("lower barrel heating zone temp", "ADJUST_DECREASE"),
    ("decrease intensification pressure", "ADJUST_DECREASE"),
    ("curtail packing time", "ADJUST_DECREASE"),
    ("reduce clamping force slightly", "ADJUST_DECREASE"),
    ("lower nozzle temperature", "ADJUST_DECREASE"),
    ("decrease melt delivery speed", "ADJUST_DECREASE"),

    # --- PHYSICAL_MECHANISM (15条) ---
    ("physical crystallization mechanism", "PHYSICAL_MECHANISM"),
    ("attributed to underlying physics", "PHYSICAL_MECHANISM"),
    ("root cause is thermal degradation", "PHYSICAL_MECHANISM"),
    ("due to rapid non-uniform cooling", "PHYSICAL_MECHANISM"),
    ("caused by cavity air entrapment", "PHYSICAL_MECHANISM"),
    ("governed by shear heating", "PHYSICAL_MECHANISM"),
    ("mechanism of secondary dendrite spacing", "PHYSICAL_MECHANISM"),
    ("solid solution strengthening effect", "PHYSICAL_MECHANISM"),
    ("driven by pressure gradient dynamics", "PHYSICAL_MECHANISM"),
    ("attributed to molecular chain orientation", "PHYSICAL_MECHANISM"),
    ("thermodynamic crystallization kinetics", "PHYSICAL_MECHANISM"),
    ("microstructural eutectic silicon growth", "PHYSICAL_MECHANISM"),
    ("mechanism of solid fraction growth", "PHYSICAL_MECHANISM"),
    ("governed by viscoelastic relaxation", "PHYSICAL_MECHANISM"),
    ("attributed to filling temperature drop", "PHYSICAL_MECHANISM"),

    # --- MANIFESTS_AS (15条) ---
    ("manifests as visible blush", "MANIFESTS_AS"),
    ("characterized by silver lines", "MANIFESTS_AS"),
    ("morphological feature shows porous holes", "MANIFESTS_AS"),
    ("exhibits surface depression", "MANIFESTS_AS"),
    ("appearance reveals flash along edge", "MANIFESTS_AS"),
    ("visual symptom of short shot", "MANIFESTS_AS"),
    ("shows V-notch seam pattern", "MANIFESTS_AS"),
    ("microscopic observation displays porosity", "MANIFESTS_AS"),
    ("surface shows wavy ripple pattern", "MANIFESTS_AS"),
    ("manifests as discolored dark spot", "MANIFESTS_AS"),
    ("characterized by dendritic microstructure", "MANIFESTS_AS"),
    ("visible cracks on surface", "MANIFESTS_AS"),
    ("appearance shows blistering bubbles", "MANIFESTS_AS"),
    ("morphology displays cold shut line", "MANIFESTS_AS"),
    ("visual appearance shows distortion", "MANIFESTS_AS"),

    # --- PARAM_CORRELATION (10条) ---
    ("parameters correlate with each other", "PARAM_CORRELATION"),
    ("interacts with melt viscosity", "PARAM_CORRELATION"),
    ("pressure gradient influences velocity", "PARAM_CORRELATION"),
    ("proportional to cooling rate", "PARAM_CORRELATION"),
    ("inversely related to wall thickness", "PARAM_CORRELATION"),
    ("coupled with cavity pressure curve", "PARAM_CORRELATION"),
    ("strongly associated with fill time", "PARAM_CORRELATION"),
    ("interdependent process parameters", "PARAM_CORRELATION"),
    ("proportional to packing time", "PARAM_CORRELATION"),
    ("correlated with melt flow index", "PARAM_CORRELATION"),

    # --- APPLIED_TO_MATERIAL (10条) ---
    ("applicable to polycarbonate", "APPLIED_TO_MATERIAL"),
    ("applied to PA66-GF30 resin", "APPLIED_TO_MATERIAL"),
    ("suitable for AlSi10MnMg alloy", "APPLIED_TO_MATERIAL"),
    ("tailored for PMMA plastic", "APPLIED_TO_MATERIAL"),
    ("used in ABS polymers", "APPLIED_TO_MATERIAL"),
    ("designated for A380 die casting", "APPLIED_TO_MATERIAL"),
    ("adapted for polypropylene material", "APPLIED_TO_MATERIAL"),
    ("optimized for magnesium alloy", "APPLIED_TO_MATERIAL"),
    ("applied to thermoplastic polymers", "APPLIED_TO_MATERIAL"),
    ("suitable for ADC12 aluminum", "APPLIED_TO_MATERIAL"),

    # --- PART_OF (10条) ---
    ("part of mold cavity", "PART_OF"),
    ("component of injection screw", "PART_OF"),
    ("installed on clamping unit", "PART_OF"),
    ("belongs to runner system", "PART_OF"),
    ("integrated into ejector pin assembly", "PART_OF"),
    ("located inside nozzle body", "PART_OF"),
    ("subsystem of hot runner manifold", "PART_OF"),
    ("portion of gating system", "PART_OF"),
    ("belongs to barrel heating zone", "PART_OF"),
    ("component of shot sleeve", "PART_OF"),

    # ================= 3. 中文工艺文献与车间俚语表达 (Chinese Technical & Workshop Jargon, 165条) =================
    # --- INDUCES_DEFECT (25条) ---
    ("导致制品严重开裂", "INDUCES_DEFECT"),
    ("诱发浇口周围发白", "INDUCES_DEFECT"),
    ("容易产生批锋溢料", "INDUCES_DEFECT"),
    ("加剧熔接痕明显化", "INDUCES_DEFECT"),
    ("引起内部缩孔缩松", "INDUCES_DEFECT"),
    ("造成大量气孔缺陷", "INDUCES_DEFECT"),
    ("加重表面银丝银纹", "INDUCES_DEFECT"),
    ("诱致翘曲变形超差", "INDUCES_DEFECT"),
    ("诱发欠注缺料问题", "INDUCES_DEFECT"),
    ("容易出现流痕波纹", "INDUCES_DEFECT"),
    ("催生飞边毛刺", "INDUCES_DEFECT"),
    ("致使内部残余应力集中", "INDUCES_DEFECT"),
    ("极易引发冷隔缺陷", "INDUCES_DEFECT"),
    ("恶化了表观起泡起皮", "INDUCES_DEFECT"),
    ("助长分子链剪切降解", "INDUCES_DEFECT"),
    ("使得缩水痕明显加重", "INDUCES_DEFECT"),
    ("加速模腔卷气产生", "INDUCES_DEFECT"),
    ("导致充型不饱模", "INDUCES_DEFECT"),
    ("引发表面灼伤黑点", "INDUCES_DEFECT"),
    ("诱发局部顶白开裂", "INDUCES_DEFECT"),
    ("造成分型面跑披锋", "INDUCES_DEFECT"),
    ("导致产品尺寸不稳定", "INDUCES_DEFECT"),
    ("促使气穴微裂纹萌生", "INDUCES_DEFECT"),
    ("加剧充填末端缺胶", "INDUCES_DEFECT"),
    ("诱发金相组织粗化", "INDUCES_DEFECT"),

    # --- ELIMINATES_DEFECT (25条) ---
    ("大幅遏制气孔产生", "ELIMINATES_DEFECT"),
    ("有效消除浇口发白", "ELIMINATES_DEFECT"),
    ("显著改善制品缩痕", "ELIMINATES_DEFECT"),
    ("防止溢边披锋产生", "ELIMINATES_DEFECT"),
    ("抑制银丝斑纹生成", "ELIMINATES_DEFECT"),
    ("彻底解决缺料短射", "ELIMINATES_DEFECT"),
    ("消除表面冷隔痕迹", "ELIMINATES_DEFECT"),
    ("缓解塑件翘曲变形", "ELIMINATES_DEFECT"),
    ("根治制品气泡气穴", "ELIMINATES_DEFECT"),
    ("克服熔接线弱化问题", "ELIMINATES_DEFECT"),
    ("有效避免飞边产生", "ELIMINATES_DEFECT"),
    ("改善充填不足短射", "ELIMINATES_DEFECT"),
    ("抑制高速卷气夹渣", "ELIMINATES_DEFECT"),
    ("消除表面流纹缺陷", "ELIMINATES_DEFECT"),
    ("削弱内部残余应力", "ELIMINATES_DEFECT"),
    ("平抑表面波纹缺陷", "ELIMINATES_DEFECT"),
    ("防止产生烧焦黑斑", "ELIMINATES_DEFECT"),
    ("有效抑制制品缩水", "ELIMINATES_DEFECT"),
    ("消除脱模顶出顶白", "ELIMINATES_DEFECT"),
    ("改善外观光泽不均", "ELIMINATES_DEFECT"),
    ("解决熔接痕结合不良", "ELIMINATES_DEFECT"),
    ("显著降低孔隙率", "ELIMINATES_DEFECT"),
    ("根除进料口白晕", "ELIMINATES_DEFECT"),
    ("预防注塑溢料问题", "ELIMINATES_DEFECT"),
    ("消除局部缩陷凹坑", "ELIMINATES_DEFECT"),

    # --- ADJUST_INCREASE (25条) ---
    ("适当调高注射压力", "ADJUST_INCREASE"),
    ("升高模具预热温度", "ADJUST_INCREASE"),
    ("提高保压压力设定", "ADJUST_INCREASE"),
    ("增加螺杆转速", "ADJUST_INCREASE"),
    ("延长保压压实时间", "ADJUST_INCREASE"),
    ("提升熔融料筒温度", "ADJUST_INCREASE"),
    ("加大快压射充填速度", "ADJUST_INCREASE"),
    ("调高增压建压压力", "ADJUST_INCREASE"),
    ("增加背压塑化阻力", "ADJUST_INCREASE"),
    ("提升射胶充填速度", "ADJUST_INCREASE"),
    ("延长冷却定型时间", "ADJUST_INCREASE"),
    ("提高金属液浇注温度", "ADJUST_INCREASE"),
    ("上调机筒加热区设定值", "ADJUST_INCREASE"),
    ("加长真空抽气时间", "ADJUST_INCREASE"),
    ("调大锁模力吨位", "ADJUST_INCREASE"),
    ("提升二级射胶压力", "ADJUST_INCREASE"),
    ("增加模具动定模温度", "ADJUST_INCREASE"),
    ("提升压射冲头行程", "ADJUST_INCREASE"),
    ("升高射嘴前端温度", "ADJUST_INCREASE"),
    ("上调保压补缩压力", "ADJUST_INCREASE"),
    ("提升溶体注射速度", "ADJUST_INCREASE"),
    ("增大背压压力", "ADJUST_INCREASE"),
    ("适当延长成型周期", "ADJUST_INCREASE"),
    ("提高慢压射过渡速度", "ADJUST_INCREASE"),
    ("增加模温水机循环流量", "ADJUST_INCREASE"),

    # --- ADJUST_DECREASE (25条) ---
    ("降低螺杆转速", "ADJUST_DECREASE"),
    ("适当调低保压压力", "ADJUST_DECREASE"),
    ("下调料筒熔融温度", "ADJUST_DECREASE"),
    ("减小注射充填速度", "ADJUST_DECREASE"),
    ("缩短保压切换时间", "ADJUST_DECREASE"),
    ("调低模具恒温水温", "ADJUST_DECREASE"),
    ("减小背压阻力设定", "ADJUST_DECREASE"),
    ("降低慢压射启动速度", "ADJUST_DECREASE"),
    ("缩短整机注塑周期", "ADJUST_DECREASE"),
    ("下调注塑机锁模力", "ADJUST_DECREASE"),
    ("减少熔体充填注射量", "ADJUST_DECREASE"),
    ("调低料温避免过热分解", "ADJUST_DECREASE"),
    ("减小末端保压阶段压力", "ADJUST_DECREASE"),
    ("下调快压射速度阈值", "ADJUST_DECREASE"),
    ("缩短制品冷却时间", "ADJUST_DECREASE"),
    ("降低浇口充模流速", "ADJUST_DECREASE"),
    ("减小增压缸峰值压力", "ADJUST_DECREASE"),
    ("下调射嘴加热圈温度", "ADJUST_DECREASE"),
    ("缩短高压建压升压时间", "ADJUST_DECREASE"),
    ("减少冲头压射行程", "ADJUST_DECREASE"),
    ("调低二级注射压力", "ADJUST_DECREASE"),
    ("减小溶胶背压值", "ADJUST_DECREASE"),
    ("降低金属液浇铸温度", "ADJUST_DECREASE"),
    ("下调模腔抽气时间", "ADJUST_DECREASE"),
    ("适度减小注射进胶量", "ADJUST_DECREASE"),

    # --- PHYSICAL_MECHANISM (15条) ---
    ("微观物理机理在于剪切热降解", "PHYSICAL_MECHANISM"),
    ("物理归因为高速剪切分子取向", "PHYSICAL_MECHANISM"),
    ("本质机理是型腔内紊流卷气", "PHYSICAL_MECHANISM"),
    ("由冷却收缩不均匀引起变形", "PHYSICAL_MECHANISM"),
    ("机理解析为共晶硅相粗化", "PHYSICAL_MECHANISM"),
    ("物理归因于二次枝晶间距过大", "PHYSICAL_MECHANISM"),
    ("内在机理在于固溶强化相析出", "PHYSICAL_MECHANISM"),
    ("微观结晶动力学差异所致", "PHYSICAL_MECHANISM"),
    ("归因于熔体流动前锋温度暴跌", "PHYSICAL_MECHANISM"),
    ("热力学固相率增长机制", "PHYSICAL_MECHANISM"),
    ("黏弹性高分子松弛机理", "PHYSICAL_MECHANISM"),
    ("气液两相界面失稳破裂机制", "PHYSICAL_MECHANISM"),
    ("分子热分解链断裂归因", "PHYSICAL_MECHANISM"),
    ("物理本质为型腔排气不畅憋气", "PHYSICAL_MECHANISM"),
    ("取向应力与热应力叠加机理", "PHYSICAL_MECHANISM"),

    # --- MANIFESTS_AS (15条) ---
    ("外观表现为进料口泛白晕", "MANIFESTS_AS"),
    ("特征形态呈现为微细银丝线条", "MANIFESTS_AS"),
    ("宏观表现为表面局部凹陷陷斑", "MANIFESTS_AS"),
    ("肉眼可见分型面毛刺溢边", "MANIFESTS_AS"),
    ("显微金相呈现弥散状气孔分布", "MANIFESTS_AS"),
    ("断面形貌显示脆性解理裂纹", "MANIFESTS_AS"),
    ("外观呈现暗淡冷接缝接痕", "MANIFESTS_AS"),
    ("宏观形态呈现局部缺肉欠注", "MANIFESTS_AS"),
    ("表面特征为水波纹样起伏", "MANIFESTS_AS"),
    ("外观表现为局部烧焦黑斑", "MANIFESTS_AS"),
    ("显微组织表现为粗大树枝晶", "MANIFESTS_AS"),
    ("宏观特征为端部翘曲上翘", "MANIFESTS_AS"),
    ("表面特征为密集微小鼓包起泡", "MANIFESTS_AS"),
    ("肉眼观测为浇口周围雾状晕影", "MANIFESTS_AS"),
    ("宏观形貌显示明显熔接痕夹线", "MANIFESTS_AS"),

    # --- PARAM_CORRELATION (12条) ---
    ("工艺参数之间密切相关", "PARAM_CORRELATION"),
    ("型腔压力与注射速度正比关联", "PARAM_CORRELATION"),
    ("熔体粘度反比于剪切速率", "PARAM_CORRELATION"),
    ("模具温度与结晶度直接耦合", "PARAM_CORRELATION"),
    ("冷却时间与壁厚呈平方正比关系", "PARAM_CORRELATION"),
    ("各段机筒温控相互影响波动", "PARAM_CORRELATION"),
    ("充填时间与流动阻力相互制约", "PARAM_CORRELATION"),
    ("背压高低直接影响塑化均匀度", "PARAM_CORRELATION"),
    ("保压压力与收缩率负相关", "PARAM_CORRELATION"),
    ("慢压射行程与浇道填充率直接联动", "PARAM_CORRELATION"),
    ("真空度高低直接制约气孔率", "PARAM_CORRELATION"),
    ("锁模力大小直接影响飞边厚度", "PARAM_CORRELATION"),

    # --- APPLIED_TO_MATERIAL (13条) ---
    ("适用于聚碳酸酯材料", "APPLIED_TO_MATERIAL"),
    ("针对PA66玻纤增强复合材料", "APPLIED_TO_MATERIAL"),
    ("应用于AlSi10MnMg压铸铝合金", "APPLIED_TO_MATERIAL"),
    ("适合PMMA高透光亚克力树脂", "APPLIED_TO_MATERIAL"),
    ("专用于ABS改性工程塑料", "APPLIED_TO_MATERIAL"),
    ("适用于A380高强度压铸件", "APPLIED_TO_MATERIAL"),
    ("针对PP均聚物热塑材质", "APPLIED_TO_MATERIAL"),
    ("应用于AZ91D镁合金薄壁压铸", "APPLIED_TO_MATERIAL"),
    ("适合ADC12合金压铸配方", "APPLIED_TO_MATERIAL"),
    ("针对通用热塑性工程塑料体系", "APPLIED_TO_MATERIAL"),
    ("专用于改性高流动性合金", "APPLIED_TO_MATERIAL"),
    ("适用于高阻燃增强复合材料", "APPLIED_TO_MATERIAL"),
    ("应用于轻合金近净成型领域", "APPLIED_TO_MATERIAL"),

    # --- PART_OF (10条) ---
    ("属于模具成型模腔零件", "PART_OF"),
    ("安装在注塑机注射螺杆前端", "PART_OF"),
    ("从属于模具分流道与浇口构件", "PART_OF"),
    ("属于顶出脱模机构组成部分", "PART_OF"),
    ("装配于机筒加热圈组件", "PART_OF"),
    ("从属于压铸机压射冲头总成", "PART_OF"),
    ("位于模具热流道热咀内部", "PART_OF"),
    ("属于模具冷却水路镶件构件", "PART_OF"),
    ("装配在料筒注射喷嘴内", "PART_OF"),
    ("从属于合模机构液压系统", "PART_OF"),

    # ================= 4. 域外负例干扰词 (Out-of-Domain Negative Samples, 55条) =================
    ("今天天气真好", "今天天气真好"),
    ("编译器优化级别", "编译器优化级别"),
    ("深度学习框架", "深度学习框架"),
    ("数据库复合索引", "数据库复合索引"),
    ("煮咖啡机开关", "煮咖啡机开关"),
    ("机械键盘红轴", "机械键盘红轴"),
    ("蓝牙无线连接", "蓝牙无线连接"),
    ("金融股票走势", "金融股票走势"),
    ("移动端界面开发", "移动端界面开发"),
    ("量子纠缠态", "量子纠缠态"),
    ("卷积神经网络", "卷积神经网络"),
    ("微服务网关路由", "微服务网关路由"),
    ("区块链智能合约", "区块链智能合约"),
    ("无人机飞控代码", "无人机飞控代码"),
    ("太阳能电池板", "太阳能电池板"),
    ("咖啡机蒸汽阀门", "咖啡机蒸汽阀门"),
    ("木质书架组装", "木质书架组装"),
    ("陶瓷咖啡杯", "陶瓷咖啡杯"),
    ("电动滑板车电池", "电动滑板车电池"),
    ("操作系统虚拟内存", "操作系统虚拟内存"),
    ("网络路由协议", "网络路由协议"),
    ("一次性环保纸杯", "一次性环保纸杯"),
    ("游泳池水循环泵", "游泳池水循环泵"),
    ("自然语言处理分词", "自然语言处理分词"),
    ("强化学习价值网络", "强化学习价值网络"),
    ("图形渲染管线着色器", "图形渲染管线着色器"),
    ("自动驾驶激光雷达", "自动驾驶激光雷达"),
    ("密码学非对称加密", "密码学非对称加密"),
    ("高频交易撮合引擎", "高频交易撮合引擎"),
    ("太空望远镜光谱仪", "太空望远镜光谱仪"),
    ("compiler optimization flag", "compiler optimization flag"),
    ("quantum computing qubit", "quantum computing qubit"),
    ("operating system kernel", "operating system kernel"),
    ("database primary key", "database primary key"),
    ("blockchain consensus algorithm", "blockchain consensus algorithm"),
    ("financial stock trading", "financial stock trading"),
    ("coffee espresso machine", "coffee espresso machine"),
    ("bluetooth audio connection", "bluetooth audio connection"),
    ("electric bicycle battery", "electric bicycle battery"),
    ("internet routing table", "internet routing table"),
    ("mechanical keyboard switch", "mechanical keyboard switch"),
    ("swimming pool water pump", "swimming pool water pump"),
    ("deep neural network loss", "deep neural network loss"),
    ("solar panel efficiency", "solar panel efficiency"),
    ("drone flight controller", "drone flight controller"),
    ("smartphone touchscreen display", "smartphone touchscreen display"),
    ("weather forecast model", "weather forecast model"),
    ("distributed file storage", "distributed file storage"),
    ("graphic card memory clock", "graphic card memory clock"),
    ("virtual reality headset", "virtual reality headset"),
    ("audio speech recognition", "audio speech recognition"),
    ("reinforcement learning policy", "reinforcement learning policy"),
    ("cloud computing instance", "cloud computing instance"),
    ("relational database schema", "relational database schema"),
    ("web browser rendering engine", "web browser rendering engine"),
]


total = len(TEST_RELATIONS)
canonical_hits = 0  # 类别1：标准关系输入，直接命中自身
aligned_hits = 0    # 类别2：同义词/短语/跨语言变体输入，成功吸附对齐到正确标准关系
rejection_hits = 0  # 类别3：域外负例干扰词输入，成功被阈值拦截（保持原词未被错误吸附）
failures = []       # 记录所有失败 case: (raw, expected, actual, failure_type)

canonical_set = set(aligner.inverted_canonical_relations.values())

for idx, (raw, expected) in enumerate(TEST_RELATIONS, 1):
    actual = aligner.aligner(raw)

    # 判定分支 A：输入本身就是标准关系名
    if raw in canonical_set:
        if actual == expected:
            canonical_hits += 1
            status = "[标准命中]"
        else:
            failures.append((raw, expected, actual, "标准词漂移"))
            status = "[对齐失败]"
    # 判定分支 B：输入是域外负样本（预期不吸附，保留原词）
    elif raw == expected:
        if actual == expected:
            rejection_hits += 1
            status = "[负例拒识]"
        else:
            failures.append((raw, expected, actual, "误吸附(假阳性)"))
            status = "[对齐失败]"
    # 判定分支 C：输入是别名/短语/跨语言短语（预期吸附到目标标准关系）
    else:
        if actual == expected:
            aligned_hits += 1
            status = "[关系吸附]"
        else:
            failure_type = "未达阈值漏对齐" if actual == raw else "错配到其他关系"
            failures.append((raw, expected, actual, failure_type))
            status = "[对齐失败]"

    if idx <= 15 or idx % 50 == 0 or idx == total:
        print(
            f"[{idx:>3}/{total}] {status} 输入: '{raw}' -> 输出: '{actual}' (预期: '{expected}')"
        )

# 统计总指标
total_success = canonical_hits + aligned_hits + rejection_hits
overall_accuracy = (total_success / total) * 100

print("\n" + "=" * 75)
print("关系对齐准确率评测总览报告 (Relation Alignment Evaluation Report)")
print("=" * 75)
print(f"总测试样本数:   {total}")
print(
    f"  ├─ 标准关系命中 (Canonical Hits):  {canonical_hits:>3} / 9   ({canonical_hits / 9 * 100:.1f}%)"
)
print(
    f"  ├─ 同义/跨语吸附 (Aligned Hits):    {aligned_hits:>3} / 315 ({aligned_hits / 315 * 100:.1f}%)"
)
print(
    f"  ├─ 负例拒识成功 (Rejection Hits): {rejection_hits:>3} / 55  ({rejection_hits / 55 * 100:.1f}%)"
)
print(
    f"  └─ 对齐失败总数 (Failures):        {len(failures):>3} / {total} ({len(failures) / total * 100:.1f}%)"
)
print("-" * 75)
print(f"全局综合准确率 (Overall Accuracy):  {overall_accuracy:.2f}%")
print("=" * 75)

if failures:
    print(f"\n失败用例明细清单 (共 {len(failures)} 条):")
    print(
        f"{'序号':<4} | {'失败类型':<14} | {'输入原始词':<30} | {'预期标准关系':<22} | {'实际输出'}"
    )
    print("-" * 90)
    for i, (r, exp, act, ftype) in enumerate(failures, 1):
        print(f"{i:<4} | {ftype:<14} | {r:<30} | {exp:<22} | {act}")

