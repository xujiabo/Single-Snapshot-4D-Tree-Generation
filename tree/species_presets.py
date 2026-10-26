# species_presets.py
"""不同物种的参数预设。你可以查文献调，或者按形态特征拍脑袋调。"""

SPECIES_PRESETS = {
    # "oak": {
    #     "MAX_AGE": 25,
    #     "INTERNODE_LENGTH": 0.35,
    #     "INTERNODE_LENGTH_DECAY": 0.92,
    #     "TRUNK_LENGTH_BOOST": 1.0,         # 主干不要突出
    #     "BRANCH_ANGLE": 78,                 # 接近水平 ← 关键
    #     "BRANCH_ANGLE_VAR": 12,
    #     "APICAL_DOMINANCE": 0.92,           # 侧枝活得久 ← 关键
    #     "LATERAL_PROB": 0.85,
    #     "MAX_ORDER": 5,                     # 多层分叉
    #     "INITIAL_RADIUS": 0.08,
    #     "RADIUS_DECAY": 0.65,
    #     "BASE_SIZE": 3,                     # 早早开始分叉
    #     "SPAWN_DECAY": 0.55,                # 让多层 spawn 都有戏
    #
    #     "TRUNK_TROPISM_Z": 1.0,
    #     "TRUNK_TROPISM_STRENGTH": 0.03,
    #     "BRANCH_TROPISM_Z": 1.0,            # 略向上抗重力
    #     "BRANCH_TROPISM_STRENGTH": 0.04,    # 弱 tropism，主要靠 BRANCH_ANGLE 控制方向
    #
    #     "WHORL_MODE": 0,
    #     "WHORL_SIZE": 5,
    #     "BRANCH_MAX_LIFE": 30,              # 大上限让侧枝长
    #
    #     "LEAF_RADIUS": 0.005,
    #     "PIPE_EXPONENT": 2.8,               # 粗壮型
    #     "SEGMENT_BEND": 4.0,
    #     "TRUNK_MAX_LIFE": 100,
    #
    # },
    "oak": {
        "MAX_AGE": 22,                       # 25→22
        "INTERNODE_LENGTH": 0.32,
        "INTERNODE_LENGTH_DECAY": 0.88,      # 0.92→0.88，深层枝条短一点
        "TRUNK_LENGTH_BOOST": 1.0,
        "BRANCH_ANGLE": 75,
        "BRANCH_ANGLE_VAR": 12,
        "BRANCH_ANGLE_DECAY": 0.65,          # ★ 新增，关键
        "APICAL_DOMINANCE": 0.88,            # 0.92→0.88，缩短直线段
        "LATERAL_PROB": 0.7,                 # 0.85→0.7，减密度
        "MAX_ORDER": 4,                      # 5→4，去掉末端小钢针
        "INITIAL_RADIUS": 0.08,
        "RADIUS_DECAY": 0.62,
        "BASE_SIZE": 3,
        "SPAWN_DECAY": 0.6,

        "TRUNK_TROPISM_Z": 1.0,
        "TRUNK_TROPISM_STRENGTH": 0.03,
        "BRANCH_TROPISM_Z": 1.0,
        "BRANCH_TROPISM_STRENGTH": 0.10,     # ★ 0.04→0.10，看得见地向上卷

        "WHORL_MODE": 0,
        "WHORL_SIZE": 5,
        "BRANCH_MAX_LIFE": 22,               # 30→22

        "LEAF_RADIUS": 0.005,
        "PIPE_EXPONENT": 2.8,
        "SEGMENT_BEND": 7.0,                 # ★ 4→7，看得见的曲折

        "TRUNK_MAX_LIFE": 8,                 # ★ 强制早分叉
        "BRANCH_MIN_LIFE": 2,                # 顺便压一下短刺
    },
    "pine": {
        "MAX_AGE": 22,                       # ★ 30→22，trunk 不要那么高
        "INTERNODE_LENGTH": 0.40,
        "INTERNODE_LENGTH_DECAY": 0.85,
        "TRUNK_LENGTH_BOOST": 1.4,           # ★ 2.0→1.4，主干没那么夸张
        "BRANCH_ANGLE": 75,
        "BRANCH_ANGLE_VAR": 12,               # 5→8，每根侧枝角度略有差异
        "BRANCH_ANGLE_DECAY": 0.7,           # ★ 新增，让二级枝顺势延伸
        "APICAL_DOMINANCE": 0.94,            # ★ 0.92→0.94，order-1 不要过早终止
        "LATERAL_PROB": 0.95,
        "MAX_ORDER": 3,
        "INITIAL_RADIUS": 0.07,              # 0.06→0.07，trunk 粗一点
        "RADIUS_DECAY": 0.55,
        "BASE_SIZE": 4,                      # 5→4，让冠层早点开始

        "TROPISM_X": 0.0,
        "TROPISM_Y": 0.0,
        "TROPISM_Z": 1.0,
        "TROPISM_STRENGTH": 0.08,

        "LEAF_RADIUS": 0.004,
        "PIPE_EXPONENT": 2.7,

        "WHORL_MODE": 0,
        "WHORL_SIZE": 5,
        "BRANCH_MAX_LIFE": 14,               # ★ 8→14，侧枝长得更远，冠变宽

        "TRUNK_TROPISM_Z": 1.0,
        "TRUNK_TROPISM_STRENGTH": 0.08,      # ★ 0.05→0.08，强化拉直
        "BRANCH_TROPISM_Z": 1.0,             # ★ 0.0→1.0，侧枝末端微微上翘
        "BRANCH_TROPISM_STRENGTH": 0.04,     # ★ 0.0→0.04，配合上

        "SEGMENT_BEND": 1.0,
        "TRUNK_SEGMENT_BEND": 0.5,           # ★ 新增，主干几乎不弯

        "TRUNK_MAX_LIFE": 100,               # 保持，pine 主干本来就该贯穿
    },
    # "birch": {
    #     "MAX_AGE": 30,                       # 之前 22，加大
    #     "INTERNODE_LENGTH": 0.30,
    #     "INTERNODE_LENGTH_DECAY": 0.90,
    #     "TRUNK_LENGTH_BOOST": 1.6,           # 桦树主干特征明显
    #     "BRANCH_ANGLE": 50,
    #     "BRANCH_ANGLE_VAR": 10,
    #     "APICAL_DOMINANCE": 0.85,            # 之前 0.78 略低，调到 0.85
    #     "MAX_ORDER": 4,
    #     "INITIAL_RADIUS": 0.045,
    #     "RADIUS_DECAY": 0.55,
    #     "BASE_SIZE": 6,
    #     "SPAWN_DECAY": 0.45,                 # 散生衰减
    #
    #     "TRUNK_TROPISM_Z": 1.0,
    #     "TRUNK_TROPISM_STRENGTH": 0.03,
    #     "BRANCH_TROPISM_Z": 1.0,
    #     "BRANCH_TROPISM_STRENGTH": 0.06,     # 侧枝轻微上扬
    #
    #     "WHORL_MODE": 0,
    #     "WHORL_SIZE": 5,
    #     "BRANCH_MAX_LIFE": 20,
    #
    #     "LEAF_RADIUS": 0.003,
    #     "PIPE_EXPONENT": 2.3,
    #
    #     "SEGMENT_BEND": 2.0,
    #     "TRUNK_MAX_LIFE": 100,
    #
    #     "TRUNK_SPAWN_END_MARGIN": 6,    # 顶端 6 节干净，不再戳出小刺
    #     "BRANCH_SPAWN_END_MARGIN": 3,   # 侧枝末端 3 节也不分叉
    #
    #     "BRANCH_MIN_LIFE": 4,   # 侧枝至少 4 节才能随机死
    #     "LATERAL_PROB": 0.55,   # 配套降低发枝率，原 0.70 在加保护期后会太密
    # },

    "birch": {
        "MAX_AGE": 30,
        "INTERNODE_LENGTH": 0.30,
        "INTERNODE_LENGTH_DECAY": 0.90,
        "BRANCH_ANGLE_VAR": 10,
        "BRANCH_ANGLE_DECAY": 0.75,          # ★ 新增，治本
        "APICAL_DOMINANCE": 0.85,
        "MAX_ORDER": 4,
        "INITIAL_RADIUS": 0.045,
        "RADIUS_DECAY": 0.55,
        "BASE_SIZE": 6,
        "SPAWN_DECAY": 0.5,                  # 0.45→0.5，深层略密

        "TRUNK_TROPISM_Z": 1.0,
        "BRANCH_TROPISM_Z": 1.0,
        "BRANCH_TROPISM_STRENGTH": 0.03,     # ★ 0.06→0.03，减弱上卷，让冠层横向展开

        "WHORL_MODE": 0,
        "WHORL_SIZE": 5,

        "LEAF_RADIUS": 0.003,
        "PIPE_EXPONENT": 2.3,

        "TRUNK_MAX_LIFE": 22,                # ★ 100→22，鞭梢留得短一些
        "BRANCH_SPAWN_END_MARGIN": 3,

        "BRANCH_MIN_LIFE": 4,
        "LATERAL_PROB": 0.55,

        "TRUNK_LENGTH_BOOST": 1.3,           # 1.4→1.3，主干横向占比上升
        "BRANCH_ANGLE": 62,                  # 60→62
        "BRANCH_MAX_LIFE": 28,               # 28→32，侧枝更长，扩宽
        "TRUNK_SPAWN_END_MARGIN": 5,         # 3→5，鞭梢回长一点

        "SEGMENT_BEND": 3.0,                 # 不变，侧枝继续微弯
        "TRUNK_SEGMENT_BEND": 1.0,           # ★ 新增，主干每段仅 1°，几乎不漂
        "TRUNK_TROPISM_STRENGTH": 0.10,      # 0.15→0.10，配合低扰动够用了
    },

    # "maple": {
    #     "MAX_AGE": 22,
    #     "INTERNODE_LENGTH": 0.3,
    #     "INTERNODE_LENGTH_DECAY": 0.92,
    #     "TRUNK_LENGTH_BOOST": 1.0,
    #     "BRANCH_ANGLE": 65,                 # 略小，让冠更紧凑
    #     "BRANCH_ANGLE_VAR": 10,
    #     "APICAL_DOMINANCE": 0.90,
    #     "LATERAL_PROB": 0.85,
    #     "MAX_ORDER": 5,
    #     "INITIAL_RADIUS": 0.06,
    #     "RADIUS_DECAY": 0.62,
    #     "BASE_SIZE": 4,
    #     "SPAWN_DECAY": 0.5,
    #
    #     "TRUNK_TROPISM_Z": 1.0,
    #     "TRUNK_TROPISM_STRENGTH": 0.03,
    #     "BRANCH_TROPISM_Z": 1.0,
    #     "BRANCH_TROPISM_STRENGTH": 0.05,
    #
    #     "WHORL_MODE": 0,
    #     "WHORL_SIZE": 5,
    #     "BRANCH_MAX_LIFE": 25,
    #
    #     "LEAF_RADIUS": 0.004,
    #     "PIPE_EXPONENT": 2.6,
    #     "SEGMENT_BEND": 3.0,
    #
    #     "TRUNK_MAX_LIFE": 100,
    # },

    "maple": {
        "MAX_AGE": 20,                       # 22→20
        "INTERNODE_LENGTH": 0.28,            # 0.30→0.28，稍短
        "INTERNODE_LENGTH_DECAY": 0.88,      # 0.92→0.88
        "TRUNK_LENGTH_BOOST": 1.0,
        "BRANCH_ANGLE": 62,                  # 65→62
        "BRANCH_ANGLE_VAR": 10,
        "BRANCH_ANGLE_DECAY": 0.7,           # ★ 新增，解决鱼骨
        "APICAL_DOMINANCE": 0.88,            # 0.90→0.88
        "INITIAL_RADIUS": 0.06,
        "RADIUS_DECAY": 0.62,
        "BASE_SIZE": 4,

        "LATERAL_PROB": 0.82,                # 0.75→0.82，密度回升
        "MAX_ORDER": 5,                      # 4→5，恢复一层细枝
        "BRANCH_MAX_LIFE": 22,               # 18→22，枝条恢复长度，但仍比原版 25 短
        "SPAWN_DECAY": 0.6,                  # 0.55→0.6，配合 MAX_ORDER=5 让深层有节点
        "BRANCH_MIN_LIFE": 3,                # 2→3，配合密度回升防止冒出短刺

        "TRUNK_TROPISM_Z": 1.0,
        "TRUNK_TROPISM_STRENGTH": 0.03,
        "BRANCH_TROPISM_Z": 1.0,
        "BRANCH_TROPISM_STRENGTH": 0.09,     # ★ 0.05→0.09，看得见的上扬

        "WHORL_MODE": 0,
        "WHORL_SIZE": 5,

        "LEAF_RADIUS": 0.004,
        "PIPE_EXPONENT": 2.6,
        "SEGMENT_BEND": 5.5,                 # ★ 3.0→5.5，明显的曲折

        "TRUNK_MAX_LIFE": 12,                # ★ 100→12，主干 12 节就分叉

        "TRUNK_SPAWN_END_MARGIN": 2,         # 顶端 2 节不再发新枝，避免顶尖
    },

    # "willow": {
    #     "MAX_AGE": 22,
    #     "INTERNODE_LENGTH": 0.30,
    #     "INTERNODE_LENGTH_DECAY": 0.95,
    #     "TRUNK_LENGTH_BOOST": 1.4,
    #     "BRANCH_ANGLE_DECAY": 0.85,
    #     "APICAL_DOMINANCE": 0.88,            # ★ 0.94→0.88，scaffold 早些自然终止
    #     "LATERAL_PROB": 0.90,                # ★ 0.85→0.90，密度补一点
    #     "MAX_ORDER": 4,
    #     "INITIAL_RADIUS": 0.05,
    #     "RADIUS_DECAY": 0.62,
    #     "BASE_SIZE": 4,
    #     "SPAWN_DECAY": 0.80,                 # ★ 0.65→0.80，order-2/3/4 都密集 spawn
    #
    #     "TRUNK_TROPISM_Z": 1.0,
    #     "TRUNK_TROPISM_STRENGTH": 0.05,
    #     "BRANCH_TROPISM_Z": -1.0,
    #     "BRANCH_TROPISM_STRENGTH": 0.30,     # ★ 0.25→0.30，下垂更急
    #
    #     "WHORL_MODE": 3,
    #     "WHORL_SIZE": 5,
    #     "BRANCH_MAX_LIFE": 18,               # ★ 25→18，柳条不超长
    #
    #     "BRANCH_ANGLE": 80,                       # ★ 85→80，scaffold 出生角度稍上扬
    #     "BRANCH_ANGLE_VAR": 8,                    # ★ 6→8，允许一些角度变化（更自然）
    #     "WILLOW_SCAFFOLD_TROPISM_Z": -1.0,        # 中等下垂
    #     "WILLOW_SCAFFOLD_TROPISM_STRENGTH": 0.08, # 弱拉力，scaffold 缓慢下垂形成拱形
    #
    #     "LEAF_RADIUS": 0.003,
    #     "PIPE_EXPONENT": 2.3,
    #     "SEGMENT_BEND": 2.5,
    #     "TRUNK_SEGMENT_BEND": 1.0,
    #
    #     "TRUNK_MAX_LIFE": 14,
    #     "BRANCH_MIN_LIFE": 0,
    # },
    "willow": {
        "MAX_AGE": 22,
        "INTERNODE_LENGTH": 0.30,
        "INTERNODE_LENGTH_DECAY": 0.95,
        "TRUNK_LENGTH_BOOST": 1.4,
        "BRANCH_ANGLE_DECAY": 0.85,
        "APICAL_DOMINANCE": 0.88,
        "MAX_ORDER": 3,                            # ★ 4→3,少一层细枝,节点数砍一刀
        "INITIAL_RADIUS": 0.05,
        "RADIUS_DECAY": 0.62,
        "BASE_SIZE": 4,
        "SPAWN_DECAY": 0.55,                       # ★ 0.80→0.55,深层 spawn 概率骤减

        "TRUNK_TROPISM_Z": 1.0,
        "TRUNK_TROPISM_STRENGTH": 0.05,
        "BRANCH_TROPISM_Z": -1.0,
        "BRANCH_TROPISM_STRENGTH": 0.30,           # 柳条强下垂,不变

        "WHORL_MODE": 3,
        "WHORL_SIZE": 5,
        "BRANCH_MAX_LIFE": 16,                     # 18→16,柳条不要长到失控

        "BRANCH_ANGLE": 95,                        # 保留上次的(出生水平偏下)
        "BRANCH_ANGLE_VAR": 5,                     # 保留上次的
        # "LATERAL_PROB": 0.80,                      # 0.90→0.80,稍降密度

        "WILLOW_SCAFFOLD_TROPISM_Z": -1.0,
        "WILLOW_SCAFFOLD_TROPISM_STRENGTH": 0.10,  # ★ 0.20→0.10,侧面 scaffold 保持水平骨架感
        # "WILLOW_TOP_SCAFFOLD_TROPISM_STRENGTH": 0.28,  # ★ 新增:顶部 5 根 scaffold 专用,大幅下拉

        "LEAF_RADIUS": 0.003,
        "PIPE_EXPONENT": 2.3,
        "SEGMENT_BEND": 2.5,
        "TRUNK_SEGMENT_BEND": 1.0,

        "TRUNK_MAX_LIFE": 14,
        # "BRANCH_MIN_LIFE": 1,                      # ★ 3→1,关键修复!只压最短的 1 节刺,不强制长 3 节

        # species_presets.py 的 willow 字典里继续微调:

        "WILLOW_TOP_SCAFFOLD_TROPISM_STRENGTH": 0.35,   # ★ 0.28→0.35,顶部再拉狠一点
        "BRANCH_SPAWN_END_MARGIN": 2,                   # ★ 新增,scaffold/柳条末端 2 节不再 spawn,消除"短刺"
        "BRANCH_MIN_LIFE": 2,                           # ★ 1→2,让 scaffold 至少长 2 节,给 tropism 一节的时间生效
        "LATERAL_PROB": 0.85,                           # 0.80→0.85,柳条稍密一点
    },
}


def perturb_params(params, perturbation=0.15, rng=None):
    import random as _r
    if rng is None:
        rng = _r.Random()

    # 完全不扰动的"分类/开关"参数
    NO_PERTURB = {
        "WHORL_MODE",          # 分类变量（0/1/3 各代表一种模式）
        "WHORL_SIZE",          # 整数计数（whorl 每圈几根）
    }
    # 扰动后取整的"离散计数"参数
    DISCRETE = {
        "MAX_AGE", "MAX_ORDER", "BASE_SIZE",
        "TRUNK_MAX_LIFE", "BRANCH_MAX_LIFE",
        "TRUNK_SPAWN_END_MARGIN", "BRANCH_SPAWN_END_MARGIN",
        "BRANCH_MIN_LIFE",
    }

    out = {}
    for k, v in params.items():
        if k in NO_PERTURB:
            out[k] = v
        elif k in DISCRETE:
            out[k] = max(0, int(round(v * (1 + rng.uniform(-perturbation, perturbation)))))
        else:
            out[k] = v * (1 + rng.uniform(-perturbation, perturbation))
    return out