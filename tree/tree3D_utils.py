import numpy as np
import open3d as o3d


def build_tree_mesh(nodes, radial_segments=8, radius_scale=1.0, min_radius=1e-31,
                    fork_angle_threshold_deg=25.0):
    """
    - chain 内（夹角 < 阈值）：共享父 ring → 光滑
    - 分叉（夹角 ≥ 阈值或非最佳对齐的兄弟）：lateral 自带 start_ring，从父表面"长出"
    """
    n = len(nodes)
    if n == 0:
        return o3d.geometry.TriangleMesh()
    R = radial_segments
    cos_thresh = float(np.cos(np.deg2rad(fork_angle_threshold_deg)))

    positions   = np.array([nd["xyz"]        for nd in nodes], dtype=np.float64)
    starts      = np.array([nd["start_xyz"]  for nd in nodes], dtype=np.float64)
    radii       = np.maximum(np.array([nd["radius"] for nd in nodes], dtype=np.float64),
                             min_radius) * radius_scale
    parent_idxs = np.array([nd["parent_idx"] for nd in nodes], dtype=np.int64)

    tangents = positions - starts
    lengths  = np.linalg.norm(tangents, axis=1)
    ok = lengths > 1e-9
    tangents[ok]  = tangents[ok] / lengths[ok, None]
    tangents[~ok] = np.array([0.0, 0.0, 1.0])

    children = [[] for _ in range(n)]
    roots = []
    for i in range(n):
        p = parent_idxs[i]
        if p >= 0:
            children[p].append(i)
        else:
            roots.append(i)

    # ---- 选每个父的 continuation 子（兄弟里与父切向夹角最小的那个，且必须 < 阈值）----
    is_continuation = np.zeros(n, dtype=bool)
    for p in range(n):
        cs = children[p]
        if not cs:
            continue
        dots = [float(np.dot(tangents[c], tangents[p])) for c in cs]
        k = int(np.argmax(dots))
        # 只有一个孩子：强制 continuation（避免父 ring 留洞）
        if len(cs) == 1 or dots[k] > cos_thresh:
            is_continuation[cs[k]] = True

    # ---- 框架辅助 ----
    def init_frame(d):
        ref = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(d, ref); un = np.linalg.norm(u)
        if un < 1e-9:
            return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
        u = u / un
        return u, np.cross(d, u)

    def rot(x, axis, ca, sa):
        return x * ca + np.cross(axis, x) * sa + axis * np.dot(axis, x) * (1 - ca)

    def transport(u, v, t_from, t_to):
        axis = np.cross(t_from, t_to); an = np.linalg.norm(axis)
        if an < 1e-6:
            return u, v
        axis = axis / an
        ang = np.arccos(float(np.clip(np.dot(t_from, t_to), -1.0, 1.0)))
        ca, sa = np.cos(ang), np.sin(ang)
        return rot(u, axis, ca, sa), rot(v, axis, ca, sa)

    # ---- BFS：continuation 用 transport，lateral 独立 init_frame ----
    u_frames = np.zeros((n, 3))
    v_frames = np.zeros((n, 3))
    visited = [False] * n
    from collections import deque
    queue = deque()
    for r in roots:
        u_frames[r], v_frames[r] = init_frame(tangents[r])
        visited[r] = True
        queue.append(r)
    while queue:
        i = queue.popleft()
        for c in children[i]:
            if visited[c]:
                continue
            if is_continuation[c]:
                u_frames[c], v_frames[c] = transport(
                    u_frames[i], v_frames[i], tangents[i], tangents[c]
                )
            else:
                u_frames[c], v_frames[c] = init_frame(tangents[c])
            visited[c] = True
            queue.append(c)

    # ---- 每个节点的 own_ring ----
    angles = np.linspace(0.0, 2.0 * np.pi, R, endpoint=False)
    cos_v = np.cos(angles)[None, :, None]
    sin_v = np.sin(angles)[None, :, None]
    offsets   = cos_v * u_frames[:, None, :] + sin_v * v_frames[:, None, :]   # (n, R, 3)
    own_rings = positions[:, None, :] + radii[:, None, None] * offsets         # (n, R, 3)

    # ---- 需要额外 start_ring 的节点：roots + laterals ----
    needs_start = np.zeros(n, dtype=bool)
    for r in roots:
        needs_start[r] = True
    for i in range(n):
        if parent_idxs[i] >= 0 and not is_continuation[i]:
            needs_start[i] = True

    extra_idxs = np.where(needs_start)[0]
    # start_ring 半径：lateral 用父半径，root 用自身
    extra_radii = radii[extra_idxs].copy()
    for k, idx in enumerate(extra_idxs):
        p = parent_idxs[idx]
        if p >= 0:
            extra_radii[k] = radii[p]
    extra_rings = (starts[extra_idxs][:, None, :]
                 + extra_radii[:, None, None] * offsets[extra_idxs])           # (m, R, 3)

    vertices = np.concatenate(
        [own_rings.reshape(-1, 3), extra_rings.reshape(-1, 3)], axis=0
    )
    extra_base = {int(extra_idxs[k]): n * R + k * R for k in range(len(extra_idxs))}

    # ---- 三角形 ----
    j      = np.arange(R)
    j_next = (j + 1) % R
    tri_list = []

    # continuation: parent.own_ring → child.own_ring
    cont_idx = np.where(is_continuation)[0]
    if len(cont_idx) > 0:
        p_base = (parent_idxs[cont_idx] * R)[:, None]
        c_base = (cont_idx * R)[:, None]
        t1 = np.stack([p_base + j,      p_base + j_next, c_base + j     ], axis=-1)
        t2 = np.stack([p_base + j_next, c_base + j_next, c_base + j     ], axis=-1)
        tri_list.append(t1.reshape(-1, 3))
        tri_list.append(t2.reshape(-1, 3))

    # lateral + root: extra_ring → own_ring
    if len(extra_idxs) > 0:
        s_base = np.array([extra_base[int(i)] for i in extra_idxs])[:, None]
        c_base = (extra_idxs * R)[:, None]
        t1 = np.stack([s_base + j,      s_base + j_next, c_base + j     ], axis=-1)
        t2 = np.stack([s_base + j_next, c_base + j_next, c_base + j     ], axis=-1)
        tri_list.append(t1.reshape(-1, 3))
        tri_list.append(t2.reshape(-1, 3))

    triangles = np.concatenate(tri_list, axis=0).astype(np.int32)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    return mesh


# 物种叶形模板：每个是 (s, w) 顶点列表，s ∈ [0, 1] 沿叶长方向（0=基部, 1=尖端），
# w ∈ [-1, 1] 横向半宽（实际宽度 = w * 0.5 * leaf_width）。
# 必须按"绕轮廓顺时针或逆时针"的顺序，从基部出发，绕一圈回基部。
LEAF_SHAPES = {
    "oak": [           # 椭圆裂叶（简化版，5 裂的轮廓）
        (0.00,  0.00), (0.10,  0.35), (0.25,  0.55), (0.30,  0.30),
        (0.45,  0.75), (0.50,  0.45), (0.65,  0.80), (0.72,  0.50),
        (0.85,  0.65), (0.92,  0.30), (1.00,  0.00),
        (0.92, -0.30), (0.85, -0.65), (0.72, -0.50), (0.65, -0.80),
        (0.50, -0.45), (0.45, -0.75), (0.30, -0.30), (0.25, -0.55),
        (0.10, -0.35),
    ],
    "maple": [         # 5 裂掌状（简化）
        (0.00,  0.00), (0.15,  0.30), (0.30,  0.40), (0.30,  0.70),
        (0.55,  0.55), (0.55,  0.95), (0.80,  0.60), (1.00,  0.00),
        (0.80, -0.60), (0.55, -0.95), (0.55, -0.55), (0.30, -0.70),
        (0.30, -0.40), (0.15, -0.30),
    ],
    "birch": [         # 卵形带细齿（用六边形近似）
        (0.00,  0.00), (0.20,  0.60), (0.50,  0.85), (0.80,  0.55),
        (1.00,  0.00),
        (0.80, -0.55), (0.50, -0.85), (0.20, -0.60),
    ],
    "willow": [        # 长条柳叶
        (0.00,  0.00), (0.15,  0.45), (0.50,  0.55), (0.85,  0.40),
        (1.00,  0.00),
        (0.85, -0.40), (0.50, -0.55), (0.15, -0.45),
    ],
    "pine": [          # 针叶——细长四边形即可（仅一对侧顶点）
        (0.00,  0.00), (0.20,  0.35), (1.00,  0.00),
        (0.20, -0.35),
    ],
    "default": [       # 兜底椭圆
        (0.00,  0.00), (0.20,  0.50), (0.50,  0.70), (0.80,  0.50),
        (1.00,  0.00),
        (0.80, -0.50), (0.50, -0.70), (0.20, -0.50),
    ],
}


def build_leaves_mesh(nodes, species=None, seed=0,
                      radius_threshold=0.020,
                      density=30.0,
                      leaf_length=0.10,
                      leaf_width=0.06,
                      droop_deg=35.0,
                      droop_var_deg=15.0,
                      size_var=0.3,
                      petiole_frac=0.10):

    rng = np.random.default_rng(seed)
    # 节点插值
    interpolate_n = 10 if species == "willow" else 1
    new_nds = [[{"xyz": nd["start_xyz"] + i * (np.array(nd["xyz"]) - nd["start_xyz"]) / (interpolate_n + 1),
                "start_xyz": nd["start_xyz"], "radius": nd["radius"], "birth_step": nd["birth_step"],
                "parent_idx": nd["parent_idx"]} for i in range(1, interpolate_n+1)]
               for nd in nodes if nd["parent_idx"] != -1]
    for new_node_list in new_nds:
        nodes = nodes + new_node_list
    gaussian_weight = 0.04
    new_nds = [{"xyz": ((1 - gaussian_weight) * np.array(nd["xyz"]) + gaussian_weight * rng.normal(size=3)).tolist(),
                "start_xyz": nd["start_xyz"], "radius": nd["radius"], "birth_step": nd["birth_step"],
                "parent_idx": nd["parent_idx"]} for nd in nodes]
    nodes = nodes + new_nds

    SPECIES_LEAF = {
        "willow":  dict(leaf_length=0.035, leaf_width=0.018, droop_deg=70, density=40),
        "birch":   dict(leaf_length=0.15, leaf_width=0.1,  droop_deg=20, density=20),
        "oak":     dict(leaf_length=0.10, leaf_width=0.07,  droop_deg=30, density=30),
        "maple":   dict(leaf_length=0.08, leaf_width=0.08,  droop_deg=25, density=28),
        "pine":    dict(leaf_length=0.3, leaf_width=0.1, droop_deg=10, density=30),
    }
    if species in SPECIES_LEAF:
        d = SPECIES_LEAF[species]
        leaf_length = d.get("leaf_length", leaf_length)
        leaf_width  = d.get("leaf_width",  leaf_width)
        droop_deg   = d.get("droop_deg",   droop_deg)
        density     = d.get("density",     density)

    # 选叶形模板
    shape = np.array(LEAF_SHAPES.get(species, LEAF_SHAPES["default"]), dtype=np.float64)
    V = len(shape)   # 模板顶点数
    s_template = shape[:, 0]   # (V,)
    w_template = shape[:, 1]   # (V,)

    rng = np.random.default_rng(seed)
    n = len(nodes)
    if n == 0:
        return o3d.geometry.TriangleMesh()

    starts = np.array([nd["start_xyz"] for nd in nodes], dtype=np.float64)
    ends   = np.array([nd["xyz"]       for nd in nodes], dtype=np.float64)
    radii  = np.array([nd["radius"]    for nd in nodes], dtype=np.float64)

    # 自动用当前树最粗半径的某一分数作为阈值
    # 这样不管 pipe model 出来多大半径都能正确剔除主干
    max_radius, min_radius = float(radii.max()), float(radii.min())
    # auto_threshold = max_radius * 0.15  # 只有最细 30% 半径以内的枝长叶
    auto_threshold = min_radius * (1.5 if species == "willow" else 1)  # 只有最细 30% 半径以内的枝长叶
    # radius_threshold = min(radius_threshold, auto_threshold)
    radius_threshold = auto_threshold

    dirs   = ends - starts
    lengths = np.linalg.norm(dirs, axis=1)

    grow = (lengths > 1e-6)

    ratio = radii / radius_threshold
    density_factor = 1.0 / (1.0 + ratio ** 2)
    density_factor[ratio > 1.5] = 0.0  # 超过 2.5 倍阈值直接归零

    expected = density * lengths * density_factor
    expected[~grow] = 0.0

    leaf_counts = rng.poisson(expected)
    total = int(leaf_counts.sum())
    if total == 0:
        return o3d.geometry.TriangleMesh()

    seg_idx = np.repeat(np.arange(n), leaf_counts)
    s = rng.uniform(0.05, 0.95, total)
    phi = rng.uniform(0.0, 2.0 * np.pi, total)
    theta = np.deg2rad(rng.normal(droop_deg, droop_var_deg, total))
    sz = rng.uniform(1.0 - size_var, 1.0 + size_var, total)

    # size_scale = 1.0 + 0.4 * (1.0 - radii[seg_idx] / radius_threshold)
    size_scale = 1.0 + 0.4 * (radii[seg_idx] / radius_threshold)
    size_scale = np.clip(size_scale, 0.7, 1.4)
    L = leaf_length * sz * size_scale
    W = leaf_width * sz * size_scale
    L = leaf_length * radii[seg_idx] * 400
    W = leaf_width * radii[seg_idx] * 400


    branch_dir = (ends[seg_idx] - starts[seg_idx]) / lengths[seg_idx, None]

    aux = np.tile(np.array([0.0, 0.0, 1.0]), (total, 1))
    aux[np.abs(branch_dir[:, 2]) > 0.9] = np.array([1.0, 0.0, 0.0])

    u_base = np.cross(branch_dir, aux)
    u_base = u_base / np.linalg.norm(u_base, axis=1, keepdims=True)
    v_base = np.cross(branch_dir, u_base)

    cos_phi = np.cos(phi)[:, None]
    sin_phi = np.sin(phi)[:, None]
    radial = cos_phi * u_base + sin_phi * v_base

    attach_axis = starts[seg_idx] + s[:, None] * (ends[seg_idx] - starts[seg_idx])
    attach = attach_axis + radii[seg_idx][:, None] * radial

    cos_t = np.cos(theta)[:, None]
    sin_t = np.sin(theta)[:, None]
    leaf_axis = cos_t * branch_dir + sin_t * radial

    # gravity_bias = 0.25
    gravity_bias = 0.05  # 0.25 → 0.05，原来的下压过强

    leaf_axis = leaf_axis - gravity_bias * np.array([0.0, 0.0, 1.0]) * np.abs(leaf_axis[:, 2:3])
    leaf_axis = leaf_axis / np.linalg.norm(leaf_axis, axis=1, keepdims=True)

    # 替换原来的 leaf_side 那一段
    # 叶面横向 = 沿 branch_dir 旋转 phi 角后的"右手方向"
    # 即在 (u_base, v_base) 平面里取与 radial 正交的那个分量
    leaf_side = -sin_phi * u_base + cos_phi * v_base  # 与 radial 同平面、互相垂直

    # 然后让叶面整体绕 leaf_axis 自由旋转一个随机角（叶片自转），破除规整感
    roll = rng.uniform(0.0, 2.0 * np.pi, total)[:, None]
    # 绕 leaf_axis 旋转 leaf_side 角度 roll
    cos_r, sin_r = np.cos(roll), np.sin(roll)
    # Rodrigues 公式：v' = v cos + (k×v) sin + k(k·v)(1-cos)
    # 其中 k = leaf_axis, v = leaf_side
    kxv = np.cross(leaf_axis, leaf_side)
    kdv = np.sum(leaf_axis * leaf_side, axis=1, keepdims=True)
    leaf_side = leaf_side * cos_r + kxv * sin_r + leaf_axis * kdv * (1 - cos_r)
    leaf_side = leaf_side / np.linalg.norm(leaf_side, axis=1, keepdims=True)


    side_norm = np.linalg.norm(leaf_side, axis=1, keepdims=True)
    bad = side_norm[:, 0] < 1e-6
    leaf_side[bad] = radial[bad]
    leaf_side = leaf_side / np.linalg.norm(leaf_side, axis=1, keepdims=True)

    petiole_end = attach + (petiole_frac * L)[:, None] * leaf_axis
    blade_len = ((1.0 - petiole_frac) * L)   # (total,)

    # ===== 关键改动：用模板生成多顶点叶轮廓 =====
    # 每个模板顶点位置 = petiole_end + s_template * blade_len * leaf_axis
    #                              + w_template * (W/2) * leaf_side
    # 用 broadcast 一次性算所有叶子的所有顶点
    s_t = s_template[None, :, None]       # (1, V, 1)
    w_t = w_template[None, :, None]       # (1, V, 1)
    blade_len_b = blade_len[:, None, None]   # (total, 1, 1)
    half_w_b = (W * 0.5)[:, None, None]      # (total, 1, 1)

    along  = s_t * blade_len_b * leaf_axis[:, None, :]    # (total, V, 3)
    across = w_t * half_w_b    * leaf_side[:, None, :]    # (total, V, 3)

    vertices = petiole_end[:, None, :] + along + across   # (total, V, 3)
    vertices = vertices.reshape(-1, 3)                    # (total * V, 3)

    # 三角化叶面：用 fan triangulation（从顶点 0 扇形铺开）
    # 模板内的三角形：[(0, k, k+1) for k in 1..V-2]
    fan_tris = np.stack([
        np.zeros(V - 2, dtype=np.int64),
        np.arange(1, V - 1, dtype=np.int64),
        np.arange(2, V,     dtype=np.int64),
    ], axis=-1)                                            # (V-2, 3)
    T = V - 2

    base = (np.arange(total) * V).reshape(-1, 1, 1)        # (total, 1, 1)
    triangles = (base + fan_tris[None, :, :]).reshape(-1, 3)

    # # 双面：补一份反向三角（避免背面剔除导致一半叶子消失）
    # open3d可以设置显示反面，这是不必要的
    # triangles_back = triangles[:, [0, 2, 1]]
    # triangles = np.concatenate([triangles, triangles_back], axis=0)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices  = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles.astype(np.int32))
    mesh.compute_vertex_normals()
    return mesh


def colorize_bark(mesh, base_color=(0.42, 0.28, 0.18),
                  noise_strength=0.08, z_gradient=0.15,
                  stripe_freq=50.0, stripe_strength=0.04, seed=42):
    """
    给树干 mesh 上"木头风"的顶点色:基色 + 局部噪声 + 纵向明暗 + 纵纹条纹。
    用 vertex_colors,兼容 draw_geometries。
    """
    rng = np.random.default_rng(seed)
    verts = np.asarray(mesh.vertices)
    n = len(verts)
    base = np.array(base_color, dtype=np.float32)

    # 1) 每顶点小噪声 — 木纹的细微颜色变化
    noise = rng.uniform(-noise_strength, noise_strength, size=(n, 3))

    # 2) 沿 z 的明暗渐变 — 底部深、顶部稍浅(也可反过来)
    z = verts[:, 2]
    z_norm = (z - z.min()) / (z.max() - z.min() + 1e-9)
    z_shade = ((z_norm - 0.5) * z_gradient)[:, None]

    # 3) 纵向高频条纹 — 伪造树皮裂纹的方向感
    stripes = stripe_strength * np.sin(z * stripe_freq +
                                       rng.uniform(0, 6.28, size=n))
    stripes = stripes[:, None]

    colors = np.clip(base + noise + z_shade + stripes, 0.0, 1.0).astype(np.float32)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    return mesh


def displace_bark(mesh, strength=0.004):
    """
    沿顶点法线施加伪噪声位移,模拟树皮的不规则凹凸。
    strength 不能太大,否则节与节之间连接处会撕裂。
    建议值:树干半径的 5%~10%。
    """
    mesh.compute_vertex_normals()
    verts = np.asarray(mesh.vertices).copy()
    normals = np.asarray(mesh.vertex_normals)

    # 用顶点坐标作为种子的伪噪声,保证空间上连贯
    h = (np.sin(verts[:, 0] * 47.0) *
         np.cos(verts[:, 1] * 53.0) *
         np.sin(verts[:, 2] * 41.0))
    h += 0.5 * np.sin(verts[:, 0] * 13.0) * np.cos(verts[:, 1] * 17.0)
    # 归一化到 [-0.5, 0.5]
    h = h / (np.abs(h).max() + 1e-9) * 0.5

    verts += normals * h[:, None] * strength
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.compute_vertex_normals()  # 位移后法线要重算
    return mesh
