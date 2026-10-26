"""
lidar_sim.py
============
从三角网格 (mesh) 模拟 ALS(机载) / TLS(地面站) / MLS(移动) 点云。

核心思路：
  所有扫描模式本质都是 "传感器位置 + 一束射线方向" → 与 mesh 求交。
  三种模式的区别只在于 **射线是怎么生成的**：
    - ALS : 沿航线飞行，扫描镜在垂直航向方向左右摆动 (±FOV)
    - TLS : 固定站，球面扫描 (方位角 0-360°, 俯仰角范围)
    - MLS : 沿轨迹移动，旋转多线扫描 (类似 Velodyne)
  求交后再叠加真实传感器效应：测距噪声、入射角掉点、最大量程、随机丢失。

依赖: numpy, open3d (>=0.17)。torch 非必需；如需 GPU/可微，可把射线生成搬到 torch，
求交仍建议用 open3d 的 RaycastingScene(基于 Embree, CPU 上已很快)。

返回统一为 ScanResult，含 points 及逐点属性 (range / incidence / intensity / source_id)，
方便你后续按需筛选或喂给网络。
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np
import open3d as o3d


# --------------------------------------------------------------------------- #
#  结果容器
# --------------------------------------------------------------------------- #
@dataclass
class ScanResult:
    points: np.ndarray                      # (N,3) 模拟得到的点
    ranges: np.ndarray = field(default=None)        # (N,)  传感器到点的距离
    incidence: np.ndarray = field(default=None)     # (N,)  入射角(弧度), 0=正打
    intensity: np.ndarray = field(default=None)     # (N,)  伪强度 [0,1]
    source_id: np.ndarray = field(default=None)     # (N,)  来自哪个站/航线/帧
    sensor_pos: np.ndarray = field(default=None)    # (M,3) 各传感器位置(便于可视化)

    def __len__(self):
        return 0 if self.points is None else len(self.points)

    def to_o3d(self) -> o3d.geometry.PointCloud:
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(self.points)
        if self.intensity is not None:
            g = self.intensity[:, None].repeat(3, axis=1)
            pc.colors = o3d.utility.Vector3dVector(g)
        return pc

    def save(self, path: str):
        o3d.io.write_point_cloud(path, self.to_o3d())


# --------------------------------------------------------------------------- #
#  主类
# --------------------------------------------------------------------------- #
class LiDARSimulator:
    def __init__(self, mesh: Union[o3d.geometry.TriangleMesh, str],
                 seed: Optional[int] = 0):
        """
        mesh : open3d legacy TriangleMesh 或 mesh 文件路径(.ply/.obj/...)
        """
        if isinstance(mesh, str):
            mesh = o3d.io.read_triangle_mesh(mesh)
        if not mesh.has_triangle_normals():
            mesh.compute_triangle_normals()
        self.mesh = mesh
        self.aabb = mesh.get_axis_aligned_bounding_box()

        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    #  底层：投射一批射线，返回命中点 + 属性
    # ------------------------------------------------------------------ #
    def _cast(self, origins: np.ndarray, directions: np.ndarray,
              max_range: float) -> dict:
        """
        origins, directions : (N,3)。directions 不必预归一化。
        return: dict(points, ranges, incidence, normals, valid_mask)
        """
        dirs = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12)
        rays = np.concatenate([origins, dirs], axis=1).astype(np.float32)
        ans = self.scene.cast_rays(o3d.core.Tensor(rays))

        t = ans["t_hit"].numpy()                       # 沿射线的命中距离
        normals = ans["primitive_normals"].numpy()     # 命中三角面法向

        valid = np.isfinite(t) & (t <= max_range) & (t > 1e-6)
        t_safe = np.where(valid, t, 0.0)               # 避免 inf*0 -> nan 警告
        pts = origins + dirs * t_safe[:, None]

        # 入射角: 射线方向与面法向的夹角 (取锐角)
        cos_inc = np.abs(np.einsum("ij,ij->i", dirs, normals))
        cos_inc = np.clip(cos_inc, 0.0, 1.0)
        incidence = np.arccos(cos_inc)

        return dict(points=pts, ranges=t, incidence=incidence,
                    normals=normals, valid=valid)

    # ------------------------------------------------------------------ #
    #  底层：叠加真实传感器效应并打包
    # ------------------------------------------------------------------ #
    def _finalize(self, cast: dict, source_id: int,
                  range_noise: float, angular_jitter: float,
                  dropout: float, grazing_limit: float,
                  sensor_pos: np.ndarray) -> ScanResult:
        """
        range_noise   : 测距高斯噪声 std (米)
        angular_jitter: 沿法向的横向抖动 std (米)，近似光斑/角度噪声
        dropout       : 随机丢点概率 [0,1] (大气/低反射)
        grazing_limit : 入射角超过此值(弧度)按掠射概率丢点
        """
        m = cast["valid"].copy()
        pts = cast["points"]
        rng_ = cast["ranges"]
        inc = cast["incidence"]
        normals = cast["normals"]

        # 1) 掠射掉点：入射角越大越易丢失，cos(inc) 作为返回概率
        if grazing_limit is not None:
            graze = inc > grazing_limit
            p_keep = np.cos(inc)                     # 0(掠射)~1(正打)
            roll = self.rng.random(len(inc))
            m &= ~(graze & (roll > p_keep))

        # 2) 随机丢点
        if dropout > 0:
            m &= self.rng.random(len(m)) > dropout

        if not m.any():
            return ScanResult(points=np.empty((0, 3)), ranges=np.empty(0),
                              incidence=np.empty(0), intensity=np.empty(0),
                              source_id=np.empty(0, int),
                              sensor_pos=np.atleast_2d(sensor_pos))

        pts, rng_, inc, normals = pts[m], rng_[m], inc[m], normals[m]
        n = len(pts)

        # 3) 测距噪声：沿射线方向加噪 (用 点-传感器 方向近似射线方向)
        if range_noise > 0:
            ray_dir = pts - np.atleast_2d(sensor_pos)[0]
            ray_dir /= (np.linalg.norm(ray_dir, axis=1, keepdims=True) + 1e-12)
            pts = pts + ray_dir * self.rng.normal(0, range_noise, n)[:, None]

        # 4) 横向/角度抖动：在切平面内随机偏移
        if angular_jitter > 0:
            rand = self.rng.normal(0, 1, (n, 3))
            tang = rand - normals * np.einsum("ij,ij->i", rand, normals)[:, None]
            tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12)
            pts = pts + tang * self.rng.normal(0, angular_jitter, n)[:, None]

        # 5) 伪强度：~ cos(入射角) / range^2，归一化到 [0,1]
        intensity = np.cos(inc) / (rng_ ** 2 + 1e-6)
        if intensity.max() > 0:
            intensity = intensity / intensity.max()

        return ScanResult(
            points=pts, ranges=rng_, incidence=inc, intensity=intensity,
            source_id=np.full(n, source_id, dtype=int),
            sensor_pos=np.atleast_2d(sensor_pos),
        )

    # ================================================================== #
    #  ALS —— 机载 (Airborne Laser Scanning)
    # ================================================================== #
    def simulate_als(self,
                     gsd: float = 0.1,
                     altitude: Optional[float] = None,
                     oblique_deg: Sequence[float] = (0.0,),
                     oblique_dir: Sequence[float] = (1, 0, 0),
                     margin: float = 0.1,
                     range_noise: float = 0.05,
                     dropout: float = 0.02,
                     grazing_limit_deg: float = 80.0,
                     max_range: Optional[float] = None) -> ScanResult:
        """
        机载俯视扫描：在物体足印正上方按地面采样距离(GSD)规则打俯视射线格网。
        这样无论物体多小/有没有地面，都能保证落到目标上(而不是把脉冲打进空气)。

        gsd         : 地面采样距离(米) —— 格网间距, 决定点密度(越小越密)
        altitude    : 相对 mesh 顶部的飞行高度; None 时按物体尺寸自动设
        oblique_deg : 斜视角列表(度)。默认 (0,)=纯天底俯视(只能看到树冠顶);
                      传 (-20,0,20) 可加侧视, 让树干/侧枝也被扫到(代价是射线更多)
        oblique_dir : 斜视倾斜所在的水平方向
        margin      : 足印外扩比例(相对物体尺寸), 留点余量
        """
        mn, mx = self.aabb.min_bound, self.aabb.max_bound
        size = mx - mn
        if altitude is None:
            altitude = float(max(size)) * 2.0 + 20.0
        z0 = mx[2] + altitude
        if max_range is None:
            max_range = (z0 - mn[2]) * 3.0 + 10.0

        # 在 (足印 + margin) 上铺规则格网
        pad = size[:2] * margin
        xs = np.arange(mn[0] - pad[0], mx[0] + pad[0] + gsd, gsd)
        ys = np.arange(mn[1] - pad[1], mx[1] + pad[1] + gsd, gsd)
        X, Y = np.meshgrid(xs, ys)
        grid = np.c_[X.ravel(), Y.ravel(), np.full(X.size, z0)]

        odir = np.asarray(oblique_dir, float)
        odir = odir / (np.linalg.norm(odir) + 1e-12)

        origins, dirs = [], []
        for ob in oblique_deg:
            th = np.deg2rad(ob)
            d = np.array([0, 0, -1.0]) * np.cos(th) + odir * np.sin(th)  # 天底向 odir 倾斜
            d = d / np.linalg.norm(d)
            origins.append(grid)
            dirs.append(np.repeat(d[None], len(grid), axis=0))
        origins = np.concatenate(origins)
        dirs = np.concatenate(dirs)

        cast = self._cast(origins, dirs, max_range)
        return self._finalize(cast, source_id=0,
                              range_noise=range_noise, angular_jitter=0.0,
                              dropout=dropout,
                              grazing_limit=np.deg2rad(grazing_limit_deg),
                              sensor_pos=np.array([[(mn[0] + mx[0]) / 2,
                                                    (mn[1] + mx[1]) / 2, z0]]))

    # ================================================================== #
    #  TLS —— 地面固定站 (Terrestrial Laser Scanning)
    # ================================================================== #
    def simulate_tls(self,
                     stations: Optional[np.ndarray] = None,
                     n_stations: int = 4,
                     sensor_height: float = 1.6,
                     ang_res_deg: float = 0.05,
                     elev_range_deg: Sequence[float] = (-40, 80),
                     range_noise: float = 0.004,
                     angular_jitter: float = 0.002,
                     dropout: float = 0.01,
                     grazing_limit_deg: float = 85.0,
                     max_range: float = 120.0) -> ScanResult:
        """
        在若干固定站做球面扫描(方位角 0-360°, 俯仰角 elev_range)。
        多站可显著减少遮挡。

        stations    : (M,3) 自定义站位; 为 None 时在场景外圈自动布 n_stations 个站
        sensor_height: 仪器架设高度(自动布站时, 站在地面+此高度)
        ang_res_deg : 角分辨率(方位与俯仰相同)
        elev_range_deg: 俯仰角范围, 0=水平, 正=向上
        """
        mn, mx = self.aabb.min_bound, self.aabb.max_bound
        center = (mn + mx) / 2

        if stations is None:
            radius = 0.6 * np.linalg.norm(mx[:2] - mn[:2])
            ang = np.linspace(0, 2 * np.pi, n_stations, endpoint=False)
            stations = np.c_[center[0] + radius * np.cos(ang),
                             center[1] + radius * np.sin(ang),
                             np.full(n_stations, mn[2] + sensor_height)]
        stations = np.atleast_2d(np.asarray(stations, float))

        az = np.deg2rad(np.arange(0, 360, ang_res_deg))
        el = np.deg2rad(np.arange(elev_range_deg[0], elev_range_deg[1] + 1e-9,
                                  ang_res_deg))
        AZ, EL = np.meshgrid(az, el)
        AZ, EL = AZ.ravel(), EL.ravel()
        base_dirs = np.c_[np.cos(EL) * np.cos(AZ),
                          np.cos(EL) * np.sin(AZ),
                          np.sin(EL)]

        results = []
        for sid, st in enumerate(stations):
            origins = np.repeat(st[None], len(base_dirs), axis=0)
            cast = self._cast(origins, base_dirs, max_range)
            r = self._finalize(cast, source_id=sid,
                               range_noise=range_noise,
                               angular_jitter=angular_jitter,
                               dropout=dropout,
                               grazing_limit=np.deg2rad(grazing_limit_deg),
                               sensor_pos=st)
            if len(r):
                results.append(r)
        return _merge(results, stations)

    # ================================================================== #
    #  MLS —— 移动测量 (Mobile Laser Scanning)
    # ================================================================== #
    def simulate_mls(self,
                     trajectory: Optional[np.ndarray] = None,
                     path: str = "loop",
                     standoff: Optional[float] = None,
                     sensor_height: float = 2.0,
                     n_beams: int = 32,
                     vert_fov_deg: Optional[Sequence[float]] = None,
                     az_res_deg: float = 0.2,
                     frame_spacing: float = 0.2,
                     range_noise: float = 0.02,
                     angular_jitter: float = 0.003,
                     dropout: float = 0.02,
                     grazing_limit_deg: float = 82.0,
                     max_range: float = 80.0) -> ScanResult:
        """
        传感器沿 trajectory 移动, 每帧做旋转多线扫描(类 Velodyne):
        n_beams 条垂直通道 × 360° 方位。

        trajectory   : (K,3) 自定义轨迹; 为 None 时按 path 自动生成
        path         : 自动轨迹形状 —— 'loop'(绕物体一圈, 适合单个物体如树/雕塑)
                       或 'pass'(从物体旁直线驶过, 适合街景式)
        standoff     : 轨迹到物体的水平距离(米); None 时按物体尺寸自动设
        sensor_height: 传感器安装高度(相对物体底部)
        n_beams      : 垂直线数
        vert_fov_deg : 垂直视场 (min,max); None 时自动张开上限以覆盖物体顶部
        az_res_deg   : 水平方位分辨率
        frame_spacing: 沿轨迹每隔多少米取一帧
        """
        mn, mx = self.aabb.min_bound, self.aabb.max_bound
        center = (mn + mx) / 2
        size = mx - mn
        obj_r = 0.5 * float(np.linalg.norm(size[:2]))     # 物体水平半径
        if standoff is None:
            standoff = max(obj_r * 0.5, 3.0)
        path_r = obj_r + standoff

        if trajectory is None:
            if path == "loop":                            # 绕物体一圈
                n = max(int(2 * np.pi * path_r / frame_spacing), 8)
                a = np.linspace(0, 2 * np.pi, n, endpoint=False)
                trajectory = np.c_[center[0] + path_r * np.cos(a),
                                   center[1] + path_r * np.sin(a),
                                   np.full(n, mn[2])]
            else:                                         # 从旁边直线驶过
                axis = int(np.argmax(size[:2]))
                other = 1 - axis
                k = max(int(size[axis] * 1.4 / frame_spacing), 2)
                traj = np.zeros((k, 3))
                ext = np.linspace(mn[axis] - 0.2 * size[axis],
                                  mx[axis] + 0.2 * size[axis], k)
                traj[:, axis] = ext
                traj[:, other] = center[other] + path_r
                traj[:, 2] = mn[2]
                trajectory = traj
        trajectory = np.asarray(trajectory, float)

        # 自动张开垂直视场: 让上限能从轨迹处看到物体顶部
        if vert_fov_deg is None:
            top_elev = np.rad2deg(np.arctan2(
                size[2] - sensor_height, max(standoff, 0.5)))
            vert_fov_deg = (-25.0, float(np.clip(top_elev + 5, 15, 88)))

        # 按 frame_spacing 沿轨迹重采样
        seg = np.diff(trajectory, axis=0)
        seglen = np.linalg.norm(seg, axis=1)
        s = np.concatenate([[0], np.cumsum(seglen)])
        n_frames = max(int(s[-1] / frame_spacing), 1)
        s_q = np.linspace(0, s[-1], n_frames)
        frames = np.c_[np.interp(s_q, s, trajectory[:, 0]),
                       np.interp(s_q, s, trajectory[:, 1]),
                       np.interp(s_q, s, trajectory[:, 2])]
        frames[:, 2] += sensor_height

        # 多线方向 (sensor 自身坐标系: 360 方位 × n_beams 俯仰)
        el = np.deg2rad(np.linspace(vert_fov_deg[0], vert_fov_deg[1], n_beams))
        az = np.deg2rad(np.arange(0, 360, az_res_deg))
        AZ, EL = np.meshgrid(az, el)
        AZ, EL = AZ.ravel(), EL.ravel()
        base_dirs = np.c_[np.cos(EL) * np.cos(AZ),
                          np.cos(EL) * np.sin(AZ),
                          np.sin(EL)]

        results = []
        for fid, fp in enumerate(frames):
            origins = np.repeat(fp[None], len(base_dirs), axis=0)
            cast = self._cast(origins, base_dirs, max_range)
            r = self._finalize(cast, source_id=fid,
                               range_noise=range_noise,
                               angular_jitter=angular_jitter,
                               dropout=dropout,
                               grazing_limit=np.deg2rad(grazing_limit_deg),
                               sensor_pos=fp)
            if len(r):
                results.append(r)
        return _merge(results, frames)


# --------------------------------------------------------------------------- #
#  合并多个 ScanResult
# --------------------------------------------------------------------------- #
def _merge(results: Sequence[ScanResult], sensor_pos: np.ndarray) -> ScanResult:
    if not results:
        return ScanResult(points=np.empty((0, 3)), ranges=np.empty(0),
                          incidence=np.empty(0), intensity=np.empty(0),
                          source_id=np.empty(0, int), sensor_pos=sensor_pos)
    return ScanResult(
        points=np.concatenate([r.points for r in results]),
        ranges=np.concatenate([r.ranges for r in results]),
        incidence=np.concatenate([r.incidence for r in results]),
        intensity=np.concatenate([r.intensity for r in results]),
        source_id=np.concatenate([r.source_id for r in results]),
        sensor_pos=np.atleast_2d(sensor_pos),
    )


# --------------------------------------------------------------------------- #
#  便捷入口
# --------------------------------------------------------------------------- #
# 密度预设：sparse=稀疏(秒出, 测流程用) / medium=中等 / dense=真实设备密度
PRESETS = {
    "als": {
        "sparse": dict(gsd=0.3),
        "medium": dict(gsd=0.1),
        "dense":  dict(gsd=0.05),
    },
    "tls": {
        "sparse": dict(n_stations=3, ang_res_deg=0.5),
        "medium": dict(n_stations=4, ang_res_deg=0.1),
        "dense":  dict(n_stations=4, ang_res_deg=0.05),
    },
    "mls": {
        "sparse": dict(n_beams=16, az_res_deg=1.0, frame_spacing=2.0),
        "medium": dict(n_beams=32, az_res_deg=0.4, frame_spacing=0.8),
        "dense":  dict(n_beams=32, az_res_deg=0.2, frame_spacing=0.2),
    },
}


def simulate(mesh, mode: str = "als", density: str = "sparse",
             seed: int = 0, **kwargs) -> ScanResult:
    """
    mode    : 'als' | 'tls' | 'mls'
    density : 'sparse' | 'medium' | 'dense'  (预设密度, 默认 sparse)
    **kwargs: 显式参数, 会覆盖预设里的同名项。
    """
    mode = mode.lower()
    params = dict(PRESETS[mode][density])     # 先取预设
    params.update(kwargs)                     # 再用显式参数覆盖
    sim = LiDARSimulator(mesh, seed=seed)
    return {"als": sim.simulate_als,
            "tls": sim.simulate_tls,
            "mls": sim.simulate_mls}[mode](**params)