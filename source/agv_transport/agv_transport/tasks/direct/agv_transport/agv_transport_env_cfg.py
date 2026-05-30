from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR



@configclass
class AgvTransportEnvCfg(DirectRLEnvCfg):
    """单 AGV 推箱子任务配置。

    第一版简化假设：
    - AGV 用一个可控方块表示；
    - AGV 通过二维速度 [vx, vy] 在平面移动；
    - payload 是一个动态刚体；
    - 目标点固定在每个环境坐标系下的 x 正方向。
    """
    # Isaac Sim 自带 AGV / AMR 视觉模型
    # 优先使用 Idealworks iwhub static，比较像工业 AGV
    agv_visual_usd_path = f"{ISAAC_NUCLEUS_DIR}/Robots/Idealworks/iwhub/iw_hub_static.usd"

    agv_visual_cfg = sim_utils.UsdFileCfg(
        usd_path=agv_visual_usd_path,
        scale=(0.50, 0.50, 0.50),
    )

    # 环境设置
    decimation = 2
    episode_length_s = 24.0

    # 动作：差速 AGV 控制 [v, w]
    # v: 线速度
    # w: 角速度
    action_space = 6

    # 观测：
    # agv_xy_rel, payload_xy_rel, target_xy_rel,
    # agv_to_payload_xy, payload_to_target_xy,
    # agv_heading_xy, agv_vel_xy, payload_vel_xy
    # 维度 = 2 + 2 + 2 + 2 + 2 + 2 + 2 + 2 = 16
    observation_space = 44
    state_space = 0

    # 仿真设置
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
    )

    # 场景设置
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128,
        env_spacing=4.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    # 简化 AGV 参数
    agv_size = (0.70, 0.45, 0.06)
    agv_mass = 20.0
    # 三台 AGV 初始位置
    agv_init_positions = (
        (-1.60, 0.00, 0.03),
        (-1.60, 0.65, 0.03),
        (-1.60, -0.65, 0.03),
    )


    # 三车并排推送队形参数
    formation_stand_off_distances = (0.90, 0.90, 0.90)
    formation_lateral_offsets = (0.0, 0.65, -0.65)

    # PPO 训练用近似接触阈值
    train_contact_threshold = 1.20

    # AGV-AGV 安全距离约束
    # agv_size = (0.70, 0.45, 0.06)，并排 y 间距 0.65，因此安全距离先取 0.55
    agv_safe_distance = 0.55

    # 若两车中心距离小于该值，认为发生严重重叠/碰撞
    agv_collision_distance = 0.42



    # 为兼容部分旧代码，保留单个 agv_init_pos
    agv_init_pos = agv_init_positions[0]

    # 是否随机化 AGV 初始 y 位置
    randomize_agv_init_y = False

    # AGV 初始 y 随机范围
    agv_init_y_range = (-0.30, 0.30)

    # 货物参数
    payload_size = (1.20, 1.60, 0.30)
    payload_mass = 30.0
    payload_init_pos = (0.0, 0.0, 0.15)

    # 目标点，基于每个 env 原点的局部坐标
    #用于兼容旧模型
    target_pos = (1.80, 0.0, 0.0)

    # V4.0 多 waypoint 路径
    # 先用轻微折线路径，不要一开始太难
    # V4.1A 平滑折线路径
    waypoints = (
        (0.60, 0.00),
        (0.95, 0.20),
        (1.30, 0.30),
        (1.60, 0.10),
        (1.90, 0.00),
    )

    # 中间 waypoint 的通过半径
    waypoint_radius = 0.20

    # 最终目标点位置误差
    target_radius = 0.20

    # 最终目标点 yaw 误差
    target_yaw_radius = 0.20



    # 工作空间限制，防止物体飞太远
    workspace_limit = 2.5

    # 差速 AGV 动作缩放
    max_agv_linear_speed = 0.5
    max_agv_angular_speed = 1.2

    # 奖励权重
    reward_progress_scale = 8.0
    reward_distance_scale = 1.0
    reward_agv_payload_distance_scale = 0.15
    reward_action_penalty_scale = 0.02
    reward_success = 20.0
    reward_out_of_bounds = -10.0

    # AGV：第一版设置为 kinematic，便于先跑通推箱子闭环
    agv1_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/AGV1",
        spawn=sim_utils.CuboidCfg(
            size=agv_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=agv_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.02, 0.02, 0.02),
                metallic=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=agv_init_positions[0],
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    agv2_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/AGV2",
        spawn=sim_utils.CuboidCfg(
            size=agv_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=agv_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.02, 0.02, 0.02),
                metallic=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=agv_init_positions[1],
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    agv3_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/AGV3",
        spawn=sim_utils.CuboidCfg(
            size=agv_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=agv_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.02, 0.02, 0.02),
                metallic=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=agv_init_positions[2],
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # Payload：动态刚体，被 AGV 推动
    payload_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Payload",
        spawn=sim_utils.CuboidCfg(
            size=payload_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=payload_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.9,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.55, 0.0),
                metallic=0.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=payload_init_pos,
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )