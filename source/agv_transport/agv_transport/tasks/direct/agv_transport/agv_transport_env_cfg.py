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
    """三 AGV 无连接协同推送任务配置。

    当前版本用于 V4.3-D0A0h-plus-turn-control：
    - 延续 D0A0b 的 two-pusher credit，允许任意两台 AGV 有效推动；
    - 不强制 AGV2 必须接触，但要求低贡献 AGV 在已有两车推动时保持低动作、低抖动；
    - 保留任意两台 AGV 有效推动的 two-pusher credit；
    - 不强制第三台 AGV 接触，但进一步抑制低贡献 AGV 抽搐；
    - 将 target 从连续 lookahead 改为当前 active waypoint 子目标；
    - 通过“当前子目标距离减少”来驱动 payload 依次经过 waypoint，
      避免开放空间中直接 shortcut 到终点；
    - 保留宽而短的小矩形 payload 与前端几何接触判定。
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
    episode_length_s = 45.0

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
        env_spacing=6.8,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    # 简化 AGV 参数
    agv_size = (0.70, 0.45, 0.06)
    agv_mass = 20.0
    # 三台 AGV 初始位置。
    # 由于 payload 加宽到 y=1.20，侧车可以保持合理横向间距而不拥挤。
    agv_init_positions = (
        (-0.90, 0.00, 0.03),
        (-0.90, 0.52, 0.03),
        (-0.90, -0.52, 0.03),
    )


    # 三车并排推送队形参数。
    # stand_off 由 payload 半长 0.45 + AGV 半长 0.35 推得，留 0.02 m clearance。
    # lateral offset = 0.52：既能让侧车与 payload 有足够横向重叠，又避免三车横向拥挤。
    formation_stand_off_distances = (0.82, 0.82, 0.82)
    formation_lateral_offsets = (0.0, 0.52, -0.52)

    # 旧版中心点距离接触阈值，仅保留给兼容脚本使用。
    # 当前 reward / observation 的 contact_flags 改为“前端接触带”几何判断，
    # 避免小矩形 payload 下 AGV2/AGV3 实际接触但中心距略大而被误判为未接触。
    train_contact_threshold = 1.08

    # D0A0-geom：前端接触带参数。
    # front point 到 payload 后缘 x 的距离小于该值，且 y 落在 payload 宽度+margin 内，即认为有效前端接触。
    # 使用 AGV 前边缘线段与 payload 后缘接触带的几何重叠判定。
    front_contact_x_margin = 0.12
    front_contact_y_margin = 0.08

    # V4.2-C0：软前端接触约束。
    # 注意：C0 阶段不要硬终止，先保护 V4.1 三车成功策略。
    front_rear_margin = 0.05
    front_contact_heading_min = 0.20

    # 前端合理接触的小奖励，权重很低，避免压过原 V4.1 推进/队形奖励。
    front_contact_reward_scale = 0.10

    # 车尾接触、接触倒车、接触朝向不合理的软惩罚。
    rear_contact_penalty_scale = 1.0
    reverse_contact_penalty_scale = 1.5
    contact_heading_penalty_scale = 0.5

    # 明显车尾倒推 payload 的软惩罚；C0 阶段不终止。
    bad_rear_push_penalty_scale = 5.0
    terminate_on_bad_rear_push = False
    bad_rear_heading_threshold = -0.10
    bad_rear_reverse_threshold = -0.05

    # AGV-AGV 安全距离约束。
    # D0A1 阶段只使用温和软分离，避免过强避让破坏角色分配。
    agv_safe_distance = 0.50
    agv_collision_distance = 0.43

    agv_overlap_penalty_scale = 5.0
    agv_collision_penalty_scale = 15.0

    # 当前阶段不因 AGV 间碰撞直接终止，只在 reward 中软惩罚。
    terminate_on_agv_collision = False


    # 为兼容部分旧代码，保留单个 agv_init_pos
    agv_init_pos = agv_init_positions[0]

    # 是否随机化 AGV 初始 y 位置
    randomize_agv_init_y = False

    # AGV 初始 y 随机范围
    agv_init_y_range = (-0.30, 0.30)

    # 货物参数：宽而短的小矩形 payload。
    # y 方向加宽到 1.20，为 AGV2/AGV3 提供稳定接触面；x 方向缩短到 0.90，便于观察曲线与 yaw。
    payload_size = (0.90, 1.20, 0.30)
    payload_mass = 24.0
    payload_init_pos = (0.0, 0.0, 0.15)

    # 目标点，基于每个 env 原点的局部坐标
    #用于兼容旧模型
    target_pos = (3.10, 0.0, 0.0)

    # V4.3-D0A0f：中等偏强曲率路径。
    # 当前阶段不再加大曲率，而是通过 success progress 与 corridor penalty 强化路径遵从。
    waypoints = (
        (0.45, 0.00),
        (0.95, 0.28),
        (1.45, 0.45),
        (2.05, 0.20),
        # D0A0h-plus：把原本 (2.05, 0.20) -> (2.65, -0.25) 的急转向段拆细，
        # 让 payload 在两车推送下先学会连续下拐，而不是到第 4 点后继续直推过头。
        (2.30, 0.02),
        (2.55, -0.16),
        (2.85, -0.16),
        (3.10, 0.00),
    )

    # V4.1C 连续路径跟踪前视距离
    path_lookahead_dist = 0.32

    # D0A0f：路径遵从约束。
    # 当前模型已经能到终点，但存在近似直线 shortcut，因此提高横向误差惩罚，
    # 并加入软路径走廊；不直接使用墙壁硬终止，先保持训练稳定。
    path_lateral_error_scale = 0.60

    # D0A0g-easy：温和路径遵从。
    # 1) 离路径较远时，正向 progress reward 只被 soft gate 部分衰减；
    # 2) 超出较宽软走廊后按平方轻惩罚；
    # 3) success 需要路径进度合格，但末端横向误差先保持较宽，避免训练骤崩。
    progress_corridor_width = 0.65
    path_corridor_half_width = 0.40
    path_corridor_penalty_scale = 2.0

    # success 需要 payload 不仅进入目标半径，还要沿路径推进到足够靠近末端。
    # easy 阶段先使用 0.96 和较宽 path lateral 条件，保护已有成功策略。
    success_progress_ratio = 0.90
    success_path_lateral_error = 0.45

    # D0A0g-easy：暂时关闭强 waypoint gate。
    # 等 soft-path 阶段把 path_lat_max 压到 0.25~0.30 后，再逐步打开。
    enable_waypoint_gate = False

    # D0A0g：waypoint gate。
    # 只有依次经过关键 waypoints 附近，最终 success 才成立。
    # 这样可以防止在开放空间里直接切直线进入目标半径。
    waypoint_gate_radius = 0.35
    waypoint_gate_reward_scale = 4.0

    # D0A0h：active waypoint 子目标驱动。
    # 当前 target 不再直接指向最终目标或 lookahead target，而是指向当前 active waypoint。
    # payload 到达当前 waypoint 后，active_goal_idx 自动切换到下一个 waypoint。
    enable_subgoal_waypoint = True
    subgoal_radius = 0.34
    subgoal_final_radius = 0.22
    subgoal_progress_reward_scale = 14.0
    subgoal_backward_penalty_scale = 8.0
    subgoal_reach_reward_scale = 18.0

    # D0A0h-plus：子目标转向稳定项。
    # near_subgoal 越大，越要求 payload 和动作降速，避免接近 waypoint 后继续高速直推。
    subgoal_slow_radius = 0.55
    subgoal_speed_penalty_scale = 1.20
    subgoal_action_slow_penalty_scale = 0.20

    # 如果 payload 已经沿当前段方向越过 active waypoint，则按平方惩罚，防止第 4 个 waypoint 后推过头。
    subgoal_overshoot_penalty_scale = 10.0

    # 引导 payload yaw 与当前 active segment 方向大致一致，帮助第 4->第 5 waypoint 下拐。
    subgoal_yaw_alignment_scale = 0.45

    # 已有两台 AGV 有效推动时，低贡献且未接触 payload 的 AGV 不应持续向前空跑。
    idle_forward_no_contact_penalty_scale = 0.80

    # 两车有效推动 credit：progress 的主奖励由第二台有效推动车辆 gate。
    progress_base_reward_scale = 10.0
    progress_two_pusher_bonus_scale = 18.0
    backward_progress_penalty_scale = 12.0
    two_pusher_gate_threshold = 0.20
    single_pusher_progress_penalty_scale = 12.0

    # 成功奖励按整个 episode 中“两车有效推动进度占比”缩放，避免单车推到终点拿满分。
    success_reward_scale = 150.0
    success_base_ratio = 0.40
    success_two_pusher_ratio = 0.60

    # 接触与冷启动奖励。
    effective_push_reward_scale = 0.50
    second_pusher_reward_scale = 0.20
    front_contact_count_reward_scale = 0.05
    pre_push_reward_scale = 0.12
    contact_zone_approach_reward_scale = 0.80
    contact_zone_error_norm = 1.00

    approach_reward_scale = 1.00

    # D0A0d：两车接触保持与最大进度保持，减少推到半途后断接触/后退。
    contact_persistence_reward_scale = 0.25
    progress_drop_penalty_scale = 4.0
    progress_drop_tolerance = 0.03

    # 只有在已有两台有效推动时，才惩罚低贡献 AGV 的无意义动作和抽搐。
    # easy 阶段保持中等强度，避免为了压 AGV2 抽搐而破坏两车推送主策略。
    idle_action_penalty_scale = 0.28
    idle_action_rate_penalty_scale = 0.32
    idle_standby_penalty_scale = 0.10
    idle_low_utility_threshold = 0.08
    idle_two_pusher_gate_threshold = 0.60
    idle_standby_min_dist = 0.80
    idle_standby_max_dist = 1.70

    # 固定队形只做弱约束，避免强行要求三台 AGV 始终同时接触。
    formation_error_mean_scale = 0.10
    formation_error_max_scale = 0.03
    heading_alignment_mean_scale = 0.15
    heading_alignment_min_scale = 0.05

    # 中间 waypoint 的通过半径
    waypoint_radius = 0.20

    # 最终目标点位置误差：D0A0f 收紧目标半径，减少“进入大半径即成功”的 shortcut。
    target_radius = 0.22

    # 最终目标点 yaw 误差
    target_yaw_radius = 0.25



    # 工作空间限制，防止物体飞太远
    workspace_limit = 3.8

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