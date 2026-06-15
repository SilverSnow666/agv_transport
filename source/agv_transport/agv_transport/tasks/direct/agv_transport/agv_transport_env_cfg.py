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

    当前版本：V5.2-B0-train-safe-narrow-corridor-clearance。

    阶段目标：在 V5.2-A5 物理边界 zero-shot 成功基础上，小幅收窄物理通道，
    并加入轻量 wall-clearance penalty，验证既有 V5.1C / V5.2-A5 策略是否能安全迁移。

    设计原则：
    - 保持规则矩形 payload，不引入异形件、不引入质心偏移。
    - 保持 V5.1C 软走廊 reward，不放松路径约束，不鼓励撞墙探索。
    - 基于手动 U 型左右边界控制点生成低矮物理墙。
    - 相比 V5.2-A5，左右墙各向内收窄约 0.03 m。
    - 从收窄通道阶段开始加入轻量 wall-clearance penalty，约束 kinematic AGV 贴墙/穿墙风险。
    - 推荐先用 V5.1.0c.pt zero-shot 评估；若失败，再 fine-tune。
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
    episode_length_s = 80.0

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
    agv_safe_distance = 0.46
    agv_collision_distance = 0.41

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
        (0.80, 0.00),
        (1.60, 0.00),
        (2.40, -0.40),  # 缓和入弯
        (3.00, -0.80),
        (3.60, -0.80),  # 直线过渡
        (4.20, -0.40),  # 缓和出弯
        (4.80, 0.00),
        (5.60, 0.00),
    )
    # V4.1C 连续路径跟踪前视距离
    path_lookahead_dist = 0.32

    # D0A0f：路径遵从约束。
    # 当前模型已经能到终点，但存在近似直线 shortcut，因此提高横向误差惩罚，
    # 并加入软路径走廊；不直接使用墙壁硬终止，先保持训练稳定。
    path_lateral_error_scale = 0.75

    # V5.2-B0：保留 V5.1C 的软走廊压力。
    # 注意：path_corridor_half_width 是 reward 中的软惩罚阈值，不等于物理墙位置。
    # 当前评估中 payload 最大横向误差约 0.18~0.19 m，因此继续保留 0.10 m 软走廊，
    # 用于提供路径贴合压力；真实墙体则采用宽通道，避免直接阻塞 AGV2/AGV3 的侧向工作空间。
    progress_corridor_width = 0.60
    path_corridor_half_width = 0.10
    path_corridor_penalty_scale = 12.0

    # success 仍要求 payload 到达终点且路径推进充分。
    success_progress_ratio = 0.92
    success_path_lateral_error = 0.25

    # 继续启用 waypoint gate，保持按 waypoint 顺序推进。
    enable_waypoint_gate = True

    # V5.2-B0：在 V5.2-A5 zero-shot 通过基础上，小幅收窄物理通道。
    # 该边界不是最终窄通道，只用于引入真实刚体碰撞反馈。
    # inner_half_width 表示路径中心线到墙体内侧面的距离。
    # 手动边界已从 V5.2-A5 向内收窄约 0.03 m，用于验证更强物理边界约束。
    enable_physical_path_boundaries = True
    path_boundary_inner_half_width = 1.35
    path_boundary_wall_thickness = 0.08
    path_boundary_wall_height = 0.10
    # 起点向后延伸，确保 AGV 初始区域也处于通道包络内。
    path_boundary_start_extension = 1.50
    # V5.2-B0：显式左右边界控制点 + 稠密短墙段 + joint cap。
    # 不再默认使用中心线自动 offset，避免大 offset 在内弯处产生自交。
    path_boundary_use_manual_edges = True
    path_boundary_smoothing_iterations = 3
    path_boundary_manual_smoothing_iterations = 2
    path_boundary_sample_step = 0.20
    # 使用 joint cap 处理连接点，不再通过墙段重叠掩盖缝隙，避免局部穿插。
    path_boundary_segment_overlap = 0.0
    path_boundary_use_joint_caps = True
    path_boundary_joint_cap_radius = 0.04
    # Large-scale training safety:
    # Keep this False for --num_envs 2048.  If True, every wall segment/cap may
    # spawn its own material and cloned environments can exceed the PhysX 64K
    # material limit.  Enable only for low-num_envs visual inspection if needed.
    path_boundary_enable_materials = False
    path_boundary_static_friction = 1.0
    path_boundary_dynamic_friction = 1.0
    path_boundary_color = (0.25, 0.25, 0.25)

    # V5.2-B0：AGV 是 kinematic 写位姿，物理墙不能完全依靠 PhysX 推回。
    # 因此从收窄通道阶段开始加入轻量 wall-clearance penalty，
    # 只惩罚贴墙/穿墙风险，不改变 payload 主路径 reward。
    enable_wall_clearance_penalty = True
    agv_wall_clearance_margin = 0.08
    agv_wall_clearance_penalty_scale = 4.0
    payload_wall_clearance_margin = 0.20
    payload_wall_clearance_penalty_scale = 2.0

    # 手动 U 型物理通道边界控制点，局部坐标。
    # 这些点表示墙体中心线，不是 payload 轨迹，也不是墙体内侧边。
    # 设计目标：左右墙各自独立成连续 U 型，避免由中心线 offset 导致内侧墙自交。
    # 若视觉检查发现边界过宽/过窄，只微调这些控制点，不改 payload 或主路径 reward。
    path_boundary_left_points = (
        (-1.50, 1.42),
        (0.40, 1.42),
        (1.60, 1.42),
        (2.40, 1.02),
        (3.00, 0.62),
        (3.60, 0.62),
        (4.20, 1.02),
        (4.80, 1.42),
        (7.10, 1.42),
    )
    path_boundary_right_points = (
        (-1.50, -1.42),
        (0.40, -1.42),
        (1.60, -1.42),
        (2.40, -1.82),
        (3.00, -2.22),
        (3.60, -2.22),
        (4.20, -1.82),
        (4.80, -1.42),
        (7.10, -1.42),
    )


    # D0A0g：waypoint gate。
    # 只有依次经过关键 waypoints 附近，最终 success 才成立。
    # 这样可以防止在开放空间里直接切直线进入目标半径。
    waypoint_gate_radius = 0.30
    waypoint_gate_reward_scale = 4.0

    # D0A0h：active waypoint 子目标驱动。
    # 当前 target 不再直接指向最终目标或 lookahead target，而是指向当前 active waypoint。
    # payload 到达当前 waypoint 后，active_goal_idx 自动切换到下一个 waypoint。
    # I-plus：在 v4.3I 稳定 20/20 的基础上，略收紧 subgoal_radius，减少“擦边过 waypoint”。
    enable_subgoal_waypoint = True
    subgoal_radius = 0.28
    subgoal_final_radius = 0.22
    subgoal_progress_reward_scale = 15.0
    subgoal_backward_penalty_scale = 8.5
    subgoal_reach_reward_scale = 20.0

    # D0A0h-plus：子目标转向稳定项。
    # near_subgoal 越大，越要求 payload 和动作降速，避免接近 waypoint 后继续高速直推。
    subgoal_slow_radius = 0.60
    subgoal_speed_penalty_scale = 1.20
    subgoal_action_slow_penalty_scale = 0.20

    # 如果 payload 已经沿当前段方向越过 active waypoint，则按平方惩罚，防止第 4 个 waypoint 后推过头。
    subgoal_overshoot_penalty_scale = 10.0

    # 引导 payload yaw 与当前 active segment 方向大致一致，帮助第 4->第 5 waypoint 下拐。
    subgoal_yaw_alignment_scale = 0.15

    # 已有两台 AGV 有效推动时，低贡献且未接触 payload 的 AGV 不应持续向前空跑。
    idle_forward_no_contact_penalty_scale = 1.20

    # D0A0i-plus：在转弯角色切换基础上轻量收紧子目标贴合。
    # 当 active segment 的 y 方向明显向下/右转时，继续鼓励 AGV2 靠近并参与有效推动，
    # 同时轻微抑制 AGV3 在该段继续过强直推；本版只小幅增强，不破坏已有 20/20 成功策略。
    turn_role_y_threshold = 0.08
    # I-plus：轻量增强 AGV2 在右转/下拐段的招募，不改成全程三车强制接触。
    turn_role_push_reward_scale = 3.5
    turn_role_contact_zone_reward_scale = 1.5
    turn_role_contact_zone_norm = 0.80
    turn_opposite_push_penalty_scale = 0.7

    # [新增] 转弯时压制中间车（AGV1），迫使其交出主导权
    turn_center_push_penalty_scale = 2.0

    # ========== 3. 新增脱队截断机制 (Truncation) ==========
    terminate_on_agv_escape = True
    agv_escape_dist_threshold = 2.0  # 距离 Payload 超过 2.0m 视为逃逸
    agv_escape_penalty_scale = 50.0  # 触发逃逸终止时的巨额惩罚

    # 两车有效推动 credit：progress 的主奖励由第二台有效推动车辆 gate。
    progress_base_reward_scale = 10.0
    progress_two_pusher_bonus_scale = 18.0
    backward_progress_penalty_scale = 12.0
    two_pusher_gate_threshold = 0.20
    single_pusher_progress_penalty_scale = 2.0

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

    # 低贡献车辆约束。
    # V5.0.3 采用中等偏松设置：允许闲置车在转弯后恢复、追赶和让位，
    # 但不完全取消动作代价，避免 AGV2/AGV3 过早抢占中心主推角色。
    idle_action_penalty_scale = 0.10
    idle_action_rate_penalty_scale = 0.15
    idle_standby_penalty_scale = 1.20
    idle_low_utility_threshold = 0.08
    idle_two_pusher_gate_threshold = 0.50
    idle_standby_min_dist = 0.80
    idle_standby_max_dist = 1.45

    # 适度队形约束。
    # 平行推送与平行 contact flag 已消除侧车内扣梯度；这里保留车道纪律，
    # 防止 AGV2/AGV3 横向漂移或过早夹占中心车空间。
    formation_error_mean_scale = 0.06
    formation_error_max_scale = 0.02
    heading_alignment_mean_scale = 0.15
    heading_alignment_min_scale = 0.05

    # 中间 waypoint 的通过半径
    waypoint_radius = 0.20

    # 最终目标点位置误差：D0A0f 收紧目标半径，减少“进入大半径即成功”的 shortcut。
    target_radius = 0.22

    # 最终目标点 yaw 误差
    target_yaw_radius = 0.25



    # 工作空间限制，防止物体飞太远
    workspace_limit = 6.0

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