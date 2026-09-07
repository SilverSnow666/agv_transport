from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


@configclass
class AgvLevelCarryEnvCfg(DirectRLEnvCfg):
    """V6.2.5：保留 V6.2.4 视觉/物理对齐，增加 AGV1 纵向位置闭环防抢跑。

    设计目标：在 V6.0 平地矩形 payload baseline 之后，先打开轻度崎岖支撑面，不直接换异形件。
    三台 kinematic AGV 作为移动支撑平台，payload 为动态刚体，靠重力、接触和摩擦
    放置在三车顶部。训练目标是把 payload 平稳送到目标点，同时限制 roll/pitch、
    滑移、支撑丢失和三车队形失稳。

    注意：这是“驮运/承载”新任务，不应覆盖 V5.3.3 的推送任务。
    """

    # ------------------------- 基本仿真设置 -------------------------
    decimation = 2
    episode_length_s = 22.0
    action_space = 6          # [v1, w1, v2, w2, v3, w3]
    observation_space = 56    # 见 agv_carry_env.py::_get_observations：26 + 3*10
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=128,
        env_spacing=5.0,
        replicate_physics=True,
        clone_in_fabric=False,
    )

    # ------------------------- AGV / Payload 几何 -------------------------
    agv_visual_usd_path = f"{ISAAC_NUCLEUS_DIR}/Robots/Idealworks/iwhub/iw_hub_static.usd"
    agv_visual_cfg = sim_utils.UsdFileCfg(
        usd_path=agv_visual_usd_path,
        scale=(0.45, 0.45, 0.45),
    )

    # ------------------------- V7.0 顶升机构 -------------------------

    # 第一阶段使用理想 kinematic 顶升平台。
    # lift_height 表示 AGV 顶面到 Lift Plate 底面的垂向伸出量。
    lift_plate_size = (0.28, 0.28, 0.04)
    lift_plate_mass = 5.0

    # 顶升行程。
    lift_min_height = 0.00
    lift_max_height = 0.10

    # 运输工作高度只预顶升 30 mm；完全收缩时 Lift Plate 底面贴合 AGV 顶面。
    lift_neutral_height = 0.03

    # 无碰撞、无质量的纯视觉伸缩杆。杆原型为单位高度 cylinder，运行时沿
    # AGV 局部 +Z 缩放并放置在黄色 USD 外观顶面与 Lift Plate 底面之间。
    lift_actuator_visual_radius = 0.028
    lift_actuator_visual_min_height = 0.002
    lift_actuator_visual_color = (0.16, 0.18, 0.22)

    # 独立的纯视觉安装基准（AGV 局部 +Z）。GUI 校准后把杆根部放在车体
    # 内部 1 mm，而不是由隐藏物理代理的 agv_top_z 在运行时推导。该量仅影响
    # marker，不改变 AGV/Lift Plate 物理位姿。
    lift_visual_mount_height = 0.079

    # 与 280 x 280 x 40 mm 隐藏碰撞代理解耦的纯视觉承载头。其顶面与隐藏
    # 物理板顶面重合，使小承载头贴近 Board；圆杆负责连接黄色车顶，不参与 PhysX。
    lift_head_visual_size = (0.14, 0.14, 0.015)
    lift_head_visual_color = (0.20, 0.23, 0.28)
    debug_show_lift_collision_proxies = False

    # 后续动态调平时使用，目前 V7.0-A 暂时不控制。
    max_lift_speed = 0.04  # m/s
    lift_position_kp = 5.0

    # ------------------------- 纯视觉高度补偿 -------------------------
    # 物理代理尺寸、根节点高度和 payload 物理高度全部恢复 V6.2.1。
    # 这里只改变无碰撞外观子节点的局部变换，避免再次改变 checkpoint 的状态分布。
    # V6.2.3 中 AGV 根高度为 0.0525、visual_z=-0.020；恢复根高度 0.080 后，
    # 使用 -0.0475 可使可视模型原点的世界高度仍约为 0.0325 m。
    agv_visual_translation = (0.05, 0.0, -0.0475)

    # V6.2.1 的 payload 物理中心比 V6.2.3 高 0.064 m
    # （代理顶面差 0.055 m + clearance 差 0.009 m）。黄色外观单独向下补偿。
    payload_visual_translation = (0.0, 0.0, 0.0)
    payload_visual_cfg = sim_utils.CuboidCfg(
        size=(1.60, 1.20, 0.08),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 0.55, 0.0), metallic=0.0
        ),
    )

    # 调试开关只影响显示，不影响碰撞和训练。
    debug_show_agv_collision_proxies = False
    debug_show_payload_collision_proxy = False

    # 三台 AGV 是移动支撑台。保持 V6.2.1 的 kinematic cuboid。
    agv_size = (0.55, 0.42, 0.16)
    agv_mass = 30.0
    agv_center_z = 0.08
    agv_top_z = agv_center_z + 0.5 * agv_size[2]

    # V7.1.1 equivalent wheel/ground contact footprint.  The visible iwhub
    # model and the kinematic collision proxy do not have identical extents,
    # so these are deliberately defined from (and kept inside) the 0.55 x
    # 0.42 m physical proxy rather than claimed as exact tyre locations.
    terrain_contact_wheelbase = 0.46
    terrain_contact_track = 0.34

    # V7.0 carrier board.
    # 暂时沿用 payload 变量名，后续加入独立 Cargo 后再正式重命名。
    payload_size = (1.60, 1.20, 0.08)
    payload_mass = 12.0

    # V7.2 free-cargo evaluation baseline.  It remains opt-in so existing
    # training, reward, action and observation behaviour is unchanged.
    enable_cargo = False
    cargo_size = (0.30, 0.30, 0.20)
    cargo_mass = 4.0
    cargo_static_friction = 0.80
    cargo_dynamic_friction = 0.65
    cargo_board_clearance = 0.0
    cargo_contact_tolerance = 0.012
    cargo_tip_threshold = 0.7853981633974483  # 45 deg

    # 保持 V6.2.1 的物理支撑间隙与初始高度，确保旧 checkpoint 兼容。
    board_support_clearance = 0.003

    payload_init_z = (
            agv_top_z
            + lift_neutral_height
            + lift_plate_size[2]
            + 0.5 * payload_size[2]
            + board_support_clearance
    )

    payload_init_pos = (0.0, 0.0, payload_init_z)

    # Cargo starts exactly face-to-face with the Board: no initial overlap and
    # no suspension gap.  It is centred and never attached to the Board.
    cargo_init_z = (
        payload_init_z + 0.5 * payload_size[2] + 0.5 * cargo_size[2] + cargo_board_clearance
    )
    cargo_init_pos = (0.0, 0.0, cargo_init_z)

    # 支撑三角形：local_x 为运输方向，local_y 为横向。
    # AGV1 在前中，AGV2 在后左，AGV3 在后右。三点都位于 payload 投影内。
    support_offsets_xy = (
        (0.50, 0.00),
        (-0.40, 0.45),
        (-0.40, -0.45),
    )

    agv_init_positions = (
        (support_offsets_xy[0][0], support_offsets_xy[0][1], agv_center_z),
        (support_offsets_xy[1][0], support_offsets_xy[1][1], agv_center_z),
        (support_offsets_xy[2][0], support_offsets_xy[2][1], agv_center_z),
    )
    agv_init_pos = agv_init_positions[0]

    # 目标点：第一版只做平地直线驮运。
    target_pos = (3.2, 0.0, 0.0)
    target_radius = 0.22
    workspace_limit = 5.0

    # ------------------------- 动作与运动学 -------------------------
    # 当前平地直线驮运阶段先限制动作空间，降低“倒车/原地大角度旋转”的探索幅度。
    max_agv_linear_speed = 0.28
    max_agv_angular_speed = 0.28

    # 平地直线驮运阶段先禁止倒车：action=-1 表示停止，action=1 表示最大前进。
    # 等直线驮运稳定后，再把该项改 False 恢复倒车能力。
    forward_only_linear_speed = True

    # 后侧线速度共享先验：当前平地直线驮运阶段，AGV2/AGV3 应保持近似相同纵向速度。
    # 这比仅靠 reward 惩罚更直接，可消除后侧两车一快一慢造成的拖拽感。
    tie_rear_linear_speed = True
    # 1.0 为硬共享；若后续需要恢复一定自由度，可改为 0.6~0.8 做软共享。
    tie_rear_linear_speed_blend = 1.0

    # V6.2.1 前车防抢跑：AGV1 的实际线速度命令最多比后侧两车平均命令领先 0.08。
    # 命令范围为 [0, 1]，在 max_agv_linear_speed=0.28 时相当于最多领先约 0.022 m/s。
    # 该约束保留少量前车调节自由度，但避免出现 0.220 对 0.145 m/s 的明显抢跑。
    limit_front_linear_speed = True
    max_front_command_lead = 0.08

    # V6.2.5 AGV1 位置闭环防抢跑。V6.2.4 诊断显示 AGV1 在 ep_step≈293 时
    # 纵向误差达到 +0.236 m，formation_error=0.241 m，刚好越过 0.24 m contact margin；
    # z_ok 始终为 True，因此不是高度问题，而是固定命令领先量长期积分成位置误差。
    enable_front_position_guard = True
    # 误差低于 0.17 m 时保留原策略自由度；超过后逐步压低 AGV1 速度。
    front_position_deadband = 0.17
    # normalized command / metre。误差每增加 0.05 m，AGV1 命令上限约再降低 0.10。
    front_position_kp = 2.0
    front_position_max_slowdown = 0.30
    # 接近 0.24 m 解析接触边界前进入硬恢复区。
    front_position_hard_limit = 0.22
    # 硬恢复时 AGV1 命令最多为后车均值减 0.10（约慢 0.028 m/s）。
    front_position_recovery_margin = 0.10

    # 左右侧 AGV 的转向角速度缩放。当前问题是侧车有向外侧逃逸趋势，
    # 先降低侧车转向自由度，等直线稳定后再逐步放开到 1.0。
    # 从 0.65 提升到 0.80：angular scale 降低未阻止偏航（行为与 v7 相似），
    # 反而限制了后侧 AGV 修正航向的能力。有了更强的 heading penalty 和
    # rear_heading_sync_penalty，PPO 已有足够动机保持航向，适度放开角速度自由度。
    side_agv_angular_scale = 0.80

    # V6.0 最小闭环采用 kinematic AGV 支撑台。PhysX 中 kinematic 物体平移时，
    # 动态 payload 不一定会被切向摩擦稳定带走，容易出现“车在下面滑、货物不动”。
    # 因此这里加入一个可关闭的虚拟摩擦/无滑移耦合项：只在至少两台 AGV 仍处于支撑
    # 区域时，把 payload 平面速度软耦合到支撑平台平均速度，并用滑移误差做小幅修正。
    # 后续若升级为真实轮式 articulation 或可产生真实切向摩擦的动态支撑平台，可关闭此项。
    enable_virtual_friction_carry = False
    # 软约束版：训练早期允许至少两车有效支撑时产生有限虚拟摩擦，
    # 但 reward 会通过 contact/formation/slip quality 鼓励最终三车稳定支撑。
    virtual_friction_min_contacts = 2.0
    virtual_friction_coupling = 0.90
    slip_correction_gain = 1.10
    max_payload_planar_speed = 0.40

    # Planar carry and pose stabilization are deliberately gated separately.
    # Two supports may still provide limited tangential transport, but they must
    # not receive artificial vertical/attitude damping.  Stabilization is only
    # allowed with all three analytical contacts and the Board CoM inside the
    # triangular support polygon.
    virtual_stabilization_min_contacts = 3.0
    virtual_stabilization_support_margin = 0.0
    payload_vertical_damping = 0.16
    payload_roll_pitch_damping = 0.22

    # Fix4：payload yaw 软稳定。
    # Fix3 解决了“车动、货不动”，但平动耦合不约束绕 z 轴自转。
    # 这些参数用于抑制货物持续 yaw spinning，并让货物航向缓慢对齐运输方向。
    payload_yaw_damping = 0.45
    payload_yaw_alignment_gain = 1.25
    payload_yaw_alignment_coupling = 0.65
    max_payload_yaw_rate = 0.35

    # 可选：模拟支撑面高度扰动。V6.0 默认关闭；V6.1 再打开。
    # 这不是完整轮-地坑洼模型，只是让 kinematic 支撑台 z 方向随位置变化，
    # 用于验证 payload 颠簸/姿态稳定 reward 是否有效。
    enable_bumpy_support = False
    # V6.2：仍保持轻度起伏，但加入相位，避免 y=0 中心线完全平坦，便于视觉观察。
    # 如果训练明显变差，先把 bump_amplitude 降回 0.020。
    bump_amplitude = 0.030
    bump_wavelength_x = 1.40
    bump_wavelength_y = 1.10
    bump_phase_x = 0.25
    bump_phase_y = 0.45

    # 可视化崎岖地形 mesh。该 mesh 主要用于显示，不作为真实轮地碰撞地形。
    # AGV 的运动高度仍由 agv_carry_env.py::_terrain_height() 决定。
    enable_visual_terrain_mesh = True
    visual_terrain_x_min = -0.80
    visual_terrain_x_max = 3.90
    visual_terrain_y_min = -1.70
    visual_terrain_y_max = 1.70
    visual_terrain_grid_x = 96
    visual_terrain_grid_y = 70
    # 1.0 表示视觉高度与训练高度一致；如果只为录视频想更明显，可临时改成 1.5~2.0。
    visual_terrain_height_scale = 1.0
    visual_terrain_z_offset = 0.0
    visual_terrain_color = (0.42, 0.36, 0.28)
    # 将默认 ground plane 放低，避免遮挡 mesh 的负高度谷底。
    visual_terrain_ground_z = -0.08

    # ------------------------- 接触/支撑判定 -------------------------
    # 解析接触判定：AGV 是否位于对应支撑目标附近，且 AGV 顶面接近 payload 底面。
    # 解析支撑判定不能过窄，否则训练早期几乎没有正反馈；
    # 也不能回到过宽，否则会出现视觉脱离但 contact_flag 仍为 True。
    support_contact_xy_margin = 0.24
    support_contact_z_margin = 0.095
    # grace 180→120：缩短宽限期，支撑丢失更快被惩罚，减少 min reward 暴跌幅度。
    support_loss_grace_steps = 160

    # 轻度崎岖矩形货物阶段：训练过程仍鼓励三车共同承载，
    # 但终点成功判定允许瞬时解析 contact 从 3 抖动到 2。
    # 这是为了避免“已到达目标且姿态稳定，但终止瞬间某一车 contact flag 短时掉出 margin”被误判失败。
    required_support_contacts = 3.0
    critical_support_contacts = 2.0

    # V6.1.1 rough success relax：成功判定要求至少两车有效支撑，
    # 同时仍保留 slip、roll/pitch、support_margin、support_lost/drop/tip/oob 等约束。
    success_required_support_contacts = 2.0
    success_support_margin_min = -0.02

    # payload 相对三车支撑结构的允许滑移。
    slip_success_threshold = 0.22
    slip_penalty_threshold = 0.16

    # payload CoM 投影到三角支撑多边形内的安全裕度。
    # 奖励和成功条件都会引用该参数；缺失会导致 AttributeError。
    support_polygon_min_margin = 0.04

    # 姿态稳定阈值：当前阶段只约束 roll/pitch，不把 yaw 作为成功或奖励目标。
    stable_roll_pitch_radius = 0.12
    tip_roll_pitch_threshold = 0.50
    payload_min_z = 0.18

    # ------------------------- 奖励权重 -------------------------
    # 软约束版奖励：训练过程用连续质量因子引导，成功条件仍严格要求三车支撑。
    progress_reward_scale = 18.0
    # distance_penalty 设为 0：-scale*goal_dist 产生恒定负偏置，淹没 progress 信号。
    # progress reward 已经捕获了距离缩减，不需要额外的距离惩罚。
    distance_penalty_scale = 0.0
    payload_speed_reward_scale = 0.65
    support_drive_reward_scale = 0.85
    # success_reward 从 100 降到 50：减少 reward cliff，降低价值函数预测范围。
    success_reward_scale = 50.0

    # 显式打破”停车少扣分”的局部最优。
    # per-AGV 检查：任一 AGV cmd < min_forward_cmd 即触发惩罚。
    stationary_speed_threshold = 0.035
    stationary_penalty_scale = 0.06
    min_forward_cmd = 0.25
    low_forward_cmd_penalty_scale = 0.08

    # 命令级速度同步惩罚：使用 [0, 1] 的 normalized command，而不是 m/s 速度差。
    # 物理速度差平方在低速阶段量级太小，之前的速度同步 reward 基本压不住策略。
    rear_cmd_sync_penalty_scale = 0.40
    all_cmd_sync_penalty_scale = 0.10
    # 只惩罚策略 raw command 超过允许前车领先量的部分；执行层限速才是主约束。
    front_command_lead_penalty_scale = 0.20

    # soft transport quality = contact_quality * exp(-k_f * formation_error) * exp(-k_s * slip_error)。
    # support_drive_reward 给 AGV 支撑结构整体向目标移动提供早期正反馈。
    formation_quality_gain = 3.0
    slip_quality_gain = 5.0

    # 支撑几何约束。deadband 内不惩罚，明显脱离各自支撑点才惩罚。
    missing_support_penalty_scale = 0.05
    formation_error_penalty_scale = 0.25
    formation_error_deadband = 0.18
    formation_warmup_steps = 150
    formation_warmup_scale = 0.30

    # 侧向约束：只惩罚 AGV2/AGV3 相对各自支撑点继续向外侧逃逸，
    # 避免普通 formation 惩罚过强导致策略重新回到原地不动。
    side_lateral_deadband = 0.06
    side_outward_penalty_scale = 1.20
    side_lateral_velocity_penalty_scale = 0.15

    # support polygon 暂时不进入 dense reward；保留在 success/debug 中。
    # 如后续恢复，必须使用限幅后的 violation，不能直接平方原始 margin。
    support_polygon_penalty_scale = 0.0

    # 承载稳定约束。V6.1 增加轻度崎岖支撑面的垂向速度和 z gap 监控。
    roll_pitch_penalty_scale = 5.0
    payload_vertical_velocity_threshold = 0.10
    vertical_velocity_penalty_scale = 0.60
    support_z_gap_penalty_deadband = 0.060
    support_z_gap_penalty_scale = 0.80
    slip_penalty_scale = 1.5

    # 动作约束：不惩罚线速度，显式惩罚角速度和车头偏离，抑制原地乱转。
    linear_action_penalty_scale = 0.0
    angular_action_penalty_scale = 0.06
    action_rate_penalty_scale = 0.012
    # heading penalty 从 0.04 提升到 0.30：原来太弱，0.3 rad 偏航时惩罚仅 0.0012/step，
    # 而 progress 奖励约 0.054/step，PPO 完全没有动力保持航向。
    agv_heading_penalty_scale = 0.30

    # 后侧 AGV 同向偏航惩罚：专治"两后车均朝左前方"的失效模式。
    # side_outward_penalty 只能捕获 AGV2 向左外扩和 AGV3 向右外扩，
    # 但当两车同时左偏时，AGV3 的左偏是"内向"的，outward penalty 完全不触发。
    # 此项直接惩罚后两车相对 move_dir 的偏航之和的平方：
    # 同向偏航 → 和值大 → 重罚；异向偏航（正常转弯）→ 和值≈0 → 不罚。
    rear_heading_sync_penalty_scale = 0.10

    # 终端惩罚从 80/80/50 降到 25/25/15：原来相对每步 shaping(~0.07) 是 1000x 级别，
    # 导致梯度爆炸和价值函数不稳定。25/25/15 仍在每步 shaping 的合理倍数范围内。
    # support_loss 10→5：减少 min reward -200 级别暴跌，仍能有效 discouraging 支撑丢失。
    support_loss_penalty_scale = 5.0
    out_of_bounds_penalty = 15.0
    drop_penalty = 25.0
    tip_penalty = 25.0

    # ------------------------- 场景对象 -------------------------
    _agv_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=1.25,
        dynamic_friction=1.10,
        restitution=0.0,
    )
    _payload_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=1.10,
        dynamic_friction=0.95,
        restitution=0.0,
    )
    _cargo_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=cargo_static_friction,
        dynamic_friction=cargo_dynamic_friction,
        restitution=0.0,
    )
    _lift_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=1.30,
        dynamic_friction=1.10,
        restitution=0.0,
    )

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
            physics_material=_agv_material,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.05), metallic=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=agv_init_positions[0], rot=(1.0, 0.0, 0.0, 0.0)),
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
            physics_material=_agv_material,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.05), metallic=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=agv_init_positions[1], rot=(1.0, 0.0, 0.0, 0.0)),
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
            physics_material=_agv_material,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.05), metallic=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=agv_init_positions[2], rot=(1.0, 0.0, 0.0, 0.0)),
    )

    lift_init_z = agv_top_z + lift_neutral_height + 0.5 * lift_plate_size[2]

    lift1_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Lift1",
        spawn=sim_utils.CuboidCfg(
            size=lift_plate_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=lift_plate_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_lift_material,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.45, 0.85), metallic=0.15),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(support_offsets_xy[0][0], support_offsets_xy[0][1], lift_init_z),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    lift2_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Lift2",
        spawn=sim_utils.CuboidCfg(
            size=lift_plate_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=lift_plate_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_lift_material,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.45, 0.85), metallic=0.15),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(support_offsets_xy[1][0], support_offsets_xy[1][1], lift_init_z),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    lift3_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Lift3",
        spawn=sim_utils.CuboidCfg(
            size=lift_plate_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=lift_plate_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_lift_material,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.45, 0.85), metallic=0.15),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(support_offsets_xy[2][0], support_offsets_xy[2][1], lift_init_z),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    payload_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Payload",
        spawn=sim_utils.CuboidCfg(
            size=payload_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=payload_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_payload_material,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.55, 0.0), metallic=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=payload_init_pos, rot=(1.0, 0.0, 0.0, 0.0)),
    )

    cargo_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cargo",
        spawn=sim_utils.CuboidCfg(
            size=cargo_size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=cargo_mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=_cargo_material,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.10, 0.65, 0.95), metallic=0.0, roughness=0.35
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=cargo_init_pos, rot=(1.0, 0.0, 0.0, 0.0)),
    )
