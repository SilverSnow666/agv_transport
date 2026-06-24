from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


@configclass
class AgvCarryEnvCfg(DirectRLEnvCfg):
    """V6.0：三 AGV 无刚性连接协同驮运任务配置。

    设计目标：先做最小闭环，不直接上坑洼路面和异形件。
    三台 kinematic AGV 作为移动支撑平台，payload 为动态刚体，靠重力、接触和摩擦
    放置在三车顶部。训练目标是把 payload 平稳送到目标点，同时限制 roll/pitch、
    滑移、支撑丢失和三车队形失稳。

    注意：这是“驮运/承载”新任务，不应覆盖 V5.3.3 的推送任务。
    """

    # ------------------------- 基本仿真设置 -------------------------
    decimation = 2
    episode_length_s = 18.0
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

    # 三台 AGV 是移动支撑台。第一版仍使用 kinematic cuboid，降低 wheel-articulation 调试成本。
    agv_size = (0.55, 0.42, 0.16)
    agv_mass = 30.0
    agv_center_z = 0.08
    agv_top_z = agv_center_z + 0.5 * agv_size[2]

    # payload 是被三车共同承载的动态刚体。
    payload_size = (1.20, 0.95, 0.18)
    payload_mass = 22.0
    payload_init_z = agv_top_z + 0.5 * payload_size[2] + 0.012
    payload_init_pos = (0.0, 0.0, payload_init_z)

    # 支撑三角形：local_x 为运输方向，local_y 为横向。
    # AGV1 在前中，AGV2 在后左，AGV3 在后右。三点都位于 payload 投影内。
    support_offsets_xy = (
        (0.38, 0.00),
        (-0.34, 0.42),
        (-0.34, -0.42),
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
    max_agv_linear_speed = 0.45
    max_agv_angular_speed = 1.40


    # V6.0 最小闭环采用 kinematic AGV 支撑台。PhysX 中 kinematic 物体平移时，
    # 动态 payload 不一定会被切向摩擦稳定带走，容易出现“车在下面滑、货物不动”。
    # 因此这里加入一个可关闭的虚拟摩擦/无滑移耦合项：只在至少两台 AGV 仍处于支撑
    # 区域时，把 payload 平面速度软耦合到支撑平台平均速度，并用滑移误差做小幅修正。
    # 后续若升级为真实轮式 articulation 或可产生真实切向摩擦的动态支撑平台，可关闭此项。
    enable_virtual_friction_carry = True
    virtual_friction_min_contacts = 2.0
    virtual_friction_coupling = 0.85
    slip_correction_gain = 1.35
    max_payload_planar_speed = 0.55
    payload_vertical_damping = 0.12
    payload_roll_pitch_damping = 0.18

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
    bump_amplitude = 0.025
    bump_wavelength_x = 1.20
    bump_wavelength_y = 1.00

    # ------------------------- 接触/支撑判定 -------------------------
    # 解析接触判定：AGV 是否位于对应支撑目标附近，且 AGV 顶面接近 payload 底面。
    support_contact_xy_margin = 0.24
    support_contact_z_margin = 0.075
    support_loss_grace_steps = 80

    # payload 相对三车支撑结构的允许滑移。
    slip_success_threshold = 0.12
    slip_penalty_threshold = 0.08

    # payload CoM 投影到三角支撑多边形的安全裕度。
    support_polygon_min_margin = 0.04

    # 姿态稳定阈值。
    stable_roll_pitch_radius = 0.10
    stable_yaw_radius = 0.25
    tip_roll_pitch_threshold = 0.45
    payload_min_z = 0.18

    # ------------------------- 奖励权重 -------------------------
    progress_reward_scale = 9.0
    success_reward_scale = 120.0
    distance_penalty_scale = 0.6

    roll_pitch_penalty_scale = 10.0
    yaw_alignment_penalty_scale = 2.5
    angular_velocity_penalty_scale = 0.30
    yaw_rate_penalty_scale = 1.5
    vertical_velocity_penalty_scale = 0.60

    support_contact_reward_scale = 2.0
    support_loss_penalty_scale = 6.0
    formation_error_penalty_scale = 2.5
    slip_penalty_scale = 8.0
    support_polygon_penalty_scale = 8.0

    action_penalty_scale = 0.025
    action_rate_penalty_scale = 0.08
    out_of_bounds_penalty = 50.0
    drop_penalty = 80.0
    tip_penalty = 80.0

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
