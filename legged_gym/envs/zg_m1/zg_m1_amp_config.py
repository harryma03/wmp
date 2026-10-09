import glob

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


MOTION_FILES = glob.glob("datasets/mocap_motions/*")


class ZGM1AMPCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        include_history_steps = None
        prop_dim = 41
        action_dim = 16
        # base velocity and domain/contact data before proprioception
        privileged_dim = 29
        height_dim = 187
        forward_height_dim = 525
        num_actions = 16
        num_observations = privileged_dim + prop_dim + action_dim + height_dim
        num_privileged_obs = num_observations
        reference_state_initialization = False
        reference_state_initialization_prob = 0.85
        amp_motion_files = MOTION_FILES

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "trimesh"
        horizontal_scale = 0.1
        vertical_scale = 0.005
        border_size = 25
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.0
        measure_heights = True
        measured_points_x = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        measured_points_y = [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        measured_forward_points_x = [0.1 * i for i in range(21)]
        measured_forward_points_y = [-1.2 + 0.1 * i for i in range(25)]
        selected = False
        terrain_kwargs = None
        max_init_terrain_level = 0
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10
        num_cols = 20
        terrain_proportions = [0.0, 0.05, 0.15, 0.15, 0.0, 0.25, 0.25, 0.05, 0.05, 0.05]
        slope_treshold = 0.75

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.62]
        default_joint_angles = {
            "FAR_ABAD_JOINT": 0.0,
            "FAR_HIP_JOINT": 0.8,
            "FAR_KNEE_JOINT": -1.5,
            "FAR_FOOT_JOINT": 0.0,
            "FBL_ABAD_JOINT": 0.0,
            "FBL_HIP_JOINT": 0.8,
            "FBL_KNEE_JOINT": -1.5,
            "FBL_FOOT_JOINT": 0.0,
            "RAR_ABAD_JOINT": 0.0,
            "RAR_HIP_JOINT": -0.8,
            "RAR_KNEE_JOINT": 1.5,
            "RAR_FOOT_JOINT": 0.0,
            "RBL_ABAD_JOINT": 0.0,
            "RBL_HIP_JOINT": -0.8,
            "RBL_KNEE_JOINT": 1.5,
            "RBL_FOOT_JOINT": 0.0,
        }

    class sim(LeggedRobotCfg.sim):
        dt = 0.005
        substeps = 1

        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23
            default_buffer_size_multiplier = 5
            contact_collection = 2

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {
            "ABAD_JOINT": 60.0,
            "HIP_JOINT": 120.0,
            "KNEE_JOINT": 120.0,
            "FOOT_JOINT": 0.0,
        }
        damping = {
            "ABAD_JOINT": 1.0,
            "HIP_JOINT": 1.0,
            "KNEE_JOINT": 1.0,
            "FOOT_JOINT": 0.5,
        }
        leg_action_scale = 0.25
        wheel_action_scale = 10.0
        action_scale = 0.25
        decimation = 4

    class depth(LeggedRobotCfg.depth):
        use_camera = True
        camera_num_envs = 1024
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20
        # Approximate optical center measured from BASE_LINK in the robot model.
        position = [0.4243, 0.0, 0.03035]
        y_angle = [-5, 5]
        z_angle = [0, 0]
        x_angle = [0, 0]
        update_interval = 5
        original = (64, 64)
        resized = (64, 64)
        horizontal_fov = 58
        buffer_len = 2
        near_clip = 0
        far_clip = 2
        dis_noise = 0.0
        scale = 1
        invert = True

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/zg_m1/urdf/zg_m1_urdf.urdf"
        foot_name = "FOOT_LINK"
        penalize_contacts_on = ["HIP_LINK", "KNEE_LINK"]
        terminate_after_contacts_on = ["BASE_LINK"]
        self_collisions = 1

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 2.0]
        randomize_restitution = True
        restitution_range = [0.0, 0.0]
        randomize_base_mass = True
        added_mass_range = [0.0, 3.0]
        randomize_link_mass = True
        link_mass_range = [0.8, 1.2]
        randomize_com_pos = True
        com_x_pos_range = [-0.05, 0.05]
        com_y_pos_range = [-0.05, 0.05]
        com_z_pos_range = [-0.05, 0.05]
        push_robots = True
        push_interval_s = 15
        min_push_interval_s = 15
        max_push_vel_xy = 1.0
        # FOOT_JOINT has Kp=0; the base privileged gain ratio divides by Kp.
        randomize_gains = False
        randomize_motor_strength = True
        motor_strength_range = [0.8, 1.2]
        randomize_action_latency = True
        latency_range = [0.0, 0.005]

    class normalization(LeggedRobotCfg.normalization):
        clip_observations = 100.0
        clip_actions = 6.0
        base_height = 0.57

    class noise(LeggedRobotCfg.noise):
        add_noise = False

    class rewards(LeggedRobotCfg.rewards):
        reward_curriculum = True
        reward_curriculum_term = ["lin_vel_z"]
        reward_curriculum_schedule = [[4000, 10000, 0.1, 1.0]]
        soft_dof_pos_limit = 0.9
        base_height_target = 0.57
        tracking_sigma = 0.15
        lin_vel_clip = 0.1

        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.5
            torques = -0.0001
            dof_acc = -2.5e-7
            collision = -1.0
            action_rate = -0.03
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -0.2
            feet_air_time = 0.0
            feet_stumble = 0.0
            stand_still = 0.0

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_lin_vel_forward_x_curriculum = 1.0
        max_lin_vel_backward_x_curriculum = 0.0
        max_lin_vel_y_curriculum = 0.0
        max_ang_vel_yaw_curriculum = 1.0
        max_flat_lin_vel_forward_x_curriculum = 1.0
        max_flat_lin_vel_backward_x_curriculum = 0.0
        max_flat_lin_vel_y_curriculum = 0.0
        max_flat_ang_vel_yaw_curriculum = 1.0
        num_commands = 4
        resampling_time = 10.0
        heading_command = True

        class ranges:
            lin_vel_x = [0.0, 0.8]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [0.0, 0.0]
            flat_lin_vel_x = [0.0, 0.8]
            flat_lin_vel_y = [0.0, 0.0]
            flat_ang_vel_yaw = [-1.0, 1.0]
            flat_heading = [-3.14 / 4, 3.14 / 4]


class ZGM1AMPCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = "WMPRunner"

    class policy:
        init_noise_std = 1.0
        encoder_hidden_dims = [256, 128]
        wm_encoder_hidden_dims = [64, 64]
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [512, 256, 128]
        latent_dim = 35
        wm_latent_dim = 32
        activation = "elu"

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        vel_predict_coef = 1.0
        amp_replay_buffer_size = 1000000
        num_learning_epochs = 5
        num_mini_batches = 4

    class runner(LeggedRobotCfgPPO.runner):
        run_name = "WMP"
        experiment_name = "zg_m1_wmp"
        algorithm_class_name = "AMPPPO"
        use_amp = False
        policy_class_name = "ActorCritic"
        max_iterations = 100000
        save_interval = 1000
        amp_reward_coef = 0.0
        amp_motion_files = MOTION_FILES
        amp_num_preload_transitions = 1
        amp_task_reward_lerp = 0.3
        amp_discr_hidden_dims = [1024, 512]
        min_std_is_absolute = True
        min_normalized_std = [0.05, 0.02, 0.05, 0.10] * 4

    class depth_predictor:
        lr = 3e-4
        weight_decay = 1e-4
        training_interval = 10
        training_iters = 1000
        batch_size = 1024
        loss_scale = 100

