
from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class long_config(LeggedRobotCfg):
    """
    Configuration class for the XBotL humanoid robot.
    """
    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 15
        c_frame_stack = 3
        num_single_obs = 47
        # num_single_obs = 41
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 73        
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 12        
        num_envs = 4096
        episode_length_s = 24  # episode length in seconds
        use_ref_actions = False

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85

    class asset(LeggedRobotCfg.asset):
        file = ('../../resources/robots/long/urdf/Robot_12_V02.urdf')
        name = "Robot_12_v02"
        foot_name = "ankle_roll"       
        knee_name = "knee"
        terminate_after_contacts_on = ['base_link']
        penalize_contacts_on = ["base_link"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        # mesh_type = 'heightfield'
        # mesh_type = 'trimesh'
        curriculum = False
        # rough terrain only:
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        restitution = 0.

    class noise:
        add_noise = True
        noise_level = 0.6    # scales other values
        class noise_scales:
            dof_pos = 0.05
            dof_vel = 0.5
            ang_vel = 0.1
            lin_vel = 0.05
            quat = 0.03
            height_measurements = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 1.09]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'J_hip_l_roll': 0.,
            'J_hip_l_yaw': 0.,
            'J_hip_l_pitch': 0.,
            'J_knee_pitch_l': 0.,
            'J_ankle_pitch_l': 0.,
            'J_ankle_roll_l': 0.,
            'J_hip_r_roll': 0.,
            'J_hip_r_yaw': 0.,
            'J_hip_r_pitch': 0.,
            'J_knee_pitch_r': 0.,
            'J_ankle_pitch_r': 0.,
            'J_ankle_roll_r': 0.,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {
            'J_hip_l_roll': 100.,
            'J_hip_l_yaw': 100.,
            'J_hip_l_pitch': 120.,
            'J_knee_pitch_l': 160.,
            'J_ankle_pitch_l': 12.,
            'J_ankle_roll_l': 12.,
            'J_hip_r_roll': 100.,
            'J_hip_r_yaw': 100.,
            'J_hip_r_pitch': 120.,
            'J_knee_pitch_r': 160.,
            'J_ankle_pitch_r': 12.,
            'J_ankle_roll_r': 12.,

        }
        damping = {
            'J_hip_l_roll': 10,
            'J_hip_l_yaw': 10,
            'J_hip_l_pitch': 12,
            'J_knee_pitch_l': 16.,
            'J_ankle_pitch_l': 1.2,
            'J_ankle_roll_l': 1.2,
            'J_hip_r_roll': 10,
            'J_hip_r_yaw': 10,
            'J_hip_r_pitch': 12,
            'J_knee_pitch_r': 16.,
            'J_ankle_pitch_r': 1.2,
            'J_ankle_roll_r': 1.2,
        }
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 1.0
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # 100hz

    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 1000 Hz
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z
        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0 #ԭ��ֵ��1
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.1  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand:
        randomize_friction = True
        friction_range = [0.1, 2.0]
        randomize_base_mass = True
        added_mass_range = [-5.0, 5.0]
        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 0.4
        max_push_ang_vel = 0.4
        dynamic_randomization = 0.02
		# dynamic randomization
        # action_delay = 0.5
        # action_noise = 0.02

    class commands(LeggedRobotCfg.commands):
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 8.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-0.3, 0.6]  # min max [m/s]
            lin_vel_y = [-0.3, 0.3]   # min max [m/s]
            ang_vel_yaw = [-0.3, 0.3]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards:
        base_height_target = 1.0
        min_dist = 0.2
        max_dist = 0.5
        # min_dist = 0.3
        # max_dist = 0.8
        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.17    # rad
        target_feet_height = 0.06       # m
        cycle_time = 0.64                # sec
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = False
        # tracking reward = exp(error*sigma)
        tracking_sigma = 5
        max_contact_force = 1000  # Forces above this value are penalized

        class scales:
            # reference motion tracking
            joint_pos = 1.6
            feet_clearance = 1.
            feet_contact_number = 1.2
            # gait
            feet_air_time = 1.
            foot_slip = -0.05
            feet_distance = 0.2
            knee_distance = 0.2
            # contact
            feet_contact_forces = -0.01
            # vel tracking
            tracking_lin_vel = 1.2
            tracking_ang_vel = 1.1
            vel_mismatch_exp = 0.5  # lin_z; ang x,y
            low_speed = 0.2
            track_vel_hard = 0.5
            # base pos
            default_joint_pos = 0.5
            orientation = 1.
            base_height = 0.2
            base_acc = 0.2
            # energy
            action_smoothness = -0.002
            torques = -1e-5
            dof_vel = -5e-4
            dof_acc = -1e-7
            collision = -1.

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1.
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
        clip_observations = 18.
        clip_actions = 18.

class long_configPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = 'OnPolicyRunner'   # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 60  # per iteration
        max_iterations = 30001  # number of policy updates
        # logging
        save_interval = 1000  # Please check for potential savings every `save_interval` iterations.
        experiment_name = 'long_ppo'
        # experiment_name = 'long_ppo_10joint'
        run_name = ''
        # Load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
