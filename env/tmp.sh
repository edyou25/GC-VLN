=========================================
GC-VLN R2R 50-episode smoke eval
episodes: 50
config: config/r2r_vlnce.yaml
experiment: r2r_50_20260917-190953
=========================================
DATASET: # Dataset settings
  CONTENT_SCENES: ['*']
  DATASET_TYPE: r2r
  DATA_PATH: data/datasets/R2R_VLNCE_v1-2_preprocessed/{split}/{split}.json.gz
  EPISODES_ALLOWED: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
  EXPERIMENT_ID: r2r_50_20260917-190953
  LANGUAGES: ['*']
  ROLES: ['guide']
  SCENES_DIR: data/scene_datasets/
  SPLIT: val_unseen
  SPLIT_INDEX: 0
  SPLIT_NUM: 1
  TYPE: VLN-CE-v1
ENVIRONMENT:
  ITERATOR_OPTIONS:
    CYCLE: True 
    GROUP_BY_SCENE: True
    MAX_SCENE_REPEAT_EPISODES: -1
    MAX_SCENE_REPEAT_STEPS: -1
    NUM_EPISODE_SAMPLE: -1
    SHUFFLE: False
    STEP_REPETITION_RANGE: 0.2
  MAX_EPISODE_SECONDS: 10000000
  MAX_EPISODE_STEPS: 5000
ENV_NAME: GCVLNEnv
EVAL:
  EPISODE_COUNT: -1
  SAVE_RESULTS: True
GPU_NUMBERS: 1
NUM_ENVIRONMENTS: 1
POLICY_CONFIG:
  GSAM2_SERVER_PORT: 7000
  MAP:
    AGENT_HEIGHT: 17.6
    DEPTH_MAX: 10.0 # sam mask + depth -> pt/bev
    DEPTH_MIN: 0.1
    MASK_CAPTIONS: ['stair', 'stairs', 'stairstep', 'step', 'steps', 'staircase', 'staircases', 'stairsteps']
    PASS_THRESHOLD: 5 # point 2 obstcle
    RESOLUTION: 20 # 20 grid/m
    SCALE: 1 # depth -> pt/bev
    SIZE: 1000 # BEV Tensor
  RS: # 语言约束+BEV约束
    DETECTAREA: 200 # 10m
    DIS_INTERVAL: 7 # 
    EXP_THRESHOLD_ONE_STAGE: 2 # 检测次数
    MAXDISTANCE: 20
    MIN_REGION_THRESHOLD: 150
    NUM_DETECTIONS_THRESHOLD: 2
    RADIUS: 15
    RANDOM_EXP_TIME: 3
    RANDOM_RADIUS: 30
    SAFEDISTANCE: 5
    TYPEWEIGHT: 2
  SCENEGRAPH: # BEV约束
    BOX_THRESHOLD: 0.3 # dino阈值
    DOORWAY_MAX_DEPTH_DIFF: 0.5 # 特殊阈值
    GROUNDING_DINO_CHECKPOINT: ./Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth
    GROUNDING_DINO_CONFIG: ./Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py
    SAM2_CHECKPOINT: ./Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt
    SAM2_MODEL_CONFIG: configs/sam2.1/sam2.1_hiera_l.yaml
    TEXT_PROMPT: chair.
    TEXT_THRESHOLD: 0.25
    VAR_THRESHOLD: 3
  UNI:
    AGENT_HEIGHT: 0.88
    CAMERA_HEIGHT: 640
    CAMERA_WIDTH: 480
    DEVICE: cuda
    MAX_RRAJ_LEN: 50
    TELEFLAG: True
    THIN_TYPE: 1 # BEV骨架化
    VIDEO_DIR: data/logs/video/
    VIDEO_OPTION: []
PYROBOT: # 仿真机器人配置
  BASE_CONTROLLER: proportional
  BASE_PLANNER: none
  BUMP_SENSOR:
    TYPE: PyRobotBumpSensor
  DEPTH_SENSOR:
    CENTER_CROP: False
    HEIGHT: 480
    MAX_DEPTH: 5.0
    MIN_DEPTH: 0.0
    NORMALIZE_DEPTH: True
    TYPE: PyRobotDepthSensor
    WIDTH: 640
  LOCOBOT:
    ACTIONS: ['BASE_ACTIONS', 'CAMERA_ACTIONS']
    BASE_ACTIONS: ['go_to_relative', 'go_to_absolute']
    CAMERA_ACTIONS: ['set_pan', 'set_tilt', 'set_pan_tilt']
  RGB_SENSOR:
    CENTER_CROP: False
    HEIGHT: 480
    TYPE: PyRobotRGBSensor
    WIDTH: 640
  ROBOT: locobot
  ROBOTS: ['locobot']
  SENSORS: ['RGB_SENSOR', 'DEPTH_SENSOR', 'BUMP_SENSOR']
RESULTS_DIR: data/logs/eval_results/
SENSORS: ['RGB_SENSOR', 'DEPTH_SENSOR', 'RGB_30', 'RGB_60', 'RGB_90', 'RGB_120', 'RGB_150', 'RGB_180', 'RGB_210', 'RGB_240', 'RGB_270', 'RGB_300', 'RGB_330', 'DEPTH_30', 'DEPTH_60', 'DEPTH_90', 'DEPTH_120', 'DEPTH_150', 'DEPTH_180', 'DEPTH_210', 'DEPTH_240', 'DEPTH_270', 'DEPTH_300', 'DEPTH_330']
SIMULATOR:
  ACTION_SPACE_CONFIG: v0
  ADDITIONAL_OBJECT_PATHS: []
  AGENTS: ['AGENT_0']
  AGENT_0:
    HEIGHT: 1.5
    IS_SET_START_STATE: False
    RADIUS: 0.18
    SENSORS: ['RGB_SENSOR', 'DEPTH_SENSOR', 'RGB_30', 'RGB_60', 'RGB_90', 'RGB_120', 'RGB_150', 'RGB_180', 'RGB_210', 'RGB_240', 'RGB_270', 'RGB_300', 'RGB_330', 'DEPTH_30', 'DEPTH_60', 'DEPTH_90', 'DEPTH_120', 'DEPTH_150', 'DEPTH_180', 'DEPTH_210', 'DEPTH_240', 'DEPTH_270', 'DEPTH_300', 'DEPTH_330']
    START_POSITION: [0, 0, 0]
    START_ROTATION: [0, 0, 0, 1]
  ARM_DEPTH_SENSOR:
    HEIGHT: 480
    HFOV: 90
    MAX_DEPTH: 10.0
    MIN_DEPTH: 0.0
    NORMALIZE_DEPTH: True
    ORIENTATION: [0.0, 0.0, 0.0]
    POSITION: [0, 1.25, 0]
    SENSOR_SUBTYPE: PINHOLE
    TYPE: HabitatSimDepthSensor
    UUID: robot_arm_depth
    WIDTH: 640
  ARM_RGB_SENSOR:
    HEIGHT: 480
    HFOV: 90
    ORIENTATION: [0.0, 0.0, 0.0]
    POSITION: [0, 1.25, 0]
    SENSOR_SUBTYPE: PINHOLE
    TYPE: HabitatSimRGBSensor
    UUID: robot_arm_rgb
    WIDTH: 640
  DEFAULT_AGENT_ID: 0
  DEPTH_120:
    HEIGHT: 480
    HFOV: 90
    MAX_DEPTH: 10.0
    MIN_DEPTH: 0.1
    NORMALIZE_DEPTH: False
    ORIENTATION: [0, -2.0943951023931953, 0]
    POSITION: [0, 0.88, 0]
    SENSOR_SUBTYPE: PINHOLE
    TYPE: HabitatSimDepthSensor
    UUID: depth_120
    WIDTH: 640
  DEPTH_150:
    HEIGHT: 480
    HFOV: 90
    MAX_DEPTH: 10.0
    MIN_DEPTH: 0.1
    NORMALIZE_DEPTH: False
    ORIENTATION: [0, -2.617993877991494, 0]
    POSITION: [0, 0.88, 0]
    SENSOR_SUBTYPE: PINHOLE
    TYPE: HabitatSimDepthSensor
    UUID: depth_150
    WIDTH: 640
  DEPTH_180:
    HEIGHT: 480
    HFOV: 90
    MAX_DEPTH: 10.0
    MIN_DEPTH: 0.1
    NORMALIZE_DEPTH: False
    ORIENTATION: [0, -3.141592653589793, 0]
    POSITION: [0, 0.88, 0]
    SENSOR_SUBTYPE: PINHOLE
    TYPE: HabitatSimDepthSensor
    UUID: depth_180
    WIDTH: 640
  DEPTH_210:
    HEIGHT: 480
    HFOV: 90
    MAX_DEPTH: 10.0
    MIN_DEPTH: 0.1
    NORMALIZE_DEPTH: False
    ORIENTATION: [0, -3.665191429188092, 0]
    POSITION: [0, 0.88, 0]
    SENSOR_SUBTYPE: PINHOLE
    TYPE: HabitatSimDepthSensor
    UUID: depth_210
    WIDTH: 640
  DEPTH_240:
    HEIGHT: 480
    HFOV: 90
    MAX_DEPTH: 10.0
    MIN_DEPTH: 0.1
    NORMALIZE_DEPTH: False
    ORIENTATION: [0, -4.1887902047863905, 0]
    POSITION: [0, 0.88, 0]
    SENSOR_SUBTYPE: PINHOLE
    TYPE: HabitatSimDepthSensor
    UUID: depth_240
    WIDTH: 640
  DEPTH_270:
    HEIGHT: 480
    HFOV: 90
    MAX_DEPTH: 10.0
    MIN_DEPTH: 0.1
    NORMALIZE_DEPTH: False
    ORIENTATION: [0, -4.71238898038469, 0]
    POSITION: [0, 0.88, 0]
    SENSOR_SUBTYPE: PINHOLE
    TYPE: HabitatSimDepthSensor
    UUID: depth_270
    WIDTH: 640
  DEPTH_30:
    HEIGHT: 480
    HFOV: 90
    MAX_DEPTH: 10.0
    MIN_DEPTH: 0.1
    NORMALIZE_DEPTH: False
    ORIENTATION: [0, -0.5235987755982988, 0]
    POSITION: [0, 0.88, 0]
    SENSOR_SUBTYPE: PINHOLE
    TYPE: HabitatSimDepthSensor
    UUID: depth_30
    WIDTH: 640
  DEPTH_300:
    HEIGHT: 480
    HFOV: 90
    MAX_DEPTH: 10.0
    MIN_DEPTH: 0.1
    NORMALIZE_DEPTH: False
    ORIENTATION: [0, -5.235987755982988, 0]
    POSITION: [0, 0.88, 0]
    SENSOR_SUBTYPE: PINHOLE
    TYPE: HabitatSimDepthSensor
    UUID: depth_300
    WIDTH: 640
  DEPTH_330:
    HEIGHT: 480
    HFOV: 90
    MAX_DEPTH: 10.0
    MIN_DEPTH: 0.1
    NORMALIZE_DEPTH: False
    ORIENTATION: [0, -5.759586531581287, 0]
    POSITION: [0, 0.88, 0]
    SENSOR_SUBTYPE: PINHOLE
    TYPE: HabitatSimDepthSensor
    UUID: depth_330
    WIDTH: 640
  DEPTH_60:
    HEIGHT: 480
    HFOV: 90
    MAX_DEPTH: 10.0
    MIN_DEPTH: 0.1
    NORMALIZE_DEPTH: False
    ORIENTATION: [0, -1.0471975511965976, 0]
    POSITION: [0, 0.88, 0]
    SENSOR_SUBTYPE: PINHOLE
    TYPE: HabitatSimDepthSensor
    UUID: depth_60
    WIDTH: 640
  DEPTH_90:
    HEIGHT: 480
    HFOV: 90
    MAX_DEPTH: 10.0
    MIN_DEPTH: 0.1
    NORMALIZE_DEPTH: False
    ORIENTATION: [0, -1.5707963267948966, 0]
    POSITION: [0, 0.88, 0]
    SENSOR_SUBTYPE: PINHOLE
    TYPE: HabitatSimDepthSensor
    UUID: depth_90
    WIDTH: 640
  DEPTH_SENSOR:
    HEIGHT: 480
    HFOV: 90
    MAX_DEPTH: 10.0
    MIN_DEPTH: 0.1
    NORMALIZE_DEPTH: False
    ORIENTATION: [0.0, 0.0, 0.0]
    POSITION: [0, 0.88, 0]
    SENSOR_SUBTYPE: PINHOLE
    TYPE: HabitatSimDepthSensor
    WIDTH: 640
  EQUIRECT_DEPTH_SENSOR:
    HEIGHT: 480
    MAX_DEPTH: 10.0
    MIN_DEPTH: 0.0
    NORMALIZE_DEPTH: True
    ORIENTATION: [0.0, 0.0, 0.0]
    POSITION: [0, 1.25, 0]
    TYPE: HabitatSimEquirectangularDepthSensor
    WIDTH: 640
  EQUIRECT_RGB_SENSOR:
    HEIGHT: 480
    ORIENTATION: [0.0, 0.0, 0.0]
    POSITION: [0, 1.25, 0]
    TYPE: HabitatSimEquirectangularRGBSensor
    WIDTH: 640
  EQUIRECT_SEMANTIC_SENSOR:
    HEIGHT: 480
    ORIENTATION: [0.0, 0.0, 0.0]
    POSITION: [0, 1.25, 0]
    TYPE: HabitatSimEquirectangularSemanticSensor
    WIDTH: 640
  FISHEYE_DEPTH_SENSOR:
    ALPHA: 0.57
    FOCAL_LENGTH: [364.84, 364.86]
    HEIGHT: 480
    MAX_DEPTH: 10.0
    MIN_DEPTH: 0.0
    NORMALIZE_DEPTH: True
    ORIENTATION: [0.0, 0.0, 0.0]
    POSITION: [0, 1.25, 0]
    PRINCIPAL_POINT_OFFSET: None
    SENSOR_MODEL_TYPE: DOUBLE_SPHERE
    TYPE: HabitatSimFisheyeDepthSensor
    WIDTH: 640
    XI: -0.27
  FISHEYE_RGB_SENSOR:
    ALPHA: 0.57
    FOCAL_LENGTH: [364.84, 364.86]
    HEIGHT: 640
    ORIENTATION: [0.0, 0.0, 0.0]
    POSITION: [0, 1.25, 0]
    PRINCIPAL_POINT_OFFSET: None
    SENSOR_MODEL_TYPE: DOUBLE_SPHERE
    TYPE: HabitatSimFisheyeRGBSensor
    WIDTH: 640
    XI: -0.27
  FISHEYE_SEMANTIC_SENSOR:
    ALPHA: 0.57
    FOCAL_LENGTH: [364.84, 364.86]
    HEIGHT: 640
    ORIENTATION: [0.0, 0.0, 0.0]
    POSITION: [0, 1.25, 0]
    PRINCIPAL_POINT_OFFSET: None
    SENSOR_MODEL_TYPE: DOUBLE_SPHERE
    TYPE: HabitatSimFisheyeSemanticSensor
    WIDTH: 640
    XI: -0.27
  FORWARD_STEP_SIZE: 0.25
  HABITAT_SIM_V0:
    ALLOW_SLIDING: True
    ENABLE_PHYSICS: False
    GPU_DEVICE_ID: 0
    GPU_GPU: False
    LEAVE_CONTEXT_WITH_BACKGROUND_RENDERER: False
    PHYSICS_CONFIG_FILE: ./data/default.physics_config.json
  HEAD_DEPTH_SENSOR:
    HEIGHT: 480
    HFOV: 90
    MAX_DEPTH: 10.0
    MIN_DEPTH: 0.0
    NORMALIZE_DEPTH: True
    ORIENTATION: [0.0, 0.0, 0.0]
    POSITION: [0, 1.25, 0]
    SENSOR_SUBTYPE: PINHOLE
    TYPE: HabitatSimDepthSensor
    UUID: robot_head_depth
    WIDTH: 640
  HEAD_RGB_SENSOR:
    HEIGHT: 480
    HFOV: 90
    ORIENTATION: [0.0, 0.0, 0.0]
    POSITION: [0, 1.25, 0]
    SENSOR_SUBTYPE: PINHOLE
    TYPE: HabitatSimRGBSensor
    UUID: robot_head_rgb
    WIDTH: 640
  RGB_120:
    HEIGHT: 480
    HFOV: 90
    ORIENTATION: [0, -2.0943951023931953, 0]
    POSITION: [0, 0.88, 0]
    SENSOR_SUBTYPE: PINHOLE
    TYPE: HabitatSimRGBSensor
    UUID: rgb_120
    WIDTH: 640
  RGB_150:
    HEIGHT: 480
    HFOV: 90
    ORIENTATION: [0, -2.617993877991494, 0]
    POSITION: [0, 0.88, 0]
    SENSOR_SUBTYPE: PINHOLE
    TYPE: HabitatSimRGBSensor
    UUID: rgb_150
    WIDTH: 640
  RGB_180:
    HEIGHT: 480
    HFOV: 90
    ORIENTATION: [0, -3.141592653589793, 0]
    POSITION: [0, 0.88, 0]
    SENSOR_SUBTYPE: PINHOLE
    TYPE: HabitatSimRGBSensor
    UUID: rgb_180
    WIDTH: 640
  RGB_210:
    HEIGHT: 480
    HFOV: 90
    ORIENTATION: [0, -3.665191429188092, 0]
    POSITION: [0, 0.88, 0]
    SENSOR_SUBTYPE: PINHOLE
    TYPE: HabitatSimRGBSensor
    UUID: rgb_210
    WIDTH: 640
  RGB_240:
    HEIGHT: 480
    HFOV: 90
    ORIENTATION: [0, -4.1887902047863905, 0]
    POSITION: [0, 0.88, 0]
    SENSOR_SUBTYPE: PINHOLE
    TYPE: HabitatSimRGBSensor
    UUID: rgb_240
    WIDTH: 640
  RGB_270:
    HEIGHT: 480
    HFOV: 90
    ORIENTATION: [0, -4.71238898038469, 0]
    POSITION: [0, 0.88, 0]
    SENSOR_SUBTYPE: PINHOLE
    TYPE: HabitatSimRGBSensor
    UUID: rgb_270
    WIDTH: 640
  RGB_30:
    HEIGHT: 480
    HFOV: 90
    ORIENTATION: [0, -0.5235987755982988, 0]
    POSITION: [0, 0.88, 0]
    SENSOR_SUBTYPE: PINHOLE
    TYPE: HabitatSimRGBSensor
    UUID: rgb_30
    WIDTH: 640
  RGB_300:
    HEIGHT: 480
    HFOV: 90
    ORIENTATION: [0, -5.235987755982988, 0]
    POSITION: [0, 0.88, 0]
    SENSOR_SUBTYPE: PINHOLE
    TYPE: HabitatSimRGBSensor
    UUID: rgb_300
    WIDTH: 640
  RGB_330:
    HEIGHT: 480
    HFOV: 90
    ORIENTATION: [0, -5.759586531581287, 0]
    POSITION: [0, 0.88, 0]
    SENSOR_SUBTYPE: PINHOLE
    TYPE: HabitatSimRGBSensor
    UUID: rgb_330
    WIDTH: 640
  RGB_60:
    HEIGHT: 480
    HFOV: 90
    ORIENTATION: [0, -1.0471975511965976, 0]
    POSITION: [0, 0.88, 0]
    SENSOR_SUBTYPE: PINHOLE
    TYPE: HabitatSimRGBSensor
    UUID: rgb_60
    WIDTH: 640
  RGB_90:
    HEIGHT: 480
    HFOV: 90
    ORIENTATION: [0, -1.5707963267948966, 0]
    POSITION: [0, 0.88, 0]
    SENSOR_SUBTYPE: PINHOLE
    TYPE: HabitatSimRGBSensor
    UUID: rgb_90
    WIDTH: 640
  RGB_SENSOR:
    HEIGHT: 480
    HFOV: 90
    ORIENTATION: [0.0, 0.0, 0.0]
    POSITION: [0, 0.88, 0]
    SENSOR_SUBTYPE: PINHOLE
    TYPE: HabitatSimRGBSensor
    WIDTH: 640
  SCENE: data/scene_datasets/habitat-test-scenes/van-gogh-room.glb
  SCENE_DATASET: default
  SEED: 100
  SEMANTIC_SENSOR:
    HEIGHT: 480
    HFOV: 90
    ORIENTATION: [0.0, 0.0, 0.0]
    POSITION: [0, 1.25, 0]
    SENSOR_SUBTYPE: PINHOLE
    TYPE: HabitatSimSemanticSensor
    WIDTH: 640
  THIRD_DEPTH_SENSOR:
    HEIGHT: 480
    HFOV: 90
    MAX_DEPTH: 10.0
    MIN_DEPTH: 0.0
    NORMALIZE_DEPTH: True
    ORIENTATION: [0.0, 0.0, 0.0]
    POSITION: [0, 1.25, 0]
    SENSOR_SUBTYPE: PINHOLE
    TYPE: HabitatSimDepthSensor
    UUID: robot_third_rgb
    WIDTH: 640
  THIRD_RGB_SENSOR:
    HEIGHT: 480
    HFOV: 90
    ORIENTATION: [0.0, 0.0, 0.0]
    POSITION: [0, 1.25, 0]
    SENSOR_SUBTYPE: PINHOLE
    TYPE: HabitatSimRGBSensor
    UUID: robot_third_rgb
    WIDTH: 640
  TILT_ANGLE: 30
  TURN_ANGLE: 30
  TYPE: Sim-v1
SIMULATOR_GPU_IDS: [0]
TASK:
  ACTIONS:
    ANSWER:
      TYPE: AnswerAction
    HIGHTOLOW:
      TYPE: MoveHighToLowAction
    HIGHTOLOWEVAL:
      TYPE: MoveHighToLowActionEval
    HIGHTOLOWINFERENCE:
      TYPE: MoveHighToLowActionInference
    LOOK_DOWN:
      TYPE: LookDownAction
    LOOK_UP:
      TYPE: LookUpAction
    MOVE_FORWARD:
      TYPE: MoveForwardAction
    STOP:
      TYPE: StopAction
    TELEPORT:
      TYPE: TeleportAction
    TURN_LEFT:
      TYPE: TurnLeftAction
    TURN_RIGHT:
      TYPE: TurnRightAction
    TURN_RIGHT_2:
      TYPE: TurnRightAction2
    VELOCITY_CONTROL:
      ANG_VEL_RANGE: [-10.0, 10.0]
      LIN_VEL_RANGE: [0.0, 0.25]
      MIN_ABS_ANG_SPEED: 1.0
      MIN_ABS_LIN_SPEED: 0.025
      TIME_STEP: 1.0
      TYPE: VelocityAction
  ANSWER_ACCURACY:
    TYPE: AnswerAccuracy
  COLLISIONS:
    TYPE: Collisions
  COMPASS_SENSOR:
    TYPE: CompassSensor
  CORRECT_ANSWER:
    TYPE: CorrectAnswer
  DISTANCE_TO_GOAL:
    DISTANCE_TO: POINT
    TYPE: DistanceToGoal
  EPISODE_INFO:
    TYPE: EpisodeInfo
  GLOBAL_GPS_SENSOR:
    DIMENSIONALITY: 3
    TYPE: GlobalGPSSensor
  GOAL_SENSOR_UUID: pointgoal
  GPS_SENSOR:
    DIMENSIONALITY: 2
    TYPE: GPSSensor
  HEADING_SENSOR:
    TYPE: HeadingSensor
  IMAGEGOAL_SENSOR:
    TYPE: ImageGoalSensor
  INSTRUCTION_SENSOR:
    TYPE: InstructionSensor
  INSTRUCTION_SENSOR_UUID: instruction
  MEASUREMENTS: ['STEPS_TAKEN', 'PATH_LENGTH', 'DISTANCE_TO_GOAL', 'SUCCESS', 'ORACLE_SUCCESS', 'SPL']
  NDTW:
    FDTW: True
    GT_PATH: data/datasets/R2R_VLNCE_v1-2_preprocessed/{split}/{split}_gt.json.gz
    SPLIT: val_seen
    SUCCESS_DISTANCE: 3.0
    TYPE: NDTW
  OBJECTGOAL_SENSOR:
    GOAL_SPEC: TASK_CATEGORY_ID
    GOAL_SPEC_MAX_VAL: 50
    TYPE: ObjectGoalSensor
  ORACLE_NAVIGATION_ERROR:
    TYPE: OracleNavigationError
  ORACLE_SPL:
    TYPE: OracleSPL
  ORACLE_SUCCESS:
    SUCCESS_DISTANCE: 3.0
    TYPE: OracleSuccess
  OREINTATION_SENSOR:
    TYPE: OrienSensor
  PATH_LENGTH:
    TYPE: PathLength
  POINTGOAL_SENSOR:
    DIMENSIONALITY: 2
    GOAL_FORMAT: POLAR
    TYPE: PointGoalSensor
  POINTGOAL_WITH_GPS_COMPASS_SENSOR:
    DIMENSIONALITY: 2
    GOAL_FORMAT: POLAR
    TYPE: PointGoalWithGPSCompassSensor
  POSITION:
    TYPE: Position
  POSITION_INFER:
    TYPE: PositionInfer
  POSSIBLE_ACTIONS: ['STOP', 'MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'HIGHTOLOW']
  PROXIMITY_SENSOR:
    MAX_DETECTION_RADIUS: 2.0
    TYPE: ProximitySensor
  QUESTION_SENSOR:
    TYPE: QuestionSensor
  RXR_INSTRUCTION_SENSOR:
    TYPE: RxRInstructionSensor
    features_path: data/datasets/RxR_VLNCE_v0/text_features/rxr_{split}/{id:06}_{lang}_text_features.npz
    max_text_len: 512
  SDTW:
    GT_PATH: data/datasets/R2R_VLNCE_v1-2_preprocessed/{split}/{split}_gt.json.gz
    SUCCESS_DISTANCE: 3.0
    TYPE: SDTW
  SENSORS: ['INSTRUCTION_SENSOR', 'COMPASS_SENSOR', 'GPS_SENSOR']
  SHORTEST_PATH_SENSOR:
    GOAL_RADIUS: 0.5
    TYPE: ShortestPathSensor
    USE_ORIGINAL_FOLLOWER: False
  SOFT_SPL:
    TYPE: SoftSPL
  SPL:
    SUCCESS_DISTANCE: 3.0
    TYPE: SPL
  STEPS_TAKEN:
    TYPE: StepsTaken
  SUCCESS:
    SUCCESS_DISTANCE: 3.0
    TYPE: Success
  SUCCESS_DISTANCE: 3.0
  TASK_TYPE: r2r
  TOP_DOWN_MAP:
    DRAW_BORDER: True
    DRAW_GOAL_AABBS: True
    DRAW_GOAL_POSITIONS: True
    DRAW_SHORTEST_PATH: True
    DRAW_SOURCE: True
    DRAW_VIEW_POINTS: True
    FOG_OF_WAR:
      DRAW: True
      FOV: 90
      VISIBILITY_DIST: 5.0
    MAP_PADDING: 3
    MAP_RESOLUTION: 1024
    MAX_EPISODE_STEPS: 1000
    TYPE: TopDownMap
  TOP_DOWN_MAP_VLNCE:
    DRAW_BORDER: False
    DRAW_FIXED_WAYPOINTS: False
    DRAW_MP3D_AGENT_PATH: False
    DRAW_REFERENCE_PATH: False
    DRAW_SHORTEST_PATH: False
    DRAW_SOURCE_AND_TARGET: True
    FOG_OF_WAR:
      DRAW: False
      FOV: 79
      VISIBILITY_DIST: 5.0
    GRAPHS_FILE: data/connectivity_graphs.pkl
    MAP_RESOLUTION: 512
    MAX_EPISODE_STEPS: 1000
    TYPE: TopDownMapVLNCE
  TYPE: VLN-v0
  VLN_ORACLE_PROGRESS_SENSOR:
    TYPE: VLNOracleProgressSensor
TASK_CONFIG:
  DATASET:
    CONTENT_SCENES: ['*']
    DATASET_TYPE: r2r
    DATA_PATH: data/datasets/R2R_VLNCE_v1-2_preprocessed/{split}/{split}.json.gz
    EPISODES_ALLOWED: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    EXPERIMENT_ID: r2r_50_20260917-190953
    LANGUAGES: ['*']
    ROLES: ['guide']
    SCENES_DIR: data/scene_datasets/
    SPLIT: val_unseen
    SPLIT_INDEX: 0
    SPLIT_NUM: 1
    TYPE: VLN-CE-v1
  ENVIRONMENT:
    ITERATOR_OPTIONS:
      CYCLE: True
      GROUP_BY_SCENE: True
      MAX_SCENE_REPEAT_EPISODES: -1
      MAX_SCENE_REPEAT_STEPS: -1
      NUM_EPISODE_SAMPLE: -1
      SHUFFLE: False
      STEP_REPETITION_RANGE: 0.2
    MAX_EPISODE_SECONDS: 10000000
    MAX_EPISODE_STEPS: 5000
  PYROBOT:
    BASE_CONTROLLER: proportional
    BASE_PLANNER: none
    BUMP_SENSOR:
      TYPE: PyRobotBumpSensor
    DEPTH_SENSOR:
      CENTER_CROP: False
      HEIGHT: 480
      MAX_DEPTH: 5.0
      MIN_DEPTH: 0.0
      NORMALIZE_DEPTH: True
      TYPE: PyRobotDepthSensor
      WIDTH: 640
    LOCOBOT:
      ACTIONS: ['BASE_ACTIONS', 'CAMERA_ACTIONS']
      BASE_ACTIONS: ['go_to_relative', 'go_to_absolute']
      CAMERA_ACTIONS: ['set_pan', 'set_tilt', 'set_pan_tilt']
    RGB_SENSOR:
      CENTER_CROP: False
      HEIGHT: 480
      TYPE: PyRobotRGBSensor
      WIDTH: 640
    ROBOT: locobot
    ROBOTS: ['locobot']
    SENSORS: ['RGB_SENSOR', 'DEPTH_SENSOR', 'BUMP_SENSOR']
  SEED: 100
  SIMULATOR:
    ACTION_SPACE_CONFIG: v0
    ADDITIONAL_OBJECT_PATHS: []
    AGENTS: ['AGENT_0']
    AGENT_0:
      HEIGHT: 1.5
      IS_SET_START_STATE: False
      RADIUS: 0.18
      SENSORS: ['RGB_SENSOR', 'DEPTH_SENSOR', 'RGB_30', 'RGB_60', 'RGB_90', 'RGB_120', 'RGB_150', 'RGB_180', 'RGB_210', 'RGB_240', 'RGB_270', 'RGB_300', 'RGB_330', 'DEPTH_30', 'DEPTH_60', 'DEPTH_90', 'DEPTH_120', 'DEPTH_150', 'DEPTH_180', 'DEPTH_210', 'DEPTH_240', 'DEPTH_270', 'DEPTH_300', 'DEPTH_330']
      START_POSITION: [0, 0, 0]
      START_ROTATION: [0, 0, 0, 1]
    ARM_DEPTH_SENSOR:
      HEIGHT: 480
      HFOV: 90
      MAX_DEPTH: 10.0
      MIN_DEPTH: 0.0
      NORMALIZE_DEPTH: True
      ORIENTATION: [0.0, 0.0, 0.0]
      POSITION: [0, 1.25, 0]
      SENSOR_SUBTYPE: PINHOLE
      TYPE: HabitatSimDepthSensor
      UUID: robot_arm_depth
      WIDTH: 640
    ARM_RGB_SENSOR:
      HEIGHT: 480
      HFOV: 90
      ORIENTATION: [0.0, 0.0, 0.0]
      POSITION: [0, 1.25, 0]
      SENSOR_SUBTYPE: PINHOLE
      TYPE: HabitatSimRGBSensor
      UUID: robot_arm_rgb
      WIDTH: 640
    DEFAULT_AGENT_ID: 0
    DEPTH_120:
      HEIGHT: 480
      HFOV: 90
      MAX_DEPTH: 10.0
      MIN_DEPTH: 0.1
      NORMALIZE_DEPTH: False
      ORIENTATION: [0, -2.0943951023931953, 0]
      POSITION: [0, 0.88, 0]
      SENSOR_SUBTYPE: PINHOLE
      TYPE: HabitatSimDepthSensor
      UUID: depth_120
      WIDTH: 640
    DEPTH_150:
      HEIGHT: 480
      HFOV: 90
      MAX_DEPTH: 10.0
      MIN_DEPTH: 0.1
      NORMALIZE_DEPTH: False
      ORIENTATION: [0, -2.617993877991494, 0]
      POSITION: [0, 0.88, 0]
      SENSOR_SUBTYPE: PINHOLE
      TYPE: HabitatSimDepthSensor
      UUID: depth_150
      WIDTH: 640
    DEPTH_180:
      HEIGHT: 480
      HFOV: 90
      MAX_DEPTH: 10.0
      MIN_DEPTH: 0.1
      NORMALIZE_DEPTH: False
      ORIENTATION: [0, -3.141592653589793, 0]
      POSITION: [0, 0.88, 0]
      SENSOR_SUBTYPE: PINHOLE
      TYPE: HabitatSimDepthSensor
      UUID: depth_180
      WIDTH: 640
    DEPTH_210:
      HEIGHT: 480
      HFOV: 90
      MAX_DEPTH: 10.0
      MIN_DEPTH: 0.1
      NORMALIZE_DEPTH: False
      ORIENTATION: [0, -3.665191429188092, 0]
      POSITION: [0, 0.88, 0]
      SENSOR_SUBTYPE: PINHOLE
      TYPE: HabitatSimDepthSensor
      UUID: depth_210
      WIDTH: 640
    DEPTH_240:
      HEIGHT: 480
      HFOV: 90
      MAX_DEPTH: 10.0
      MIN_DEPTH: 0.1
      NORMALIZE_DEPTH: False
      ORIENTATION: [0, -4.1887902047863905, 0]
      POSITION: [0, 0.88, 0]
      SENSOR_SUBTYPE: PINHOLE
      TYPE: HabitatSimDepthSensor
      UUID: depth_240
      WIDTH: 640
    DEPTH_270:
      HEIGHT: 480
      HFOV: 90
      MAX_DEPTH: 10.0
      MIN_DEPTH: 0.1
      NORMALIZE_DEPTH: False
      ORIENTATION: [0, -4.71238898038469, 0]
      POSITION: [0, 0.88, 0]
      SENSOR_SUBTYPE: PINHOLE
      TYPE: HabitatSimDepthSensor
      UUID: depth_270
      WIDTH: 640
    DEPTH_30:
      HEIGHT: 480
      HFOV: 90
      MAX_DEPTH: 10.0
      MIN_DEPTH: 0.1
      NORMALIZE_DEPTH: False
      ORIENTATION: [0, -0.5235987755982988, 0]
      POSITION: [0, 0.88, 0]
      SENSOR_SUBTYPE: PINHOLE
      TYPE: HabitatSimDepthSensor
      UUID: depth_30
      WIDTH: 640
    DEPTH_300:
      HEIGHT: 480
      HFOV: 90
      MAX_DEPTH: 10.0
      MIN_DEPTH: 0.1
      NORMALIZE_DEPTH: False
      ORIENTATION: [0, -5.235987755982988, 0]
      POSITION: [0, 0.88, 0]
      SENSOR_SUBTYPE: PINHOLE
      TYPE: HabitatSimDepthSensor
      UUID: depth_300
      WIDTH: 640
    DEPTH_330:
      HEIGHT: 480
      HFOV: 90
      MAX_DEPTH: 10.0
      MIN_DEPTH: 0.1
      NORMALIZE_DEPTH: False
      ORIENTATION: [0, -5.759586531581287, 0]
      POSITION: [0, 0.88, 0]
      SENSOR_SUBTYPE: PINHOLE
      TYPE: HabitatSimDepthSensor
      UUID: depth_330
      WIDTH: 640
    DEPTH_60:
      HEIGHT: 480
      HFOV: 90
      MAX_DEPTH: 10.0
      MIN_DEPTH: 0.1
      NORMALIZE_DEPTH: False
      ORIENTATION: [0, -1.0471975511965976, 0]
      POSITION: [0, 0.88, 0]
      SENSOR_SUBTYPE: PINHOLE
      TYPE: HabitatSimDepthSensor
      UUID: depth_60
      WIDTH: 640
    DEPTH_90:
      HEIGHT: 480
      HFOV: 90
      MAX_DEPTH: 10.0
      MIN_DEPTH: 0.1
      NORMALIZE_DEPTH: False
      ORIENTATION: [0, -1.5707963267948966, 0]
      POSITION: [0, 0.88, 0]
      SENSOR_SUBTYPE: PINHOLE
      TYPE: HabitatSimDepthSensor
      UUID: depth_90
      WIDTH: 640
    DEPTH_SENSOR:
      HEIGHT: 480
      HFOV: 90
      MAX_DEPTH: 10.0
      MIN_DEPTH: 0.1
      NORMALIZE_DEPTH: False
      ORIENTATION: [0.0, 0.0, 0.0]
      POSITION: [0, 0.88, 0]
      SENSOR_SUBTYPE: PINHOLE
      TYPE: HabitatSimDepthSensor
      WIDTH: 640
    EQUIRECT_DEPTH_SENSOR:
      HEIGHT: 480
      MAX_DEPTH: 10.0
      MIN_DEPTH: 0.0
      NORMALIZE_DEPTH: True
      ORIENTATION: [0.0, 0.0, 0.0]
      POSITION: [0, 1.25, 0]
      TYPE: HabitatSimEquirectangularDepthSensor
      WIDTH: 640
    EQUIRECT_RGB_SENSOR:
      HEIGHT: 480
      ORIENTATION: [0.0, 0.0, 0.0]
      POSITION: [0, 1.25, 0]
      TYPE: HabitatSimEquirectangularRGBSensor
      WIDTH: 640
    EQUIRECT_SEMANTIC_SENSOR:
      HEIGHT: 480
      ORIENTATION: [0.0, 0.0, 0.0]
      POSITION: [0, 1.25, 0]
      TYPE: HabitatSimEquirectangularSemanticSensor
      WIDTH: 640
    FISHEYE_DEPTH_SENSOR:
      ALPHA: 0.57
      FOCAL_LENGTH: [364.84, 364.86]
      HEIGHT: 480
      MAX_DEPTH: 10.0
      MIN_DEPTH: 0.0
      NORMALIZE_DEPTH: True
      ORIENTATION: [0.0, 0.0, 0.0]
      POSITION: [0, 1.25, 0]
      PRINCIPAL_POINT_OFFSET: None
      SENSOR_MODEL_TYPE: DOUBLE_SPHERE
      TYPE: HabitatSimFisheyeDepthSensor
      WIDTH: 640
      XI: -0.27
    FISHEYE_RGB_SENSOR:
      ALPHA: 0.57
      FOCAL_LENGTH: [364.84, 364.86]
      HEIGHT: 640
      ORIENTATION: [0.0, 0.0, 0.0]
      POSITION: [0, 1.25, 0]
      PRINCIPAL_POINT_OFFSET: None
      SENSOR_MODEL_TYPE: DOUBLE_SPHERE
      TYPE: HabitatSimFisheyeRGBSensor
      WIDTH: 640
      XI: -0.27
    FISHEYE_SEMANTIC_SENSOR:
      ALPHA: 0.57
      FOCAL_LENGTH: [364.84, 364.86]
      HEIGHT: 640
      ORIENTATION: [0.0, 0.0, 0.0]
      POSITION: [0, 1.25, 0]
      PRINCIPAL_POINT_OFFSET: None
      SENSOR_MODEL_TYPE: DOUBLE_SPHERE
      TYPE: HabitatSimFisheyeSemanticSensor
      WIDTH: 640
      XI: -0.27
    FORWARD_STEP_SIZE: 0.25
    HABITAT_SIM_V0:
      ALLOW_SLIDING: True
      ENABLE_PHYSICS: False
      GPU_DEVICE_ID: 0
      GPU_GPU: False
      LEAVE_CONTEXT_WITH_BACKGROUND_RENDERER: False
      PHYSICS_CONFIG_FILE: ./data/default.physics_config.json
    HEAD_DEPTH_SENSOR:
      HEIGHT: 480
      HFOV: 90
      MAX_DEPTH: 10.0
      MIN_DEPTH: 0.0
      NORMALIZE_DEPTH: True
      ORIENTATION: [0.0, 0.0, 0.0]
      POSITION: [0, 1.25, 0]
      SENSOR_SUBTYPE: PINHOLE
      TYPE: HabitatSimDepthSensor
      UUID: robot_head_depth
      WIDTH: 640
    HEAD_RGB_SENSOR:
      HEIGHT: 480
      HFOV: 90
      ORIENTATION: [0.0, 0.0, 0.0]
      POSITION: [0, 1.25, 0]
      SENSOR_SUBTYPE: PINHOLE
      TYPE: HabitatSimRGBSensor
      UUID: robot_head_rgb
      WIDTH: 640
    RGB_120:
      HEIGHT: 480
      HFOV: 90
      ORIENTATION: [0, -2.0943951023931953, 0]
      POSITION: [0, 0.88, 0]
      SENSOR_SUBTYPE: PINHOLE
      TYPE: HabitatSimRGBSensor
      UUID: rgb_120
      WIDTH: 640
    RGB_150:
      HEIGHT: 480
      HFOV: 90
      ORIENTATION: [0, -2.617993877991494, 0]
      POSITION: [0, 0.88, 0]
      SENSOR_SUBTYPE: PINHOLE
      TYPE: HabitatSimRGBSensor
      UUID: rgb_150
      WIDTH: 640
    RGB_180:
      HEIGHT: 480
      HFOV: 90
      ORIENTATION: [0, -3.141592653589793, 0]
      POSITION: [0, 0.88, 0]
      SENSOR_SUBTYPE: PINHOLE
      TYPE: HabitatSimRGBSensor
      UUID: rgb_180
      WIDTH: 640
    RGB_210:
      HEIGHT: 480
      HFOV: 90
      ORIENTATION: [0, -3.665191429188092, 0]
      POSITION: [0, 0.88, 0]
      SENSOR_SUBTYPE: PINHOLE
      TYPE: HabitatSimRGBSensor
      UUID: rgb_210
      WIDTH: 640
    RGB_240:
      HEIGHT: 480
      HFOV: 90
      ORIENTATION: [0, -4.1887902047863905, 0]
      POSITION: [0, 0.88, 0]
      SENSOR_SUBTYPE: PINHOLE
      TYPE: HabitatSimRGBSensor
      UUID: rgb_240
      WIDTH: 640
    RGB_270:
      HEIGHT: 480
      HFOV: 90
      ORIENTATION: [0, -4.71238898038469, 0]
      POSITION: [0, 0.88, 0]
      SENSOR_SUBTYPE: PINHOLE
      TYPE: HabitatSimRGBSensor
      UUID: rgb_270
      WIDTH: 640
    RGB_30:
      HEIGHT: 480
      HFOV: 90
      ORIENTATION: [0, -0.5235987755982988, 0]
      POSITION: [0, 0.88, 0]
      SENSOR_SUBTYPE: PINHOLE
      TYPE: HabitatSimRGBSensor
      UUID: rgb_30
      WIDTH: 640
    RGB_300:
      HEIGHT: 480
      HFOV: 90
      ORIENTATION: [0, -5.235987755982988, 0]
      POSITION: [0, 0.88, 0]
      SENSOR_SUBTYPE: PINHOLE
      TYPE: HabitatSimRGBSensor
      UUID: rgb_300
      WIDTH: 640
    RGB_330:
      HEIGHT: 480
      HFOV: 90
      ORIENTATION: [0, -5.759586531581287, 0]
      POSITION: [0, 0.88, 0]
      SENSOR_SUBTYPE: PINHOLE
      TYPE: HabitatSimRGBSensor
      UUID: rgb_330
      WIDTH: 640
    RGB_60:
      HEIGHT: 480
      HFOV: 90
      ORIENTATION: [0, -1.0471975511965976, 0]
      POSITION: [0, 0.88, 0]
      SENSOR_SUBTYPE: PINHOLE
      TYPE: HabitatSimRGBSensor
      UUID: rgb_60
      WIDTH: 640
    RGB_90:
      HEIGHT: 480
      HFOV: 90
      ORIENTATION: [0, -1.5707963267948966, 0]
      POSITION: [0, 0.88, 0]
      SENSOR_SUBTYPE: PINHOLE
      TYPE: HabitatSimRGBSensor
      UUID: rgb_90
      WIDTH: 640
    RGB_SENSOR:
      HEIGHT: 480
      HFOV: 90
      ORIENTATION: [0.0, 0.0, 0.0]
      POSITION: [0, 0.88, 0]
      SENSOR_SUBTYPE: PINHOLE
      TYPE: HabitatSimRGBSensor
      WIDTH: 640
    SCENE: data/scene_datasets/habitat-test-scenes/van-gogh-room.glb
    SCENE_DATASET: default
    SEED: 100
    SEMANTIC_SENSOR:
      HEIGHT: 480
      HFOV: 90
      ORIENTATION: [0.0, 0.0, 0.0]
      POSITION: [0, 1.25, 0]
      SENSOR_SUBTYPE: PINHOLE
      TYPE: HabitatSimSemanticSensor
      WIDTH: 640
    THIRD_DEPTH_SENSOR:
      HEIGHT: 480
      HFOV: 90
      MAX_DEPTH: 10.0
      MIN_DEPTH: 0.0
      NORMALIZE_DEPTH: True
      ORIENTATION: [0.0, 0.0, 0.0]
      POSITION: [0, 1.25, 0]
      SENSOR_SUBTYPE: PINHOLE
      TYPE: HabitatSimDepthSensor
      UUID: robot_third_rgb
      WIDTH: 640
    THIRD_RGB_SENSOR:
      HEIGHT: 480
      HFOV: 90
      ORIENTATION: [0.0, 0.0, 0.0]
      POSITION: [0, 1.25, 0]
      SENSOR_SUBTYPE: PINHOLE
      TYPE: HabitatSimRGBSensor
      UUID: robot_third_rgb
      WIDTH: 640
    TILT_ANGLE: 30
    TURN_ANGLE: 30
    TYPE: Sim-v1
  TASK:
    ACTIONS:
      ANSWER:
        TYPE: AnswerAction
      HIGHTOLOW:
        TYPE: MoveHighToLowAction
      HIGHTOLOWEVAL:
        TYPE: MoveHighToLowActionEval
      HIGHTOLOWINFERENCE:
        TYPE: MoveHighToLowActionInference
      LOOK_DOWN:
        TYPE: LookDownAction
      LOOK_UP:
        TYPE: LookUpAction
      MOVE_FORWARD:
        TYPE: MoveForwardAction
      STOP:
        TYPE: StopAction
      TELEPORT:
        TYPE: TeleportAction
      TURN_LEFT:
        TYPE: TurnLeftAction
      TURN_RIGHT:
        TYPE: TurnRightAction
      TURN_RIGHT_2:
        TYPE: TurnRightAction2
      VELOCITY_CONTROL:
        ANG_VEL_RANGE: [-10.0, 10.0]
        LIN_VEL_RANGE: [0.0, 0.25]
        MIN_ABS_ANG_SPEED: 1.0
        MIN_ABS_LIN_SPEED: 0.025
        TIME_STEP: 1.0
        TYPE: VelocityAction
    ANSWER_ACCURACY:
      TYPE: AnswerAccuracy
    COLLISIONS:
      TYPE: Collisions
    COMPASS_SENSOR:
      TYPE: CompassSensor
    CORRECT_ANSWER:
      TYPE: CorrectAnswer
    DISTANCE_TO_GOAL:
      DISTANCE_TO: POINT
      TYPE: DistanceToGoal
    EPISODE_INFO:
      TYPE: EpisodeInfo
    GLOBAL_GPS_SENSOR:
      DIMENSIONALITY: 3
      TYPE: GlobalGPSSensor
    GOAL_SENSOR_UUID: pointgoal
    GPS_SENSOR:
      DIMENSIONALITY: 2
      TYPE: GPSSensor
    HEADING_SENSOR:
      TYPE: HeadingSensor
    IMAGEGOAL_SENSOR:
      TYPE: ImageGoalSensor
    INSTRUCTION_SENSOR:
      TYPE: InstructionSensor
    INSTRUCTION_SENSOR_UUID: instruction
    MEASUREMENTS: ['STEPS_TAKEN', 'PATH_LENGTH', 'DISTANCE_TO_GOAL', 'SUCCESS', 'ORACLE_SUCCESS', 'SPL']
    NDTW:
      FDTW: True
      GT_PATH: data/datasets/R2R_VLNCE_v1-2_preprocessed/{split}/{split}_gt.json.gz
      SPLIT: val_seen
      SUCCESS_DISTANCE: 3.0
      TYPE: NDTW
    OBJECTGOAL_SENSOR:
      GOAL_SPEC: TASK_CATEGORY_ID
      GOAL_SPEC_MAX_VAL: 50
      TYPE: ObjectGoalSensor
    ORACLE_NAVIGATION_ERROR:
      TYPE: OracleNavigationError
    ORACLE_SPL:
      TYPE: OracleSPL
    ORACLE_SUCCESS:
      SUCCESS_DISTANCE: 3.0
      TYPE: OracleSuccess
    OREINTATION_SENSOR:
      TYPE: OrienSensor
    PATH_LENGTH:
      TYPE: PathLength
    POINTGOAL_SENSOR:
      DIMENSIONALITY: 2
      GOAL_FORMAT: POLAR
      TYPE: PointGoalSensor
    POINTGOAL_WITH_GPS_COMPASS_SENSOR:
      DIMENSIONALITY: 2
      GOAL_FORMAT: POLAR
      TYPE: PointGoalWithGPSCompassSensor
    POSITION:
      TYPE: Position
    POSITION_INFER:
      TYPE: PositionInfer
    POSSIBLE_ACTIONS: ['STOP', 'MOVE_FORWARD', 'TURN_LEFT', 'TURN_RIGHT', 'HIGHTOLOW']
    PROXIMITY_SENSOR:
      MAX_DETECTION_RADIUS: 2.0
      TYPE: ProximitySensor
    QUESTION_SENSOR:
      TYPE: QuestionSensor
    RXR_INSTRUCTION_SENSOR:
      TYPE: RxRInstructionSensor
      features_path: data/datasets/RxR_VLNCE_v0/text_features/rxr_{split}/{id:06}_{lang}_text_features.npz
      max_text_len: 512
    SDTW:
      GT_PATH: data/datasets/R2R_VLNCE_v1-2_preprocessed/{split}/{split}_gt.json.gz
      SUCCESS_DISTANCE: 3.0
      TYPE: SDTW
    SENSORS: ['INSTRUCTION_SENSOR', 'COMPASS_SENSOR', 'GPS_SENSOR']
    SHORTEST_PATH_SENSOR:
      GOAL_RADIUS: 0.5
      TYPE: ShortestPathSensor
      USE_ORIGINAL_FOLLOWER: False
    SOFT_SPL:
      TYPE: SoftSPL
    SPL:
      SUCCESS_DISTANCE: 3.0
      TYPE: SPL
    STEPS_TAKEN:
      TYPE: StepsTaken
    SUCCESS:
      SUCCESS_DISTANCE: 3.0
      TYPE: Success
    SUCCESS_DISTANCE: 3.0
    TASK_TYPE: r2r
    TOP_DOWN_MAP:
      DRAW_BORDER: True
      DRAW_GOAL_AABBS: True
      DRAW_GOAL_POSITIONS: True
      DRAW_SHORTEST_PATH: True
      DRAW_SOURCE: True
      DRAW_VIEW_POINTS: True
      FOG_OF_WAR:
        DRAW: True
        FOV: 90
        VISIBILITY_DIST: 5.0
      MAP_PADDING: 3
      MAP_RESOLUTION: 1024
      MAX_EPISODE_STEPS: 1000
      TYPE: TopDownMap
    TOP_DOWN_MAP_VLNCE:
      DRAW_BORDER: False
      DRAW_FIXED_WAYPOINTS: False
      DRAW_MP3D_AGENT_PATH: False
      DRAW_REFERENCE_PATH: False
      DRAW_SHORTEST_PATH: False
      DRAW_SOURCE_AND_TARGET: True
      FOG_OF_WAR:
        DRAW: False
        FOV: 79
        VISIBILITY_DIST: 5.0
      GRAPHS_FILE: data/connectivity_graphs.pkl
      MAP_RESOLUTION: 512
      MAX_EPISODE_STEPS: 1000
      TYPE: TopDownMapVLNCE
    TYPE: VLN-v0
    VLN_ORACLE_PROGRESS_SENSOR:
      TYPE: VLNOracleProgressSensor
TORCH_GPU_ID: 0
TORCH_GPU_IDS: [0]
local_rank: 0
R2R 50-episode evaluation completed: r2r_50_20260917-190953
