# 代码注释


## 创建环境
```python
def construct_envs(
    config: Config,
    env_class: Type[Union[Env, RLEnv]],
    workers_ignore_signals: bool = False,
    auto_reset_done: bool = True,
    episodes_allowed: Optional[List[str]] = None,
) -> VectorEnv:

...

    is_debug = True if sys.gettrace() else False
    # 创建仿真器，并返回handler
    env_entry = habitat.ThreadedVectorEnv if is_debug else habitat.VectorEnv
    envs = env_entry(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(zip(configs, env_classes)), 
        auto_reset_done=auto_reset_done,
        workers_ignore_signals=workers_ignore_signals,
    )
    return envs
```

## 创建Agent

```python
class Agent():
    def __init__(self, env, config):

        # r2r和rxr处理逻辑稍有不同
        self.region_solver = Region_Solver(
            config.POLICY_CONFIG,
            self.device,
            self.dataset
        )
        self.camera_matrix = get_camera_matrix(640, 480, 90)

        # NUM_ENVIRONMENTS = 1, 没有并行
        self.sg_list = []
        self.rs_list = []

```


## Agent.act()
```python
    def act(self):
        '''experiment main function
        '''
        # experiment
        self.stat_eps = {} # 结果字典
        self.pbar = tqdm.tqdm(total=eps_to_eval)
        while len(self.stat_eps) < eps_to_eval:
            self.rollout() # 逐个episode，直到获得足够结果
        self.envs.close()

        # 处理异常结果

        num_episodes = len(self.stat_eps) # 统计平均指标
        for stat_key in next(iter(self.stat_eps.values())).keys():
        
```

## Agent.rollout()

```python
    def rollout(self):
        # 重置环境
        # 计算DAG
        stage_num, instruction_graphs, objects_sum, wrong_dags = \
            get_instruction_graphs(observations)
        # 重置sg/rs两个solver，加载DAG
        self._reset_policy(
            observations, 
            instruction_graphs, 
            stage_num,
            objects_sum,
            wrong_dags
        )
        
        # 循环，高层目标waypoint
        infos = [{'action_valid': 'complete'} for _ in range(self.envs.num_envs)]
        for stepk in range(self.max_len): 
            env_actions = []       
            for i in range(self.envs.num_envs): # 依旧num_envs为1,当作没有循环
                try:
                    # no dag
                    if not_done_index[i] in wrong_dags: # 没有env并行，not_done_index[i]等价于i
                        # 停止机器人
                        continue
                    if infos[i]['action_valid'] == 'threshold':
                        if self.thin_type < 2: # 目前THIN_TYPE = 1（thining()，not thining_v2()）
                            # 如果上个waypoint没有到达，处理下tree
                        
                    pose_matrix = get_pose_matrix(observations[i])
                    panoramas = rgbs_to_panorama(observations[i])
                    camera_matrix = self.camera_matrix
                    # 全景图
                    panoramas, camera_matrix, pose_matrix = rotate_180(panoramas, camera_matrix, pose_matrix)

                    sg_result = self.sg_list[i].get_scenegraph(
                        self.rs_list[i].stage, 
                        panoramas[0], 
                        panoramas[1], 
                        camera_matrix, 
                        pose_matrix,
                        [observations[i]['gps'][0], -observations[i]['gps'][1]],
                        self.thin_type, 
                    )
                    bev = self.sg_list[i].map_draft.bev_map # 可通行地图
                    bev_fmm = self.sg_list[i].map_draft.bev_map_fmm # fmm格式地图
                    bev_wall = self.sg_list[i].map_draft.bev_wall # 墙体地图
                    bev_thin = self.sg_list[i].map_draft.bev_thin # 骨架地图
                    scene_graph = self.sg_list[i].map_draft.scene_graph # 语义地图

                    # the location of the agent will adapt to bev in the function
                    next_point, stage_result = self.rs_list[i].get_next_point(
                        [observations[i]['gps'][0], -observations[i]['gps'][1]], 
                        observations[i]['compass'][0], 
                        scene_graph, 
                        infos[i]['action_valid'], 
                        stepk == self.max_len - 1, 
                        bev_wall, 
                        bev,
                        self.thin_type,  
                        bev_thin, 
                    )
                    agent_location = np.array([
                        -observations[i]['gps'][1] * self.config.POLICY_CONFIG.MAP.RESOLUTION
                          + self.config.POLICY_CONFIG.MAP.SIZE // 2,
                        observations[i]['gps'][0] * self.config.POLICY_CONFIG.MAP.RESOLUTION
                          + self.config.POLICY_CONFIG.MAP.SIZE // 2
                    ])

                    rs = self.rs_list[i]
                    vis_info = {
                        'nodes': list(rs.navigation_tree.stage_begin.values()),
                        'ghosts': list(rs.navigation_tree.waypoints_tree.nodes()),
                        'predict_ghost': next_point,
                    }
                    if stage_result == 'episode end' \
                        or stepk == self.max_len - 1 \
                        or next_point is None:
                        # get to the next_point and stop
                        act = 0
                        if next_point is None:
                            stage_begin = rs.navigation_tree.stage_begin
                            next_point = stage_begin[len(stage_begin)-1]
                    else:
                        act = 4
                    # add action to the action list
                    env_actions.append(
                        {
                            'action': {
                                'act': act,
                                'next_point': next_point,
                                'current_pos': agent_location,
                                'current_heading': observations[i]['compass'][0],
                                'location': observations[i]['gps'], 
                            },
                            'vis_info': vis_info,
                            'bev': bev_fmm
                        }
                    )
                    if False:
                        # visualize
                        _stage = self.rs_list[i].stage
                        if stage_result == 'next turn' or \
                            (stage_result == 'episode end' and stepk == self.max_len - 1):
                            _stage -= 1
                        if _stage == 0:
                            _stage = 1
                        if sg_result:
                            annotated_frame = draw_panorama(
                                panoramas[0],
                                self.sg_list[i].segment2d_results[-1]['xyxy'],
                                self.sg_list[i].segment2d_results[-1]['mask'],
                                self.sg_list[i].segment2d_results[-1]['caption'],
                                self.sg_list[i].segment2d_results[-1]['caption_conf'],
                            )
                        else:
                            annotated_frame = cv2.cvtColor(panoramas[0], cv2.COLOR_RGB2BGR)
                        other_masks = []
                        for stage_t, value in rs.masks.items():
                            if stage_t == _stage:
                                for j in range(len(value['object_masks'])):
                                    if 'keys' in value:
                                        key = value['keys'][j]
                                        current_constraint = rs.constraints[stage_t][key[0]]
                                        mask_name = current_constraint['nodes'][key[1]]
                                        if mask_name in rs.navigation_tree.navigation_tree:
                                            other_masks = other_masks + [value['object_masks'][j]]
                                    else:
                                        other_masks = other_masks + [value['object_masks'][j]]
                        if _stage in rs.masks:
                            valid_mask = rs.masks[_stage]['nav_mask']
                        else:
                            valid_mask = None
                        bev_for_vis = bev_fmm if self.dataset == 'rxr' else bev
                        bev_img = draw_layers(
                            bev_for_vis,
                            bev_thin,
                            valid_mask,
                            other_masks,
                            list(rs.navigation_tree.navigation_tree.nodes(data=True)),
                            list(rs.navigation_tree.navigation_tree.edges()),
                            scene_graph,
                            act,
                            next_point,
                            rs.navigation_tree.stage_begin,
                            rs.best_point,
                            agent_location,
                            observations[i]['compass'][0],
                            self.thin_type,
                        )

                        curr_eps = self.envs.current_episodes()
                        visualization_image = combine_image(observations[i], self.stat_eps, curr_eps[i].episode_id, _stage, bev_img, annotated_frame)
                        self.visualization_image_dict[curr_eps[i].episode_id].append(visualization_image)

                except KeyboardInterrupt:
                    print("KeyboardInterrupt received. Exiting...")
                    exit(1)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    action = {
                            'action': {
                                'act': 0,
                                'next_point': None,
                                'current_pos': observations[i]['gps'],
                                'current_heading': observations[i]['compass'],
                                'location': [0,0]
                            },
                            'vis_info': None,
                            'bev': None
                        }
                    if len(env_actions) > i:
                        env_actions[i] = action
                    else:
                        env_actions.append(action)

            # execute the action in corresponding envs
            outputs = self.envs.step(env_actions)
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]

            # calculate the metric for the finished episode
            for k in range(self.envs.num_envs):
                curr_eps = self.envs.current_episodes()
                ep_id = curr_eps[k].episode_id
                metric = self.calculate_metric(infos[k])
                self.stat_eps[ep_id] = metric
                if not dones[k]:
                    continue
                self.pbar.update()

            # pause completed environment
            if sum(dones) > 0:
                for m in reversed(list(range(self.envs.num_envs))):
                    if dones[m]:
                        not_done_index.pop(m)
                        self.envs.pause_at(m)
                        observations.pop(m)
                        infos.pop(m)

            if self.envs.num_envs == 0:
                break

        # write the result of each episode
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        fname = os.path.join(
            self.log_save_dir,
            f"zero_shot_vln_{split}_r{self.local_rank}_w{self.world_size}_{self.config.DATASET.SPLIT_INDEX}({self.config.DATASET.SPLIT_NUM}).json",
        )
        with open(fname, "w") as f:
            json.dump(self.stat_eps, f, indent=2)
        save_video(self.visualization_save_dir, self.visualization_image_dict)

```
