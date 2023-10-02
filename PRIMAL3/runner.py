import numpy as np
import ray
import torch

from alg_parameters import *
from mapf_gym import MAPFEnv
from model import Model
from od_mstar3 import od_mstar
from od_mstar3.col_set_addition import OutOfTimeError, NoSolutionError
from util import one_step, update_perf, reset_env,set_global_seeds


@ray.remote(num_cpus=1, num_gpus=SetupParameters.NUM_GPU / (TrainingParameters.N_ENVS + 1))
class Runner(object):
    """sub-process used to collect experience"""

    def __init__(self, env_id):
        """initialize model0 and environment"""
        self.ID = env_id
        set_global_seeds(env_id * 123)
        self.num_agent = EnvParameters.N_AGENTS
        self.imitation_num_agent = EnvParameters.N_AGENTS
        self.one_episode_perf = {'num_step': 0, 'episode_reward': 0, 'invalid': 0, 'block': 0, 'num_leave_goal': 0,
                                 'wrong_blocking': 0, 'num_collide': 0}
        self.env = MAPFEnv(num_agents=self.num_agent)
        self.imitation_env = MAPFEnv(num_agents=self.imitation_num_agent)

        self.local_device = torch.device('cuda') if SetupParameters.USE_GPU_LOCAL else torch.device('cpu')
        self.local_model = Model(env_id, self.local_device)
        self.hidden_state = (
            torch.zeros((self.num_agent, NetParameters.NET_SIZE )).to(self.local_device),
            torch.zeros((self.num_agent, NetParameters.NET_SIZE )).to(self.local_device))

        self.done, self.valid_actions, self.obs, self.vector, self.train_valid = reset_env(self.env, self.num_agent)

    def run(self, weights):
        """run multiple steps and collect data for reinforcement learning"""
        with torch.no_grad():
            mb_obs, mb_vector, mb_rewards, mb_values, mb_done, mb_ps, mb_actions = [], [], [], [], [], [], []
            mb_hidden_state = []
            mb_train_valid, mb_blocking = [], []
            performance_dict = {'per_r': [],  'per_valid_rate': [],
                                'per_episode_len': [], 'per_block': [],
                                'per_leave_goal': [], 'per_final_goals': [], 'per_half_goals': [], 'per_block_acc': [],
                                'per_max_goals': [], 'per_num_collide': []}

            self.local_model.set_weights(weights)
            for _ in range(TrainingParameters.N_STEPS):
                mb_obs.append(self.obs)
                mb_vector.append(self.vector)
                mb_hidden_state.append(
                    [self.hidden_state[0].cpu().detach().numpy(), self.hidden_state[1].cpu().detach().numpy()])

                actions, ps, values, pre_block, self.hidden_state, num_invalid = \
                    self.local_model.step(self.obs, self.vector, self.valid_actions, self.hidden_state, self.num_agent)
                self.one_episode_perf['invalid'] += num_invalid
                mb_values.append(values)
                mb_train_valid.append(self.train_valid)
                mb_ps.append(ps)
                mb_done.append(self.done)

                rewards, self.valid_actions, self.obs, self.vector, self.train_valid, self.done, blockings, \
                    num_on_goals, self.one_episode_perf, max_on_goals, action_status \
                    = one_step(self.env, self.one_episode_perf, actions, pre_block, self.num_agent)

                for i in range(self.num_agent):
                    if action_status[i] == -3:
                        mb_train_valid[-1][i][int(actions[i])] = 0

                mb_actions.append(actions)
                mb_rewards.append(rewards)
                mb_blocking.append(blockings)

                self.one_episode_perf['episode_reward'] += np.sum(rewards)
                if self.one_episode_perf['num_step'] == EnvParameters.EPISODE_LEN // 2:
                    performance_dict['per_half_goals'].append(num_on_goals)

                if self.done:
                    performance_dict = update_perf(self.one_episode_perf, performance_dict, num_on_goals, max_on_goals,
                                                   self.num_agent)
                    self.one_episode_perf = {'num_step': 0, 'episode_reward': 0, 'invalid': 0, 'block': 0,
                                             'num_leave_goal': 0, 'wrong_blocking': 0, 'num_collide': 0}
                    self.num_agent = EnvParameters.N_AGENTS

                    self.done, self.valid_actions, self.obs, self.vector, self.train_valid = reset_env(self.env,
                                                                                                       self.num_agent)
                    self.done = True

                    self.hidden_state = (
                        torch.zeros((self.num_agent, NetParameters.NET_SIZE)).to(self.local_device),
                        torch.zeros((self.num_agent, NetParameters.NET_SIZE)).to(self.local_device))

            mb_obs = np.concatenate(mb_obs, axis=0)
            mb_vector = np.concatenate(mb_vector, axis=0)
            mb_rewards = np.concatenate(mb_rewards, axis=0)
            mb_values = np.squeeze(np.concatenate(mb_values, axis=0), axis=-1)
            mb_actions = np.asarray(mb_actions, dtype=np.int64)
            mb_ps = np.stack(mb_ps)
            mb_done = np.asarray(mb_done, dtype=np.bool_)
            mb_hidden_state = np.stack(mb_hidden_state)
            mb_train_valid = np.stack(mb_train_valid)
            mb_blocking = np.concatenate(mb_blocking, axis=0)

            last_values  = np.squeeze(
                self.local_model.value(self.obs, self.vector, self.hidden_state))

            # calculate advantages
            mb_advs = np.zeros_like(mb_rewards)
            last_gaelam = 0
            for t in reversed(range(TrainingParameters.N_STEPS)):
                if t == TrainingParameters.N_STEPS - 1:
                    next_nonterminal = 1.0 - self.done
                    next_values = last_values
                else:
                    next_nonterminal = 1.0 - mb_done[t + 1]
                    next_values= mb_values[t + 1]

                delta = np.subtract(np.add(mb_rewards[t], TrainingParameters.GAMMA * next_nonterminal *
                                              next_values), mb_values[t])

                mb_advs[t] = last_gaelam = np.add(delta, TrainingParameters.GAMMA * TrainingParameters.LAM
                                                        * next_nonterminal * last_gaelam)

            mb_returns = np.add(mb_advs, mb_values)

        return mb_obs, mb_vector, mb_returns, mb_values, mb_actions, mb_ps, mb_hidden_state, mb_train_valid, mb_blocking,\
            len(performance_dict['per_r']), performance_dict

    def imitation(self, weights):
        """run multiple steps and collect corresponding data for imitation learning"""
        with torch.no_grad():
            self.local_model.set_weights(weights)

            mb_obs, mb_vector, mb_hidden_state, mb_actions = [], [], [], []
            step = 0
            episode = 0
            self.imitation_num_agent = EnvParameters.N_AGENTS
            while step <= TrainingParameters.N_STEPS:
                self.imitation_env._reset(num_agents=self.imitation_num_agent)

                world = self.imitation_env.get_obstacle_map()
                start_positions = tuple(self.imitation_env.get_positions())
                goals = tuple(self.imitation_env.get_goals())

                try:
                    obs = None
                    mstar_path = od_mstar.find_path(world, start_positions, goals, inflation=2, time_limit=5)
                    obs, vector, actions, hidden_state = self.parse_path(mstar_path)
                except OutOfTimeError:
                    print("timeout")
                except NoSolutionError:
                    print("nosol????", start_positions)

                if obs is not None:  # no error
                    mb_obs.append(obs)
                    mb_vector.append(vector)
                    mb_actions.append(actions)
                    mb_hidden_state.append(hidden_state)
                    step += np.shape(vector)[0]
                    episode += 1

            mb_obs = np.concatenate(mb_obs, axis=0)
            mb_vector = np.concatenate(mb_vector, axis=0)
            mb_actions = np.concatenate(mb_actions, axis=0)
            mb_hidden_state = np.concatenate(mb_hidden_state, axis=0)
        return mb_obs, mb_vector, mb_actions, mb_hidden_state, episode, step

    def parse_path(self, path):
        """take the path generated from M* and create the corresponding inputs and actions"""
        mb_obs, mb_vector, mb_actions, mb_hidden_state = [], [], [], []
        hidden_state = (
            torch.zeros((self.imitation_num_agent, NetParameters.NET_SIZE )).to(self.local_device),
            torch.zeros((self.imitation_num_agent, NetParameters.NET_SIZE )).to(self.local_device))
        obs = np.zeros((1, self.imitation_num_agent, NetParameters.NUM_CHANNEL, EnvParameters.FOV_SIZE, EnvParameters.FOV_SIZE),
                       dtype=np.float32)
        vector = np.zeros((1, self.imitation_num_agent, NetParameters.VECTOR_LEN), dtype=np.float32)

        for i in range(self.imitation_num_agent):
            s = self.imitation_env.observe(i + 1)
            obs[:, i, :, :, :] = s[0]
            vector[:, i, : 3] = s[1]

        for t in range(len(path[:-1])):
            mb_obs.append(obs)
            mb_vector.append(vector)
            mb_hidden_state.append([hidden_state[0].cpu().detach().numpy(), hidden_state[1].cpu().detach().numpy()])

            hidden_state = self.local_model.generate_state(obs, vector, hidden_state)

            actions = np.zeros(self.imitation_num_agent)
            for i in range(self.imitation_num_agent):
                pos = path[t][i]
                new_pos = path[t + 1][i]  # guaranteed to be in bounds by loop guard
                direction = (new_pos[0] - pos[0], new_pos[1] - pos[1])
                actions[i] = self.imitation_env.world.get_action(direction)
            mb_actions.append(actions)

            obs, vector, rewards, done, _,  _, valid_actions, _, _, _, _, _, _ = \
                self.imitation_env.joint_step(actions, 0)

            vector[:, :, -1] = actions

            if not all(valid_actions):  # M* can not generate collisions
                print('invalid action')
                return None, None, None, None

        mb_obs = np.concatenate(mb_obs, axis=0)
        mb_vector = np.concatenate(mb_vector, axis=0)
        mb_actions = np.asarray(mb_actions, dtype=np.int64)
        mb_hidden_state = np.stack(mb_hidden_state)
        return mb_obs, mb_vector, mb_actions, mb_hidden_state
