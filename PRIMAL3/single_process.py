import os
import os.path as osp

import numpy as np
import setproctitle
import tensorflow as tf
import torch
import wandb

from alg_parameters import *
from mapf_gym import MAPFEnv
from model import Model
from od_mstar3 import od_mstar
from od_mstar3.col_set_addition import OutOfTimeError, NoSolutionError
from util import set_global_seeds, write_to_tensorboard, write_to_wandb, make_gif, reset_env, one_step, update_perf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Runner(object):
    """sub-process used to collect experience"""

    def __init__(self, env_id):
        """Initialize model and environment"""
        self.ID = env_id
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

                actions, ps, values, pre_block, output_state, num_invalid= \
                    self.local_model.step(self.obs, self.vector, self.valid_actions, self.hidden_state,self.num_agent)
                self.one_episode_perf['invalid'] += num_invalid
                mb_values.append(values)
                mb_train_valid.append(self.train_valid)
                mb_ps.append(ps)
                mb_done.append(self.done)

                rewards, self.valid_actions, self.obs, self.vector, self.train_valid, self.done, blockings, \
                    num_on_goals, self.one_episode_perf, max_on_goals, action_status \
                    = one_step(self.env, self.one_episode_perf, actions, pre_block, self.num_agent)

                mb_actions.append(actions)
                for i in range(self.num_agent):
                    if action_status[i] == -3:
                        mb_train_valid[-1][i][int(actions[i])] = 0

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
                        torch.zeros((self.num_agent, NetParameters.NET_SIZE )).to(self.local_device),
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

                mb_advs[t] = last_gaelam = np.add(delta,
                                                        TrainingParameters.GAMMA * TrainingParameters.LAM
                                                        * next_nonterminal * last_gaelam)

            mb_returns = np.add(mb_advs, mb_values)

        return mb_obs, mb_vector, mb_returns, mb_values, mb_actions, mb_ps, mb_hidden_state, mb_train_valid, mb_blocking, \
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
        obs = np.zeros((1, self.imitation_num_agent, 4, EnvParameters.FOV_SIZE, EnvParameters.FOV_SIZE),
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


def main():
    """Main code."""
    # preparing for training
    if RecordingParameters.RETRAIN:
        restore_path = ''
        net_path_checkpoint = restore_path + "/net_checkpoint.pkl"
        net_dict = torch.load(net_path_checkpoint)

    if RecordingParameters.WANDB:
        if RecordingParameters.RETRAIN:
            wandb_id = None
        else:
            wandb_id = wandb.util.generate_id()
        wandb.init(project=RecordingParameters.EXPERIMENT_PROJECT,
                   name=RecordingParameters.EXPERIMENT_NAME,
                   entity=RecordingParameters.ENTITY,
                   notes=RecordingParameters.EXPERIMENT_NOTE,
                   config=all_args,
                   id=wandb_id,
                   resume='allow')
        print('id is:{}'.format(wandb_id))
        print('Launching wandb...\n')

    if RecordingParameters.TENSORBOARD:
        if RecordingParameters.RETRAIN:
            summary_path = ''
        else:
            summary_path = RecordingParameters.SUMMARY_PATH
        if not os.path.exists(summary_path):
            os.makedirs(summary_path)
        global_summary = tf.summary.FileWriter(summary_path)
        print('Launching tensorboard...\n')

        if RecordingParameters.TXT_WRITER:
            txt_path = summary_path + '/' + RecordingParameters.TXT_NAME
            with open(txt_path, "w") as f:
                f.write(str(all_args))
            print('Logging txt...\n')

    setproctitle.setproctitle(
        RecordingParameters.EXPERIMENT_PROJECT + RecordingParameters.EXPERIMENT_NAME + "@" + RecordingParameters.ENTITY)
    set_global_seeds(SetupParameters.SEED)

    # create classes
    global_device = torch.device('cuda') if SetupParameters.USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if SetupParameters.USE_GPU_LOCAL else torch.device('cpu')
    global_model = Model(0, global_device, True)

    if RecordingParameters.RETRAIN:
        global_model.network.load_state_dict(net_dict['model'])
        global_model.net_optimizer.load_state_dict(net_dict['optimizer'])

    envs = Runner(1)
    eval_env = MAPFEnv(num_agents=EnvParameters.N_AGENTS)

    if RecordingParameters.RETRAIN:
        curr_steps = net_dict["step"]
        curr_episodes = net_dict["episodes"]
        best_perf = net_dict["reward"]
    else:
        curr_steps = curr_episodes = best_perf = 0

    update_done = True
    demon = True
    last_test_t = -RecordingParameters.EVAL_INTERVAL - 1
    last_model_t = -RecordingParameters.SAVE_INTERVAL - 1
    last_best_t = -RecordingParameters.BEST_INTERVAL - 1
    last_gif_t = -RecordingParameters.GIF_INTERVAL - 1

    # start training
    try:
        while curr_steps < TrainingParameters.N_MAX_STEPS:
            if update_done:
                # start a data collection
                if global_device != local_device:
                    net_weights = global_model.network.to(local_device).state_dict()
                    global_model.network.to(global_device)
                else:
                    net_weights = global_model.network.state_dict()

                demon_probs = np.random.rand()
                if demon_probs < TrainingParameters.DEMONSTRATION_PROB:
                    demon = True
                    mb_obs, mb_vector, mb_actions, mb_hidden_state, num_episode, num_steps = envs.imitation(
                        net_weights)
                    curr_steps += num_steps
                    curr_episodes += num_episode
                else:
                    demon = False
                    mb_obs, mb_vector, mb_returns, mb_values, mb_actions, mb_ps, mb_hidden_state, mb_train_valid, mb_blocking, \
                        num_episode, performance_dict = envs.run(net_weights)
                    curr_steps += TrainingParameters.N_STEPS
                    curr_episodes += num_episode

                    for i in performance_dict.keys():
                        performance_dict[i] = np.nanmean(performance_dict[i])

            if demon:
                # training of imitation learning
                mb_imitation_loss = []
                for start in range(0, np.shape(mb_obs)[0], TrainingParameters.MINIBATCH_SIZE):
                    end = start + TrainingParameters.MINIBATCH_SIZE
                    slices = (arr[start:end] for arr in
                              (mb_obs, mb_vector, mb_actions, mb_hidden_state))
                    mb_imitation_loss.append(global_model.imitation_train(*slices))
                mb_imitation_loss = np.nanmean(mb_imitation_loss, axis=0)

                # record training result
                if RecordingParameters.WANDB:
                    write_to_wandb(curr_steps, imitation_loss=mb_imitation_loss, evaluate=False)
                if RecordingParameters.TENSORBOARD:
                    write_to_tensorboard(global_summary, curr_steps, imitation_loss=mb_imitation_loss, evaluate=False)
            else:
                # training of reinforcement learning
                mb_loss = []
                inds = np.arange(TrainingParameters.N_STEPS)
                for _ in range(TrainingParameters.N_EPOCHS):
                    np.random.shuffle(inds)
                    for start in range(0, TrainingParameters.N_STEPS, TrainingParameters.MINIBATCH_SIZE):
                        end = start + TrainingParameters.MINIBATCH_SIZE
                        mb_inds = inds[start:end]
                        slices = (arr[mb_inds] for arr in
                                  (mb_obs, mb_vector, mb_returns, mb_values, mb_actions, mb_ps, mb_hidden_state,
                                   mb_train_valid, mb_blocking))
                        mb_loss.append(global_model.train(*slices))

                # record training result
                if RecordingParameters.WANDB:
                    write_to_wandb(curr_steps, performance_dict, mb_loss, evaluate=False)
                if RecordingParameters.TENSORBOARD:
                    write_to_tensorboard(global_summary, curr_steps, performance_dict, mb_loss, evaluate=False)

            if (curr_steps - last_test_t) / RecordingParameters.EVAL_INTERVAL >= 1.0:
                # if save gif
                if (curr_steps - last_gif_t) / RecordingParameters.GIF_INTERVAL >= 1.0:
                    save_gif = True
                    last_gif_t = curr_steps
                else:
                    save_gif = False

                # evaluate training model
                last_test_t = curr_steps
                with torch.no_grad():
                    # greedy_eval_performance_dict = evaluate(eval_env,eval_memory, global_model,
                    # global_device, save_gif0, curr_steps, True)
                    eval_performance_dict = evaluate(eval_env, global_model, global_device, save_gif,
                                                     curr_steps, False)
                # record evaluation result
                if RecordingParameters.WANDB:
                    # write_to_wandb(curr_steps, greedy_eval_performance_dict, evaluate=True, greedy=True)
                    write_to_wandb(curr_steps, eval_performance_dict, evaluate=True, greedy=False)
                if RecordingParameters.TENSORBOARD:
                    # write_to_tensorboard(global_summary, curr_steps, greedy_eval_performance_dict, evaluate=True,
                    #                      greedy=True)
                    write_to_tensorboard(global_summary, curr_steps, eval_performance_dict, evaluate=True, greedy=False,
                                         )

                print('episodes: {}, step: {},episode reward: {}, final goals: {} \n'.format(
                    curr_episodes, curr_steps, eval_performance_dict['per_r'],
                    eval_performance_dict['per_final_goals']))
                # save model with the best performance
                if RecordingParameters.RECORD_BEST:
                    if eval_performance_dict['per_r'] > best_perf and (
                            curr_steps - last_best_t) / RecordingParameters.BEST_INTERVAL >= 1.0:
                        best_perf = eval_performance_dict['per_r']
                        last_best_t = curr_steps
                        print('Saving best model \n')
                        model_path = osp.join(RecordingParameters.MODEL_PATH, 'best_model')
                        if not os.path.exists(model_path):
                            os.makedirs(model_path)
                        path_checkpoint = model_path + "/net_checkpoint.pkl"
                        net_checkpoint = {"model": global_model.network.state_dict(),
                                          "optimizer": global_model.net_optimizer.state_dict(),
                                          "step": curr_steps,
                                          "episode": curr_episodes,
                                          "reward": best_perf}
                        torch.save(net_checkpoint, path_checkpoint)

            # save model
            if (curr_steps - last_model_t) / RecordingParameters.SAVE_INTERVAL >= 1.0:
                last_model_t = curr_steps
                print('Saving Model !\n')
                model_path = osp.join(RecordingParameters.MODEL_PATH, '%.5i' % curr_steps)
                os.makedirs(model_path)
                path_checkpoint = model_path + "/net_checkpoint.pkl"
                net_checkpoint = {"model": global_model.network.state_dict(),
                                  "optimizer": global_model.net_optimizer.state_dict(),
                                  "step": curr_steps,
                                  "episode": curr_episodes,
                                  "reward": eval_performance_dict['per_r']}
                torch.save(net_checkpoint, path_checkpoint)

    except KeyboardInterrupt:
        print("CTRL-C pressed. killing remote workers")

    # save final model
    print('Saving Final Model !\n')
    model_path = RecordingParameters.MODEL_PATH + '/final'
    os.makedirs(model_path)
    path_checkpoint = model_path + "/net_checkpoint.pkl"
    net_checkpoint = {"model": global_model.network.state_dict(),
                      "optimizer": global_model.net_optimizer.state_dict(),
                      "step": curr_steps,
                      "episode": curr_episodes,
                      "reward": eval_performance_dict['per_r']}
    torch.save(net_checkpoint, path_checkpoint)

    if RecordingParameters.WANDB:
        wandb.finish()


def evaluate(eval_env, model, device, save_gif, curr_steps, greedy):
    """Evaluate Model."""
    eval_performance_dict = {'per_r': [],  'per_valid_rate': [], 'per_episode_len': [],
                             'per_block': [], 'per_leave_goal': [], 'per_final_goals': [], 'per_half_goals': [],
                             'per_block_acc': [], 'per_max_goals': [], 'per_num_collide': []}
    episode_frames = []

    for i in range(RecordingParameters.EVAL_EPISODES):
        num_agent = EnvParameters.N_AGENTS

        # reset environment and buffer
        hidden_state = (torch.zeros((num_agent, NetParameters.NET_SIZE )).to(device),
                        torch.zeros((num_agent, NetParameters.NET_SIZE )).to(device))

        done, valid_actions, obs, vector, _ = reset_env(eval_env, num_agent)

        one_episode_perf = {'num_step': 0, 'episode_reward': 0, 'invalid': 0, 'block': 0,
                            'num_leave_goal': 0, 'wrong_blocking': 0, 'num_collide': 0}
        if save_gif:
            episode_frames.append(eval_env._render(mode='rgb_array', screen_width=900, screen_height=900))

        # stepping
        while not done:
            # predict
            actions, pre_block, hidden_state, num_invalid, v, ps = model.evaluate(obs, vector,
                                                                                               valid_actions,
                                                                                               hidden_state,
                                                                                               greedy,num_agent)
            one_episode_perf['invalid'] += num_invalid

            # move
            rewards, valid_actions, obs, vector, _, done, _, num_on_goals, one_episode_perf, max_on_goals, \
                _,  = one_step(eval_env, one_episode_perf, actions, pre_block, num_agent)

            if save_gif:
                episode_frames.append(eval_env._render(mode='rgb_array', screen_width=900, screen_height=900))

            one_episode_perf['episode_reward'] += np.sum(rewards)

            if one_episode_perf['num_step'] == EnvParameters.EPISODE_LEN // 2:
                eval_performance_dict['per_half_goals'].append(num_on_goals)

            if done:
                # save gif
                if save_gif:
                    if not os.path.exists(RecordingParameters.GIFS_PATH):
                        os.makedirs(RecordingParameters.GIFS_PATH)
                    images = np.array(episode_frames)
                    make_gif(images,
                             '{}/steps_{:d}_reward{:.1f}_final_goals{:.1f}_greedy{:d}.gif'.format(
                                 RecordingParameters.GIFS_PATH,
                                 curr_steps, one_episode_perf[
                                     'episode_reward'],
                                 num_on_goals, greedy))
                    save_gif = False

                eval_performance_dict = update_perf(one_episode_perf, eval_performance_dict, num_on_goals, max_on_goals,
                                                    num_agent)

    # average performance of multiple episodes
    for i in eval_performance_dict.keys():
        eval_performance_dict[i] = np.nanmean(eval_performance_dict[i])

    return eval_performance_dict


if __name__ == "__main__":
    main()
