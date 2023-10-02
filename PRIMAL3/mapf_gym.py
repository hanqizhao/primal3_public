import copy
import math
import random
import sys

import gym
import numpy as np
from gym import spaces
from gym.envs.classic_control import rendering
from matplotlib.colors import hsv_to_rgb

from alg_parameters import *
from od_mstar3 import od_mstar
from od_mstar3.col_set_addition import NoSolutionError

opposite_actions = {0: -1, 1: 3, 2: 4, 3: 1, 4: 2, 5: 7, 6: 8, 7: 5, 8: 6}
dirDict = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1), 4: (-1, 0), 5: (1, 1), 6: (1, -1), 7: (-1, -1),
           8: (-1, 1)}  # x,y operation for corresponding action
# -{0:STILL, 1:MOVE_NORTH, 2:MOVE_EAST, 3:MOVE_SOUTH, 4:MOVE_WEST}
actionDict = {v: k for k, v in dirDict.items()}


class State(object):
    """ map the environment as 2 2d numpy arrays """

    def __init__(self, world0, goals, num_agents):
        """initialization"""
        self.state = world0.copy()  # static obstacle: -1,empty: 0,agent = positive integer (agent_id)
        self.goals = goals.copy()  # empty: 0, goal = positive integer (corresponding to agent_id)
        self.num_agents = num_agents
        self.agents, self.agent_goals = self.scan_for_agents()  # position of agents, and position of goals

        assert (len(self.agents) == num_agents)

    def scan_for_agents(self):
        """find the position of agents and goals"""
        agents = [(-1, -1) for _ in range(self.num_agents)]
        agent_goals = [(-1, -1) for _ in range(self.num_agents)]

        for i in range(self.state.shape[0]):
            for j in range(self.state.shape[1]):  # check every position in the environment
                if self.state[i, j] > 0:  # agent
                    agents[self.state[i, j] - 1] = (i, j)
                if self.goals[i, j] > 0:  # goal
                    agent_goals[self.goals[i, j] - 1] = (i, j)
        assert ((-1, -1) not in agents and (-1, -1) not in agent_goals)
        return agents, agent_goals

    def get_pos(self, agent_id):
        """agent's current position"""
        return self.agents[agent_id - 1]

    def get_goal(self, agent_id):
        """the position of agent's goal"""
        return self.agent_goals[agent_id - 1]

    def find_swap(self, curr_position, past_position, actions):
        """check if there is a swap collision"""
        swap_index = []
        for i in range(self.num_agents):
            if actions[i] == 0:  # stay can not cause swap error
                continue
            else:
                ax = curr_position[i][0]
                ay = curr_position[i][1]
                agent_index = [index for (index, value) in enumerate(past_position) if value == (ax, ay)]
                for item in agent_index:
                    if i != item and curr_position[item] == past_position[i]:
                        swap_index.append([i, item])
        return swap_index

    def joint_move(self, actions):
        """simultaneously move agents and checks for collisions on the joint action """
        imag_state = (self.state > 0).astype(int)  # map of world 0-no agent, 1- have agent
        past_position = copy.deepcopy(self.agents)  # the position of agents before moving
        curr_position = copy.deepcopy(self.agents)  # the current position of agents after moving
        agent_status = np.zeros(self.num_agents)  # use to determine rewards and invalid actions

        # imagine moving
        for i in range(self.num_agents):
            direction = self.get_dir(actions[i])
            ax = self.agents[i][0]
            ay = self.agents[i][1]  # current position

            # Not moving is always allowed
            if direction == (0, 0):
                continue

            # Otherwise, let's look at the validity of the move
            dx, dy = direction[0], direction[1]
            if ax + dx >= self.state.shape[0] or ax + dx < 0 or ay + dy >= self.state.shape[1] or ay + dy < 0:
                # out of boundaries
                agent_status[i] = -1
                continue

            if self.state[ax + dx, ay + dy] < 0:  # collide with static obstacles
                agent_status[i] = -2
                continue

            imag_state[ax, ay] -= 1  # set the previous position to empty
            imag_state[ax + dx, ay + dy] += 1  # move to the new position
            curr_position[i] = (ax + dx, ay + dy)  # update agent's current position

        # solve collision between agents
        swap_index = self.find_swap(curr_position, past_position, actions)  # search for swapping collision
        collide_poss = np.argwhere(imag_state > 1)  # search for vertex collision
        while len(swap_index) > 0 or len(collide_poss) > 0:
            while len(collide_poss) > 0:
                agent_index = [index for (index, value) in enumerate(curr_position) if
                               all(value == collide_poss[0])]  # solve collisions one by one
                for i in agent_index:  # stop at it previous position
                    imag_state[curr_position[i]] -= 1
                    imag_state[past_position[i]] += 1
                    curr_position[i] = past_position[i]
                    agent_status[i] = -3

                collide_poss = np.argwhere(imag_state > 1)  # recheck

            swap_index = self.find_swap(curr_position, past_position, actions)

            while len(swap_index) > 0:
                couple = swap_index[0]  # solve collision one by one
                for i in couple:
                    imag_state[curr_position[i]] -= 1
                    imag_state[past_position[i]] += 1
                    curr_position[i] = past_position[i]
                    agent_status[i] = -3

                swap_index = self.find_swap(curr_position, past_position, actions)  # recheck

            collide_poss = np.argwhere(imag_state > 1)  # recheck

        assert len(np.argwhere(imag_state < 0)) == 0

        # Ture moving
        for i in range(self.num_agents):
            direction = self.get_dir(actions[i])
            # execute valid action
            if agent_status[i] == 0:
                dx, dy = direction[0], direction[1]
                ax = self.agents[i][0]
                ay = self.agents[i][1]
                self.state[ax, ay] = 0  # clean previous position
                self.agents[i] = (ax + dx, ay + dy)  # update agent's current position
                if self.goals[ax + dx, ay + dy] == i + 1:
                    agent_status[i] = 1  # reach goal
                    continue
                elif self.goals[ax + dx, ay + dy] != i + 1 and self.goals[ax, ay] == i + 1:
                    agent_status[i] = 2
                    continue  # on goal in last step and leave goal now
                else:
                    agent_status[i] = 0  # nothing happen

        for i in range(self.num_agents):
            self.state[self.agents[i]] = i + 1  # move to new position
        return agent_status

    def get_dir(self, action):
        """obtain corresponding x,y operation based on action"""
        return dirDict[action]

    def get_action(self, direction):
        """obtain corresponding action based on x,y operation"""
        return actionDict[direction]

    def task_done(self):
        """check if all agents on their goal"""
        num_complete = 0
        for i in range(1, len(self.agents) + 1):
            agent_pos = self.agents[i - 1]
            if self.goals[agent_pos[0], agent_pos[1]] == i:
                num_complete += 1
        return num_complete == len(self.agents), num_complete


class MAPFEnv(gym.Env):
    """map MAPF problems to a standard RL environment"""

    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, num_agents=EnvParameters.N_AGENTS, size=EnvParameters.WORLD_SIZE,
                 prob=EnvParameters.OBSTACLE_PROB):
        """initialization"""
        self.num_agents = num_agents
        self.observation_size = EnvParameters.FOV_SIZE
        self.SIZE = size  # size of a side of the square grid
        self.PROB = prob  # obstacle density
        self.max_on_goal = 0
        assert len(self.SIZE) == 2
        assert len(self.PROB) == 2

        self.set_world()
        self.action_space = spaces.Tuple([spaces.Discrete(self.num_agents), spaces.Discrete(EnvParameters.N_ACTIONS)])
        self.viewer = None

    def is_connected(self, world0):
        """check if each agent's start position and goal position are sampled from the same connected region"""
        sys.setrecursionlimit(10000)
        world0 = world0.copy()

        def first_free(world):
            for x in range(world.shape[0]):
                for y in range(world.shape[1]):
                    if world[x, y] == 0:
                        return x, y

        def flood_fill(world, k, g):
            sx, sy = world.shape[0], world.shape[1]
            if k < 0 or k >= sx or g < 0 or g >= sy:  # out of boundaries
                return
            if world[k, g] == -1:
                return  # obstacles
            world[k, g] = -1
            flood_fill(world, k + 1, g)
            flood_fill(world, k, g + 1)
            flood_fill(world, k - 1, g)
            flood_fill(world, k, g - 1)

        i, j = first_free(world0)
        flood_fill(world0, i, j)
        if np.any(world0 == 0):
            return False
        else:
            return True

    def get_obstacle_map(self):
        """get obstacle map"""
        return (self.world.state == -1).astype(int)

    def get_goals(self):
        """get all agents' goal position"""
        result = []
        for i in range(1, self.num_agents + 1):
            result.append(self.world.get_goal(i))
        return result

    def get_positions(self):
        """get all agents' position"""
        result = []
        for i in range(1, self.num_agents + 1):
            result.append(self.world.get_pos(i))
        return result

    def set_world(self):
        """randomly generate a new task"""

        def get_connected_region(world0, regions_dict, x0, y0):
            # ensure at the beginning of an episode, all agents and their goal at the same connected region
            sys.setrecursionlimit(1000000)
            if (x0, y0) in regions_dict:  # have done
                return regions_dict[(x0, y0)]
            visited = set()
            sx, sy = world0.shape[0], world0.shape[1]
            work_list = [(x0, y0)]
            while len(work_list) > 0:
                (i, j) = work_list.pop()
                if i < 0 or i >= sx or j < 0 or j >= sy:
                    continue
                if world0[i, j] == -1:
                    continue  # crashes
                if world0[i, j] > 0:
                    regions_dict[(i, j)] = visited
                if (i, j) in visited:
                    continue
                visited.add((i, j))
                work_list.append((i + 1, j))
                work_list.append((i, j + 1))
                work_list.append((i - 1, j))
                work_list.append((i, j - 1))
            regions_dict[(x0, y0)] = visited
            return visited

        prob = np.random.triangular(self.PROB[0], .33 * self.PROB[0] + .66 * self.PROB[1],
                                    self.PROB[1])  # sample a value from triangular distribution
        size = np.random.choice([self.SIZE[0], self.SIZE[0] * .5 + self.SIZE[1] * .5, self.SIZE[1]],
                                p=[.5, .25, .25])  # sample a value according to the given probability
        # prob = self.PROB
        # size = self.SIZE  # fixed world0 size and obstacle density for evaluation
        world = -(np.random.rand(int(size), int(size)) < prob).astype(int)  # -1 obstacle,0 nothing, >0 agent id

        # randomize the position of agents
        agent_counter = 1
        agent_locations = []
        while agent_counter <= self.num_agents:
            x, y = np.random.randint(0, world.shape[0]), np.random.randint(0, world.shape[1])
            if world[x, y] == 0:
                world[x, y] = agent_counter
                agent_locations.append((x, y))
                agent_counter += 1

        # randomize the position of goals
        goals = np.zeros(world.shape).astype(int)
        goal_counter = 1
        agent_regions = dict()
        while goal_counter <= self.num_agents:
            agent_pos = agent_locations[goal_counter - 1]
            valid_tiles = get_connected_region(world, agent_regions, agent_pos[0], agent_pos[1])
            x, y = random.choice(list(valid_tiles))
            if goals[x, y] == 0 and world[x, y] != -1:
                # ensure new goal does not at the same grid of old goals or obstacles
                goals[x, y] = goal_counter
                goal_counter += 1
        self.world = State(world, goals, self.num_agents)

    def observe(self, agent_id):
        """return one agent's observation"""
        assert (agent_id > 0)
        top_left = (self.world.get_pos(agent_id)[0] - self.observation_size // 2,
                    self.world.get_pos(agent_id)[1] - self.observation_size // 2)  # (top, left)
        obs_shape = (self.observation_size, self.observation_size)
        goal_map = np.zeros(obs_shape)  # own goal
        poss_map = np.zeros(obs_shape)  # agents
        goals_map = np.zeros(obs_shape)  # other observable agents' goal
        obs_map = np.zeros(obs_shape)  # obstacle
        visible_agents = []
        for i in range(top_left[0], top_left[0] + self.observation_size):  # top and bottom
            for j in range(top_left[1], top_left[1] + self.observation_size):  # left and right
                if i >= self.world.state.shape[0] or i < 0 or j >= self.world.state.shape[1] or j < 0:
                    # out of boundaries
                    obs_map[i - top_left[0], j - top_left[1]] = 1
                    continue
                if self.world.state[i, j] == -1:
                    # obstacles
                    obs_map[i - top_left[0], j - top_left[1]] = 1
                if self.world.state[i, j] == agent_id:
                    # own position
                    poss_map[i - top_left[0], j - top_left[1]] = 1
                if self.world.goals[i, j] == agent_id:
                    # own goal
                    goal_map[i - top_left[0], j - top_left[1]] = 1
                if self.world.state[i, j] > 0 and self.world.state[i, j] != agent_id:
                    # other agents' positions
                    visible_agents.append(self.world.state[i, j])
                    poss_map[i - top_left[0], j - top_left[1]] = 1

        for agent in visible_agents:
            x, y = self.world.get_goal(agent)
            # project the goal out of FOV to the boundary of FOV
            min_node = (max(top_left[0], min(top_left[0] + self.observation_size - 1, x)),
                        max(top_left[1], min(top_left[1] + self.observation_size - 1, y)))
            goals_map[min_node[0] - top_left[0], min_node[1] - top_left[1]] = 1

        dx = self.world.get_goal(agent_id)[0] - self.world.get_pos(agent_id)[0]  # distance on x axes
        dy = self.world.get_goal(agent_id)[1] - self.world.get_pos(agent_id)[1]  # distance on y axes
        mag = (dx ** 2 + dy ** 2) ** .5  # total distance
        if mag != 0:  # normalized
            dx = dx / mag
            dy = dy / mag
        return [poss_map, goal_map, goals_map, obs_map], [dx, dy, mag]

    def _reset(self, num_agents):
        """restart a new task"""
        self.num_agents = num_agents
        self.max_on_goal = 0
        if self.viewer is not None:
            self.viewer = None

        self.set_world()  # back to the initial situation
        return False

    def astar(self, world, start, goal, robots):
        """A* function for single agent"""
        for (i, j) in robots:
            world[i, j] = 1
        try:
            path = od_mstar.find_path(world, [start], [goal], inflation=1, time_limit=5)
        except NoSolutionError:
            path = None
        for (i, j) in robots:
            world[i, j] = 0
        return path

    def get_blocking_reward(self, agent_id):
        """calculates how many agents are prevented from reaching goal and returns the blocking penalty"""
        other_agents = []
        other_locations = []
        inflation = 10
        top_left = (self.world.get_pos(agent_id)[0] - self.observation_size // 2,
                    self.world.get_pos(agent_id)[1] - self.observation_size // 2)
        bottom_right = (top_left[0] + self.observation_size, top_left[1] + self.observation_size)
        for agent in range(1, self.num_agents):
            if agent == agent_id:
                continue
            x, y = self.world.get_pos(agent)
            if x < top_left[0] or x >= bottom_right[0] or y >= bottom_right[1] or y < top_left[1]:
                # exclude agent not in FOV
                continue
            other_agents.append(agent)
            other_locations.append((x, y))

        num_blocking = 0
        world = self.get_obstacle_map()
        for agent in other_agents:
            other_locations.remove(self.world.get_pos(agent))
            # before removing
            path_before = self.astar(world, self.world.get_pos(agent), self.world.get_goal(agent),
                                     robots=other_locations + [self.world.get_pos(agent_id)])
            # after removing
            path_after = self.astar(world, self.world.get_pos(agent), self.world.get_goal(agent),
                                    robots=other_locations)
            other_locations.append(self.world.get_pos(agent))
            if path_before is None and path_after is None:
                continue
            if path_before is not None and path_after is None:
                continue
            if (path_before is None and path_after is not None) or (len(path_before) > len(path_after) + inflation):
                num_blocking += 1
        return num_blocking * EnvParameters.BLOCKING_COST, num_blocking

    def list_next_valid_actions(self, agent_id, prev_action=0):
        """obtain the valid actions that can not lead to colliding with obstacles and boundaries
        or backing to previous position at next time step"""
        available_actions = [0]  # staying still always allowed

        agent_pos = self.world.get_pos(agent_id)
        ax, ay = agent_pos[0], agent_pos[1]

        for action in range(1, EnvParameters.N_ACTIONS):  # every action except 0
            direction = self.world.get_dir(action)
            dx, dy = direction[0], direction[1]
            if ax + dx >= self.world.state.shape[0] or ax + dx < 0 or ay + dy >= self.world.state.shape[
                    1] or ay + dy < 0:  # out of boundaries
                continue
            if self.world.state[ax + dx, ay + dy] < 0:  # collide with static obstacles
                continue
            # otherwise we are ok to carry out the action
            available_actions.append(action)

        if opposite_actions[prev_action] in available_actions:  # back to previous position
            available_actions.remove(opposite_actions[prev_action])
        return available_actions

    def joint_step(self, actions, num_step):
        """execute joint action and obtain reward"""
        action_status = self.world.joint_move(actions)
        valid_actions = [action_status[i] >= 0 for i in range(self.num_agents)]
        #     2: action executed and agent leave its own goal
        #     1: action executed and reached/stayed on goal
        #     0: action executed
        #    -1: out of boundaries
        #    -2: collision with obstacles
        #    -3: collision with agents

        # initialization
        blockings = np.zeros((1, self.num_agents), dtype=np.float32)
        rewards = np.zeros((1, self.num_agents), dtype=np.float32)
        obs = np.zeros((1, self.num_agents, NetParameters.NUM_CHANNEL, EnvParameters.FOV_SIZE, EnvParameters.FOV_SIZE),
                       dtype=np.float32)
        vector = np.zeros((1, self.num_agents, NetParameters.VECTOR_LEN), dtype=np.float32)
        next_valid_actions = []
        num_blockings = 0
        leave_goals = 0
        num_collide = 0

        for i in range(self.num_agents):
            if actions[i] == 0:  # staying still
                if action_status[i] == 1:  # stayed on goal
                    rewards[:, i] = EnvParameters.GOAL_REWARD
                    if self.num_agents < 32:  # do not calculate A* for increasing speed
                        x, num_blocking = self.get_blocking_reward(i + 1)
                        num_blockings += num_blocking
                        rewards[:, i] += x
                        if x < 0:
                            blockings[:, i] = 1
                elif action_status[i] == 0:  # stayed off goal
                    rewards[:, i] = EnvParameters.IDLE_COST  # stop penalty
                elif action_status[i] == -3 or action_status[i] == -2 or action_status[i] == -1:
                    rewards[:, i] = EnvParameters.COLLISION_COST
                    num_collide += 1

            else:  # moving
                if action_status[i] == 1:  # reached goal
                    rewards[:, i] = EnvParameters.GOAL_REWARD
                elif action_status[i] == -2 or action_status[i] == -1 or action_status[i] == -3:
                    rewards[:, i] = EnvParameters.COLLISION_COST
                    num_collide += 1
                elif action_status[i] == 2:  # leave own goal
                    rewards[:, i] = EnvParameters.ACTION_COST
                    leave_goals += 1
                else:  # nothing happen
                    rewards[:, i] = EnvParameters.ACTION_COST

            state = self.observe(i + 1)
            obs[:, i, :, :, :] = state[0]
            vector[:, i, : 3] = state[1]

            next_valid_actions.append(self.list_next_valid_actions(i + 1, actions[i]))

        done, num_on_goal = self.world.task_done()
        if num_on_goal > self.max_on_goal:
            self.max_on_goal = num_on_goal
        if num_step >= EnvParameters.EPISODE_LEN - 1:
            done = True
        return obs, vector, rewards, done, next_valid_actions, blockings, valid_actions, num_blockings, \
            leave_goals, num_on_goal, self.max_on_goal, num_collide, action_status

    def create_rectangle(self, x, y, width, height, fill, permanent=False):
        """draw a rectangle to represent an agent"""
        ps = [(x, y), ((x + width), y), ((x + width), (y + height)), (x, (y + height))]
        rect = rendering.FilledPolygon(ps)
        rect.set_color(fill[0], fill[1], fill[2])
        rect.add_attr(rendering.Transform())
        if permanent:
            self.viewer.add_geom(rect)
        else:
            self.viewer.add_onetime(rect)

    def create_circle(self, x, y, diameter, size, fill, resolution=20):
        """draw a circle to represent a goal"""
        c = (x + size / 2, y + size / 2)
        dr = math.pi * 2 / resolution
        ps = []
        for i in range(resolution):
            x = c[0] + math.cos(i * dr) * diameter / 2
            y = c[1] + math.sin(i * dr) * diameter / 2
            ps.append((x, y))
        circ = rendering.FilledPolygon(ps)
        circ.set_color(fill[0], fill[1], fill[2])
        circ.add_attr(rendering.Transform())
        self.viewer.add_onetime(circ)

    def init_colors(self):
        """the colors of agents and goals"""
        c = {a + 1: hsv_to_rgb(np.array([a / float(self.num_agents), 1, 1])) for a in range(self.num_agents)}
        return c

    def _render(self, mode='human', close=False, screen_width=800, screen_height=800, action_probs=None):
        if close:
            return
        # values is an optional parameter which provides a visualization for the value of each agent per step
        size = screen_width / max(self.world.state.shape[0], self.world.state.shape[1])
        colors = self.init_colors()
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.reset_renderer = True
        if self.reset_renderer:
            self.create_rectangle(0, 0, screen_width, screen_height, (.6, .6, .6), permanent=True)
            for i in range(self.world.state.shape[0]):
                start = 0
                end = 1
                scanning = False
                write = False
                for j in range(self.world.state.shape[1]):
                    if self.world.state[i, j] != -1 and not scanning:  # free
                        start = j
                        scanning = True
                    if (j == self.world.state.shape[1] - 1 or self.world.state[i, j] == -1) and scanning:
                        end = j + 1 if j == self.world.state.shape[1] - 1 else j
                        scanning = False
                        write = True
                    if write:
                        x = i * size
                        y = start * size
                        self.create_rectangle(x, y, size, size * (end - start), (1, 1, 1), permanent=True)
                        write = False
        for agent in range(1, self.num_agents + 1):
            i, j = self.world.get_pos(agent)
            x = i * size
            y = j * size
            color = colors[self.world.state[i, j]]
            self.create_rectangle(x, y, size, size, color)
            i, j = self.world.get_goal(agent)
            x = i * size
            y = j * size
            color = colors[self.world.goals[i, j]]
            self.create_circle(x, y, size, size, color)
            if self.world.get_goal(agent) == self.world.get_pos(agent):
                color = (0, 0, 0)
                self.create_circle(x, y, size, size, color)
        if action_probs is not None:
            for agent in range(1, self.num_agents + 1):
                # take the a_dist from the given data and draw it on the frame
                a_dist = action_probs[agent - 1]
                if a_dist is not None:
                    for m in range(EnvParameters.N_ACTIONS):
                        dx, dy = self.world.get_dir(m)
                        x = (self.world.get_pos(agent)[0] + dx) * size
                        y = (self.world.get_pos(agent)[1] + dy) * size
                        s = a_dist[m] * size
                        self.create_circle(x, y, s, size, (0, 0, 0))
        self.reset_renderer = False
        result = self.viewer.render(return_rgb_array=mode == 'rgb_array')
        return result


        # before removing
        # local_agent_index is obstacle
        # after removing


