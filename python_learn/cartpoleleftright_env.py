import math
import numpy as np
from typing import Optional, Union

from gym.envs.classic_control.cartpole import CartPoleEnv
from gym import spaces, logger
import pygame
from pygame import gfxdraw
import matplotlib.pyplot as plt


class CartPoleLeftRightEnv(CartPoleEnv):
    def __init__(self, config=None):
        super(CartPoleLeftRightEnv, self).__init__()
        # set default angle threshold to 15°
        self.theta_threshold_radians = 15 * 2 * math.pi / 360
        self.x_threshold = 2.4
        # for _goal_reward function
        self.x_reward_threshold = 0.4
        self.x_reward_interval = self.x_threshold - self.x_reward_threshold
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
                1.0
            ],
            dtype=np.float32,
        )
        # set low obs value for goal
        low = -high
        low[4] = 0.0
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        # set reward function
        reward_fn = 'goal'
        if config is not None and 'reward_fn' in config:
            reward_fn = config['reward_fn']
            if reward_fn not in ['goal', 'smooth']:
                reward_fn = 'goal'
        if reward_fn == 'goal':
            self.reward_fn = self._goal_reward
        elif reward_fn == 'smooth':
            self.reward_fn = self._smooth_reward
        self.time_limit = None
        if 'time_limit' in config:
            self.time_limit = config['time_limit']

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        self.cur_step += 1
        # add goal extraction
        x, x_dot, theta, theta_dot, goal = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (
                       force + self.polemass_length * theta_dot ** 2 * sintheta
               ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        # add goal
        self.state = (x, x_dot, theta, theta_dot, goal)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )
        # check for time limit
        time_done = False
        if self.time_limit is not None:
            if self.cur_step > self.time_limit:
                done = True
                time_done = True

        if not done:
            reward = self.reward_fn(goal, x)
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            if time_done:
                reward = 0.0
            else:
                if goal == 0:
                    reward = -1.0 if x > 0 else -0.5
                else:
                    reward = -1.0 if x < 0 else -0.5
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ):
        super(CartPoleEnv, self).reset()
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(5,))
        # set initial position exactly to 0.0
        # self.state[0] = 0.0
        # random goal selection
        self.state[4] = self.np_random.integers(2)
        self.cur_goal = self.state[4]
        self.cur_step = 0
        self.steps_beyond_done = None
        if not return_info:
            return np.array(self.state, dtype=np.float32)
        else:
            return np.array(self.state, dtype=np.float32), {}

    '''
    Sparse goal reward.
    Agent receives 1.0 if it is on the correct side of platform due to goal
    and it is in specified interval from the bound.
    '''
    def _goal_reward(self, goal, cart_pos):
        if goal == 1.0:
            return 1.0 if cart_pos > self.x_reward_interval else 0.0
        else:
            return 1.0 if cart_pos < -self.x_reward_interval else 0.0

    '''
    Smooth reward.
    Agent receives reward from 0.0 to 1.0 when it is near to correct bound.
    '''
    def _smooth_reward(self, goal, cart_pos):
        if abs(cart_pos) > self.x_threshold:
            return 0
        else:
            if goal == 1.0:
                r = np.exp(cart_pos - 2.4)
            else:
                r = np.exp(-cart_pos - 2.4)
        return r

    # add rendering of goal as red line
    def render(self, mode="human"):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, screen_width, carty, (0, 0, 0))

        # draw goal
        goal_coords = [(0, 100), (0, 150), (5, 150), (5, 100)] if self.cur_goal == 0 else \
            [(screen_width-6, 100), (screen_width-6, 150), (screen_width-1, 150), (screen_width-1, 100)]
        gfxdraw.filled_polygon(self.surf, goal_coords, (255, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.isopen

def plot_reward(fn, x_thr):
    x = np.linspace(-x_thr, x_thr, 100)
    y1 = np.array([fn(0, t) for t in x])
    y2 = np.array([fn(1, t) for t in x])
    print(y2[0])
    print(y1[0])
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.show()


if __name__ == "__main__":
    env = CartPoleLeftRightEnv(config={"reward_fn": "goal"})
    plot_reward(env._goal_reward, env.x_threshold)
    plot_reward(env._smooth_reward, env.x_threshold)
    for _ in range(10):
        obs = env.reset()
        goal = obs[4]
        score = 0.0
        ep_len = 0
        for _ in range(500):
            obs, reward, done, info = env.step(env.action_space.sample())
            env.render()
            score += reward
            ep_len += 1
            if done:
                break
        print(ep_len, " ", score, "goal", goal)
