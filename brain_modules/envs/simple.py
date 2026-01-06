import gymnasium as gym
import torch as tc


class RoomEnv(gym.Env):
    batch_size = 512
    env_dim = 1
    act_dim = 1

    @property
    def obs(s):
        return s.pos.detach()

    def reset(s):
        s.pos = tc.rand(s.batch_size, s.env_dim)
        return s.obs, {}

    def step(s, act):
        velocity = act
        s.pos = tc.clamp(s.pos + velocity, 0.0, 1.0)
        rew, term, trunc, info = 0, False, False, {}
        return s.obs, rew, term, trunc, info

    def rand_act(s):
        return 0.05 * tc.randn(s.batch_size, s.act_dim)
