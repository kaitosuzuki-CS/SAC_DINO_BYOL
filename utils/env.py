import os
from collections import deque

import numpy as np
import torch

os.environ["MUJOCO_GL"] = "egl" if torch.cuda.is_available() else "osmesa"

from dm_control import suite
from dm_control.suite.wrappers import pixels


class ActionRepeatWrapper:
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def __getattr__(self, name):
        return getattr(self._env, name)

    def action_spec(self):
        return self._env.action_spec()

    def step(self, action):
        reward = 0.0
        discount = 1.0

        for _ in range(self._num_repeats):
            time_step = self._env.step(action)
            reward = reward + (time_step.reward or 0.0) * discount
            discount = discount * time_step.discount

            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def reset(self):
        return self._env.reset()


class FrameStackWrapper:
    def __init__(self, env, num_frames, pixels_key="pixels"):
        self._env = env
        self._num_frames = num_frames
        self._pixels_key = pixels_key

        self.frames = deque([], maxlen=num_frames)

    def _get_obs(self, time_step):
        frame = time_step.observation[self._pixels_key]
        frame = frame.transpose(2, 0, 1)

        if len(self.frames) == 0:
            for _ in range(self._num_frames):
                self.frames.append(frame)
        else:
            self.frames.append(frame)

        frames = np.concatenate(list(self.frames), axis=0)

        return frames

    def action_spec(self):
        return self._env.action_spec()

    def step(self, action):
        time_step = self._env.step(action)

        return (
            self._get_obs(time_step),
            time_step.reward,
            time_step.last(),
            time_step.discount,
        )

    def reset(self):
        self.frames.clear()
        time_step = self._env.reset()

        return (
            self._get_obs(time_step),
            time_step.reward,
            time_step.last(),
            time_step.discount,
        )


class MaxEpisodeStepsWrapper:
    def __init__(self, env, max_steps):
        self._env = env
        self._max_steps = max_steps

        self.step_count = 0

    def __getattr__(self, name):
        return getattr(self._env, name)

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        self.step_count = 0
        return self._env.reset()

    def step(self, action):
        obs, reward, done, discount = self._env.step(action)
        self.step_count += 1
        if self.step_count >= self._max_steps:
            done = True

        return obs, reward, done, discount


def create_environment(
    domain_name, task_name, action_repeat, horizon, frame_stack, image_size
):
    env = suite.load(domain_name=domain_name, task_name=task_name)
    env = ActionRepeatWrapper(env, action_repeat)
    env = pixels.Wrapper(
        env,
        pixels_only=True,
        render_kwargs={"height": image_size, "width": image_size, "camera_id": 0},
    )
    env = FrameStackWrapper(env, num_frames=frame_stack)
    # env = MaxEpisodeStepsWrapper(env, horizon)

    return env
