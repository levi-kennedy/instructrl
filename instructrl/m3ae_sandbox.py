import dataclasses
import os
import pprint
from functools import partial
from typing import Any, Callable, Optional

import absl.app
import absl.flags
import einops
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import torch
from absl import app, flags, logging
from flax import jax_utils
from flax import linen as nn
from flax.jax_utils import prefetch_to_device
from flax.training import checkpoints, common_utils, train_state
from flax.training.train_state import TrainState
from tqdm.auto import tqdm, trange

#from .envs import rollout
from .model import BC
from .utils import (
    JaxRNG,
    WandBLogger,
    define_flags_with_default,
    get_user_flags,
    load_pickle,
    next_rng,
    set_random_seed,
)

FLAGS_DEF = define_flags_with_default(
    seed=42,
    load_checkpoint="",
    logging=WandBLogger.get_default_config(),
    log_all_worker=False,
    model=BC.get_default_config(),
    window_size=4,
    episode_length=500,
    instruct="",
    dataset_name="reach_target",
    num_test_episodes=5,
    num_actions=8,
    obs_shape=(256, 256, 3),
)

def load_BC_model():
    FLAGS = FLAGS_DEF
   

    # Build the BC model
    model = BC(
            config_updates=FLAGS.model,
            num_actions=FLAGS.num_actions,
            obs_shape=FLAGS.obs_shape,
            patch_dim=16,
        )
    return model


if __name__ == "__main__":
    model = load_BC_model()
    print(model)