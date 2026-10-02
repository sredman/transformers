# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tempfile
import unittest
from dataclasses import dataclass, field

from transformers import Trainer
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.trainer_utils import SaveStrategy


@dataclass
class _Args:
    """The attributes `_init_training_state` reads off `self.args`."""

    logging_steps: int = 500
    eval_steps: int = 500
    save_steps: int = 500
    n_gpu: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    ignore_data_skip: bool = False
    save_strategy: SaveStrategy = SaveStrategy.STEPS


@dataclass
class _CallbackHandler:
    callbacks: list = field(default_factory=list)


class _StubTrainer:
    """Enough of a Trainer for `_init_training_state`, without building a model."""

    _init_training_state = Trainer._init_training_state

    def __init__(self, args):
        self.args = args
        self.state = TrainerState()
        self.control = TrainerControl()
        self.callback_handler = _CallbackHandler()
        self._train_batch_size = 1
        self.hp_name = None
        self._trial = None

    def _load_callback_state(self):
        pass

    def is_local_process_zero(self):
        return True

    def is_world_process_zero(self):
        return True


def _checkpoint_written_with(save_steps, eval_steps, logging_steps, directory, global_step=617):
    state = TrainerState()
    state.compute_steps(
        _Args(save_steps=save_steps, eval_steps=eval_steps, logging_steps=logging_steps), 1000
    )
    state.global_step = global_step
    state.train_batch_size = 1
    state.save_to_json(os.path.join(directory, "trainer_state.json"))


class TrainerResumeCadenceTest(unittest.TestCase):
    """Resuming must honour this run's save/eval cadence, not the checkpoint's.

    `TrainerState` is restored wholesale from `trainer_state.json`, and
    `DefaultFlowCallback` reads `state.save_steps`. Without re-applying the
    current arguments, a resume silently keeps the interval the checkpoint was
    written with -- which is exactly the interval the user just changed.
    """

    def test_resume_honours_the_current_save_and_eval_steps(self):
        with tempfile.TemporaryDirectory() as checkpoint:
            _checkpoint_written_with(44, 44, 44, checkpoint)
            trainer = _StubTrainer(_Args(save_steps=11, eval_steps=22, logging_steps=33))

            trainer._init_training_state(
                max_steps=1000,
                num_update_steps_per_epoch=100,
                num_train_epochs=10,
                resume_from_checkpoint=checkpoint,
                trial=None,
            )

            self.assertEqual(trainer.state.save_steps, 11)
            self.assertEqual(trainer.state.eval_steps, 22)
            self.assertEqual(trainer.state.logging_steps, 33)

    def test_resume_resolves_ratio_arguments_against_max_steps(self):
        with tempfile.TemporaryDirectory() as checkpoint:
            _checkpoint_written_with(44, 44, 44, checkpoint)
            trainer = _StubTrainer(_Args(save_steps=0.1, eval_steps=0.25, logging_steps=0.5))

            trainer._init_training_state(
                max_steps=1000,
                num_update_steps_per_epoch=100,
                num_train_epochs=10,
                resume_from_checkpoint=checkpoint,
                trial=None,
            )

            self.assertEqual(trainer.state.save_steps, 100)
            self.assertEqual(trainer.state.eval_steps, 250)

    def test_resume_still_restores_progress(self):
        with tempfile.TemporaryDirectory() as checkpoint:
            _checkpoint_written_with(44, 44, 44, checkpoint, global_step=617)
            trainer = _StubTrainer(_Args(save_steps=11))

            epochs_trained, _ = trainer._init_training_state(
                max_steps=1000,
                num_update_steps_per_epoch=100,
                num_train_epochs=10,
                resume_from_checkpoint=checkpoint,
                trial=None,
            )

            self.assertEqual(trainer.state.global_step, 617)
            self.assertEqual(epochs_trained, 6)

    def test_fresh_run_is_unaffected(self):
        trainer = _StubTrainer(_Args(save_steps=11, eval_steps=22))

        trainer._init_training_state(
            max_steps=1000,
            num_update_steps_per_epoch=100,
            num_train_epochs=10,
            resume_from_checkpoint=None,
            trial=None,
        )

        self.assertEqual(trainer.state.save_steps, 11)
        self.assertEqual(trainer.state.global_step, 0)


if __name__ == "__main__":
    unittest.main()
