"""Train constructed model using tf.estimator API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf
from tensorflow.python import debug as tf_debug
from config import TrainingConfig, ModelConfig
import shutil
from model import model_fn
from data_utils.dataset_util import input_fn

# if TrainingConfig.model_type:
#     os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# else:
#     os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def main(unused_argv):
  # Using the Winograd non-fused algorithm provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  if TrainingConfig.clean_model_dir:
    shutil.rmtree(TrainingConfig.model_dir, ignore_errors=True)


  # Set up a RunConfig to only save checkpoints once per training cycle.
  run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9)
  model = tf.estimator.Estimator(
      model_fn=model_fn,
      model_dir=TrainingConfig.model_dir,
      config=run_config,
      params={
      })

  for _ in range(TrainingConfig.train_epochs // TrainingConfig.epochs_per_eval):
    tensors_to_log = {
      'learning_rate': 'learning_rate',
      'total_loss': 'total_loss',
      'classification_loss': 'classification_loss',
      'segmentation_loss' : 'segmentation_loss',
      'segmentation_px_acc': 'segmentation_px_acc',
      'classification_acc': 'classification_acc',
      # 'mean_iou': 'mean_iou',
    }

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=1)
    train_hooks = [logging_hook]
    eval_hooks = None

    if ModelConfig.debug:
      debug_hook = tf_debug.LocalCLIDebugHook()
      train_hooks.append(debug_hook)
      eval_hooks = [debug_hook]

    tf.logging.info("Start training.")
    model.train(
        input_fn=lambda: input_fn(True, ModelConfig.data_dir, TrainingConfig.batch_size, TrainingConfig.epochs_per_eval),
        hooks=train_hooks,
        # steps=1  # For debugging
    )

    tf.logging.info("Start evaluation.")
    
    # Evaluate the model and print results
    eval_results = model.evaluate(
        # Batch size must be 1 for testing because the images' size differs
        input_fn=lambda: input_fn(False, ModelConfig.data_dir, 1),
        hooks=eval_hooks,
        # steps=1  # For debug
    )
    print(eval_results)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(main=main)
