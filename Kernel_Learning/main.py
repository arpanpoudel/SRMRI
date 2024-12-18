import torch
from ml_collections.config_flags import config_flags
import os
import logging
from absl import flags
from absl import app
import tensorflow as tf
import run_mri
from model import utils as mutils

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum("mode", None, ["train", "train_regression", "eval"], "Running mode: train, train_regression, or eval")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.mark_flags_as_required(["workdir", "config", "mode"])


def main(args):
    #print(FLAGS.config)
    if FLAGS.mode == "train":
         # Create the working directory
        tf.io.gfile.makedirs(FLAGS.workdir)
        # Set logger so that it outputs to both console and file
        # Make logging work for both disk and Google Cloud Storage
        gfile_stream = open(os.path.join(FLAGS.workdir, 'stdout.txt'), 'w')
        handler = logging.StreamHandler(gfile_stream)
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel('INFO')
        
        run_mri.train(FLAGS.config, FLAGS.workdir)

if __name__ == "__main__":
    app.run(main)