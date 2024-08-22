import run_lib
from absl import app  # Abseil Python Common Libraries, app 是Abseil Python应用程序的通用入口点
from absl import flags
from ml_collections.config_flags import config_flags
import logging
import os
import shutil

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")  # 定义一个字符串类型的参数，名字是workdir，没有默认值，描述是Work directory
flags.DEFINE_enum("mode", None, ["train", "sample"], "Running mode: train or sample")  # 获取字符串列表
flags.DEFINE_string("eval_folder", "eval", "The folder name for storing evaluation results")
flags.mark_flags_as_required(["workdir", "config", "mode"])  # 指定必须输入的参数


def main(argv):
  if FLAGS.mode == "train":
    
    if os.path.exists(FLAGS.workdir):
      overwrite = False
      response = input('The work dictionary exists, overwrite? (Y/N)')
      if response.upper() == 'Y':
        overwrite = True
      if overwrite:
        shutil.rmtree(FLAGS.workdir)
        os.makedirs(FLAGS.workdir)     
    else:  
      os.makedirs(FLAGS.workdir)
    
    # Set logger so that it outputs to both console and file
    # Make logging work for both disk and Google Cloud Storage
    gfile_stream = open(os.path.join(FLAGS.workdir, 'stdout.txt'), 'w')
    
    handler = logging.StreamHandler(gfile_stream)  # 创建StreamHandler对（控制台输出日志）象
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')  # 根据指定的形式，创建一个格式器
    handler.setFormatter(formatter)  # 为handler指定一个格式器
    logger = logging.getLogger()  # 创建一个logger对象，名字可以不填
    logger.addHandler(handler)  # 用于加载handler对象
    logger.setLevel('INFO')
    # Run the training pipeline
    run_lib.train(FLAGS.config, FLAGS.workdir)
  elif FLAGS.mode == "sample":
    # Run the evaluation pipeline
    run_lib.sample(FLAGS.config, FLAGS.workdir)
  else:
    raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
  app.run(main)
  
  
  # python main.py --config /home/liaohx/score_based_v1/configs/ve/ncsn/cine.py --workdir /data/liaohx/score_based --mode train