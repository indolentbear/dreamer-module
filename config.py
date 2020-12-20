import pathlib
import tool

def define_config():
  config = tool.AttrDict()
  # General.
  config.logdir = pathlib.Path('./data/log_test/atari_boxing/dreamer/0')
  config.seed = 0
  config.steps = 1.5e7       # 训练总步长
  config.eval_every = 1e5
  config.log_every = 1e3
  config.log_scalars = True
  config.log_images = True
  config.gpu_growth = True   # True
  config.precision = 16
  # Environment.
  config.task = 'atari_boxing'
  config.envs = 1               # server-40 2*8
  config.parallel = 'none'     # none | thread | process
  config.action_repeat = 4      # atari 4, mujoco 4
  config.time_limit = 1e6       # atari 1e6,mujoco 1000, episode 长度限制
  config.prefill = 5000
  config.eval_noise = 0.001     # atari 0.001,mujoco 0.0
  config.clip_rewards = 'tanh'  # atari tanh, mujoco none
  # Model.
  config.deter_size = 200   # model_size/hidden_size 确定性
  config.stoch_size = 30    # 30  随机性
  config.num_units = 400    # 400
  config.dense_act = 'elu'
  config.cnn_act = 'relu'  #
  config.cnn_depth = 32     # 32
  config.pcont = True      # atari True, mujoco False 这是啥啊 # discrete ? continuous
  config.free_nats = 3.0    #
  config.kl_scale = 0.1     # atari 0.1 mujoco 1.0 ,beta
  config.pcont_scale = 0.99 # atari 0.99 mujoco 10.0
  config.weight_decay = 0.0
  config.weight_decay_pattern = r'.*'
  # Training.
  config.batch_size = 50        # 一次训练所选取的样本/episode数
  config.batch_length = 50      # 每个样本/episode长度
  config.train_every = 1000     # 每N步进行一次训练
  config.train_steps = 100
  config.pretrain = 100         # 预训练
  config.model_lr = 6e-4
  config.value_lr = 8e-5
  config.actor_lr = 8e-5
  config.grad_clip = 100.0      # 梯度限幅
  config.dataset_balance = False
  # Behavior.
  config.discount = 1.0         # atari 1.0, mujoco 0.99
  config.disclam = 0.95         # 非0即蒙特卡洛折扣回报
  config.horizon = 10           # atari 10, mujoco 15
  config.action_dist = 'onehot' # atari onehot, mujoco tanh_normal
  config.action_init_std = 5.0  # 5.0
  config.expl = 'epsilon_greedy'# exploration : atari epsilon_greedy, mujoco epsilon_greedy
  config.expl_amount = 0.4      # atari 0.4, mujoco 0.3
  config.expl_decay = 1e5       # atari 2e5, mujoco 0.0
  config.expl_min = 0.1         # atari 0.1 in paper/0.01, mujoco 0.0
  return config