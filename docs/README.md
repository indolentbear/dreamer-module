## Instructions

Get dependencies:

```
pip3 install --user tensorflow-gpu==2.2.0
pip3 install --user tensorflow_probability==0.10.1
pip3 install --user pandas
pip3 install --user matplotlib gym imageio imageio-ffmpeg
pip3 install -e .
cudatoolkit==11.0.221
pip3 install --user git+git://github.com/deepmind/dm_control.git
```

Train the agent:

```
python3 dreamer.py --logdir ../data/logdir/dmc_walker_walk/dreamer/1 --task dmc_walker_walk
```

Generate plots:

```
python3 plotting.py --indir ../data --outdir ./plots --xaxis step --yaxis test/return --bins 3e4

python3 --expl epsilon_greedy --horizon 10 --kl_scale 0.1 --action_dist onehot --expl_amount 0.4 --expl_min 0.1 --expl_decay 100000 --pcont 1 --time_limit 1000000
```

Graphs and GIFs:

```
tensorboard --logdir ./logdir
python3 -m tensorboard.main --logdir=./logdir
```

commond run:
```
 nohup python3 ./train.py >> ./tari_Boxing.out 
```

