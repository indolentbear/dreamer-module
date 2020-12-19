import tensorflow.compat.v1 as tf1
import numpy as np
import tensorflow as tf
import json
import tool
import datetime
import io
import pathlib
import uuid

def summarize_episode(episode, config, datadir, writer, prefix):
	episodes, steps = count_episodes(datadir)
	length = (len(episode['reward']) - 1) * config.action_repeat
	ret = episode['reward'].sum()
	print(f'{prefix.title()} episode of length {length} with return {ret:.1f}.')
	metrics = [
		(f'{prefix}/return', float(episode['reward'].sum())),
		(f'{prefix}/length', len(episode['reward']) - 1),
		(f'episodes', episodes)]
	step = count_episodes(datadir)[1] * config.action_repeat
	with (config.logdir / 'metrics.jsonl').open('a') as f:
		f.write(json.dumps(dict([('step', step)] + metrics)) + '\n')
	with writer.as_default():  # Env might run in a different thread.
		tf.summary.experimental.set_step(step)
		[tf.summary.scalar('sim/' + k, v) for k, v in metrics]
		if prefix == 'test':
			video_summary(f'sim/{prefix}/video', episode['image'][None])

def nest_summary(structure):
	if isinstance(structure, dict):
		return {k: nest_summary(v) for k, v in structure.items()}
	if isinstance(structure, list):
		return [nest_summary(v) for v in structure]
	if hasattr(structure, 'shape'):
		return str(structure.shape).replace(', ', 'x').strip('(), ')
	return '?'

def graph_summary(writer, fn, *args):
	step = tf.summary.experimental.get_step()
	def inner(*args):
		tf.summary.experimental.set_step(step)
		with writer.as_default():
			fn(*args)
	return tf.numpy_function(inner, args, [])

def video_summary(name, video, step=None, fps=20):
	name = name if isinstance(name, str) else name.decode('utf-8')
	if np.issubdtype(video.dtype, np.floating):
		video = np.clip(255 * video, 0, 255).astype(np.uint8)
	B, T, H, W, C = video.shape
	try:
		frames = video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
		summary = tf1.Summary()
		image = tf1.Summary.Image(height=B * H, width=T * W, colorspace=C)
		image.encoded_image_string = encode_gif(frames, fps)
		summary.value.add(tag=name + '/gif', image=image)
		tf.summary.experimental.write_raw_pb(summary.SerializeToString(), step)
	except (IOError, OSError) as e:
		print('GIF summaries require ffmpeg in $PATH.', e)
		frames = video.transpose((0, 2, 1, 3, 4)).reshape((1, B * H, T * W, C))
		tf.summary.image(name + '/grid', frames, step)

def encode_gif(frames, fps):
	from subprocess import Popen, PIPE
	h, w, c = frames[0].shape
	pxfmt = {1: 'gray', 3: 'rgb24'}[c]
	cmd = ' '.join([
		f'ffmpeg -y -f rawvideo -vcodec rawvideo',
		f'-r {fps:.02f} -s {w}x{h} -pix_fmt {pxfmt} -i - -filter_complex',
		f'[0:v]split[x][z];[z]palettegen[y];[x]fifo[x];[x][y]paletteuse',
		f'-r {fps:.02f} -f gif -'])
	proc = Popen(cmd.split(' '), stdin=PIPE, stdout=PIPE, stderr=PIPE)
	for image in frames:
		proc.stdin.write(image.tostring())
	out, err = proc.communicate()
	if proc.returncode:
		raise IOError('\n'.join([' '.join(cmd), err.decode('utf8')]))
	del proc
	return out

def save_episodes(directory, episodes):
	directory = pathlib.Path(directory).expanduser()
	directory.mkdir(parents=True, exist_ok=True)
	timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
	for episode in episodes:
		identifier = str(uuid.uuid4().hex)
		length = len(episode['reward'])
		filename = directory / f'{timestamp}-{identifier}-{length}.npz'
		with io.BytesIO() as f1:
			np.savez_compressed(f1, **episode)
			f1.seek(0)
			with filename.open('wb') as f2:
				f2.write(f1.read())

def count_episodes(directory):
	filenames = directory.glob('*.npz')
	lengths = [int(n.stem.rsplit('-', 1)[-1]) - 1 for n in filenames]
	episodes, steps = len(lengths), sum(lengths)
	return episodes, steps