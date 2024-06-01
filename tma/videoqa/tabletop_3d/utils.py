import json
import os
import subprocess

from ..metadata import ObjaverseVideoMetaData

grid_options = [2, 3]

grid_mappings = {
	2:
		{
			'back left'  : 0,
			'back right' : 1,
			'front left' : 2,
			'front right': 3
		},
	3:
		{
			'back left'   : 0,
			'back middle' : 1,
			'back right'  : 2,
			'middle left' : 3,
			'middle'      : 4,
			'middle right': 5,
			'front left'  : 6,
			'front middle': 7,
			'front right' : 8
		}
}

relative_positions = ['left', 'right', 'back', 'front', 'back left', 'back right', 'front left', 'front right']
relative_position_phrase = {
	'left'       : 'to the left of',
	'right'      : 'to the right of',
	'back'       : 'behind',
	'front'      : 'in front of',
	'back left'  : 'behind and to the left of',
	'back right' : 'behind and to the right of',
	'front left' : 'in front and to the left of',
	'front right': 'in front and to the right of'
}
reverse_relative_positions = {
	'left'       : 'right',
	'right'      : 'left',
	'back'       : 'front',
	'front'      : 'back',
	'front left' : 'back right',
	'front right': 'back left',
	'back left'  : 'front right',
	'back right' : 'front left'
}


def relative_grid(grid_size, grid, reference_pos):
	if 'right' in reference_pos:
		if grid % grid_size == 0: return -1
		grid = grid - 1
	if 'left' in reference_pos:
		if grid % grid_size == grid_size - 1: return -1
		grid = grid + 1
	if 'back' in reference_pos:
		if grid + grid_size >= grid_size * grid_size: return -1
		grid = grid + grid_size
	if 'front' in reference_pos:
		if grid - grid_size < 0: return -1
		grid = grid - grid_size
	return grid


import tempfile
import diskcache

run_script_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "run_blender.py")


def make_video(scene_json, metadata: ObjaverseVideoMetaData, VIDEO_H, VIDEO_W):
	device = metadata.render_device
	blender_cache = metadata.blender_cache
	assert len(scene_json["objects"]) <= (scene_json["grid number"] ** 2)
	scene_json["VIDEO_H"] = VIDEO_H
	scene_json["VIDEO_W"] = VIDEO_W

	with diskcache.Cache(blender_cache, size_limit=100 * (2 ** 30)) as cache:
		key = json.dumps(scene_json, sort_keys=True)
		video = cache.get(key, None)
		if video is None:
			with (tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as tmp_video,
				  tempfile.NamedTemporaryFile(delete=True, suffix=".json") as tmp_json):
				json.dump(scene_json, open(tmp_json.name, 'w'))

				env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device))
				command = (
					f"{metadata.blender_path} -b -noaudio --python {run_script_path} -- "
					f"--save_video_path {tmp_video.name} "
					f"--json_file {tmp_json.name}"
				)

				subprocess.run(command, shell=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

				with open(tmp_video.name, 'rb') as video_file:
					video = video_file.read()  # save video to a binary files
				cache.set(key, video)

	return video
