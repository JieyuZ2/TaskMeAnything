import json
import subprocess

from ..metadata import Objaverse3DMetaData

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


import os
import tempfile
import io, base64
from PIL import Image
import diskcache

run_script_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "run_blender.py")


def image_to_base64(pil_image):
	import io
	import base64
	img_byte_arr = io.BytesIO()
	pil_image.save(img_byte_arr, format='PNG')
	img_byte_arr = img_byte_arr.getvalue()
	base64_str = base64.b64encode(img_byte_arr).decode('utf-8')
	return base64_str


def make_image(scene_json, metadata: Objaverse3DMetaData, H=512, W=512):
	device = metadata.render_device
	blender_cache = metadata.blender_cache
	assert len(scene_json["objects"]) <= (scene_json["grid number"] ** 2)
	scene_json["H"] = H
	scene_json["W"] = W

	with diskcache.Cache(blender_cache, size_limit=100 * (2 ** 30)) as cache:
		key = json.dumps(scene_json, sort_keys=True)
		base64_str = cache.get(key, None)
		if base64_str is None:
			with (tempfile.NamedTemporaryFile(delete=True, suffix=".png") as tmp_image,
				  tempfile.NamedTemporaryFile(delete=True, suffix=".json") as tmp_json):
				json.dump(scene_json, open(tmp_json.name, 'w'))

				env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device))
				command = (
					f"{metadata.blender_path} -b -noaudio --python {run_script_path} -- "
					f"--save_image_path {tmp_image.name} "
					f"--json_file {tmp_json.name}"
				)

				subprocess.run(command, shell=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

				img = Image.open(tmp_image.name).convert("RGB")
				cache.set(key, image_to_base64(img))
		else:
			img = Image.open(io.BytesIO(base64.decodebytes(bytes(base64_str, "utf-8"))))

	return img
