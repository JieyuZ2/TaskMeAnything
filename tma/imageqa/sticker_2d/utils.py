import random

import numpy as np
from PIL import Image

grid_options = [2, 3]
grid_mappings = {
	2:
		{
			'top left'    : 0,
			'top right'   : 1,
			'bottom left' : 2,
			'bottom right': 3
		},
	3:
		{
			'top left'     : 0,
			'top middle'   : 1,
			'top right'    : 2,
			'middle left'  : 3,
			'middle'       : 4,
			'middle right' : 5,
			'bottom left'  : 6,
			'bottom middle': 7,
			'bottom right' : 8
		}
}

relative_positions = ['left', 'right', 'top', 'bottom', 'top left', 'top right', 'bottom left', 'bottom right']
relative_position_phrase = {
	'left'        : 'to the left of',
	'right'       : 'to the right of',
	'top'         : 'above',
	'bottom'      : 'below',
	'top left'    : 'above and to the left of',
	'top right'   : 'above and to the right of',
	'bottom left' : 'below and to the left of',
	'bottom right': 'below and to the right of'
}


def relative_grid(grid_size, grid, reference_pos):
	if 'right' in reference_pos:
		if grid % grid_size == 0: return -1
		grid = grid - 1
	if 'left' in reference_pos:
		if grid % grid_size == grid_size - 1: return -1
		grid = grid + 1
	if 'top' in reference_pos:
		if grid + grid_size >= grid_size * grid_size: return -1
		grid = grid + grid_size
	if 'bottom' in reference_pos:
		if grid - grid_size < 0: return -1
		grid = grid - grid_size
	return grid


def does_overlap(box1, box2):
	# Returns True if box1 and box2 overlap, False otherwise
	x1, y1, x2, y2 = box1
	x3, y3, x4, y4 = box2
	return not (x2 < x3 or x4 < x1 or y2 < y3 or y4 < y1)


def sample_bounding_boxes(num_objects, H, W, size_range=(0.3, 0.45)):
	while True:
		frac = random.uniform(*size_range)
		boxes = []
		count = 0
		num_chances = 5
		while len(boxes) < num_objects and count < num_chances:
			box_w = int(frac * W)
			box_h = int(frac * H)
			box_x = random.randint(0, W - box_w)
			box_y = random.randint(0, H - box_h)
			new_box = (box_x, box_y, box_x + box_w, box_y + box_h)
			if not any(does_overlap(new_box, box) for box in boxes):
				boxes.append(new_box)
			count += 1
		if count >= num_chances:
			continue
		return boxes


def grid_to_box(H, W, grid_size, grid_index, grid_H, grid_W):
	grid_height = H // grid_size
	grid_width = W // grid_size

	# grid_x, grid_y = np.unravel_index(grid_index, (grid_size, grid_size))
	grid_y, grid_x = np.unravel_index(grid_index, (grid_size, grid_size))

	box_x = grid_x * grid_width
	box_y = grid_y * grid_height
	box_w = grid_W * grid_width
	box_h = grid_H * grid_height
	return (box_x, box_y, box_x + box_w, box_y + box_h)


def paste_image(background, obj, box):
	obj = obj.resize((box[2] - box[0], box[3] - box[1]))
	background.paste(obj, box=box, mask=obj)


def make_image(metadata, H=512, W=512):
	# sample bounding boxes
	grid_size = metadata["grid number"]
	object_paths = metadata["object paths"]
	assert len(metadata["objects"]) <= (grid_size ** 2)
	boxes = [grid_to_box(H, W, grid_size, x, 1, 1) for x in metadata["grids"]]

	im_target = Image.new("RGBA", (W, H), 'WHITE')  # you can load this as a background image if you want

	for view, box in zip(object_paths, boxes):
		obj = Image.open(view)
		paste_image(im_target, obj, box)

	return im_target.convert('RGB')
