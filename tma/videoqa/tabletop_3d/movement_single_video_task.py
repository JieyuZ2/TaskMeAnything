from tqdm import tqdm

from .single_video_task import GridVideoTaskGenerator
from .utils import relative_positions
from ..metadata import ObjaverseVideoMetaData as MetaData
from ...constant import NUM_OPTIONS, VIDEO_FPS, VIDEO_NUM_FRAMES
from ...task_store import TaskStore

grid_options = [2]
DEFAULT_OBJECT_SIZE_MULTIPLIER = 1.3

moving_options = {'left', 'right', 'up', 'down'}


def direction_to_keyframes(direction):
	if direction == 'left':
		return [{}, {}, {}, {}, {'movement': (0, 0.35)}]
	elif direction == 'right':
		return [{}, {}, {}, {}, {'movement': (0, -0.35)}]
	elif direction == 'up':
		return [{}, {}, {}, {}, {'movement': (0.45, 0)}]
	elif direction == 'down':
		return [{}, {}, {}, {}, {'movement': (-0.45, 0)}]


class MovementVideoGridTaskGenerator(GridVideoTaskGenerator):
	def _make_video_metadata(self, grid_size, grids, queries, remaining_query=..., target_object_moving_direction='left', are_other_objects_moving="No", object_size_multiplier=DEFAULT_OBJECT_SIZE_MULTIPLIER):
		objects = [self.metadata.sample(self.rng, 1, "object", q) for q in queries]
		remaining_grids = [g for g in range(grid_size ** 2) if g not in grids]
		for _ in remaining_grids:
			uid = self.metadata.sample(self.rng, 1, "object", remaining_query)
			objects.append(uid)

		remaining_moving_direction = list(moving_options - {target_object_moving_direction})
		keyframes = [direction_to_keyframes(target_object_moving_direction)]
		if are_other_objects_moving == "Yes":
			remaining_keyframes = [direction_to_keyframes(self.rng.choice(remaining_moving_direction, size=1)) for _ in range(len(remaining_grids))]
		else:
			remaining_keyframes = [[{}, {}, {}, {}, {}] for _ in range(len(remaining_grids))]

		object_path = {k: self.metadata.get_object_path(k) for k in objects}
		angles = [self.metadata.sample_object_angle(self.rng, obj) for obj in objects]

		video_metadata = {
			'grid number'     : grid_size,
			'objects'         : objects,
			'object_path'     : object_path,
			'object_angles'   : angles,
			'grids'           : grids + remaining_grids,
			'blender_config'  : self.metadata.sample_blender_configuration(self.rng),
			'fps'             : VIDEO_FPS,
			'total_num_frames': VIDEO_NUM_FRAMES,
			'sizes'           : [object_size_multiplier for _ in objects],
			'keyframes'       : keyframes + remaining_keyframes,
		}
		return video_metadata


class WhatMovementVideoGridTaskGenerator(MovementVideoGridTaskGenerator):
	schema = {
		'task type'           : 'str',
		'grid number'             : 'int',
		'target category'         : 'str',
		'absolute position'       : 'str',
		'attribute type'          : 'str',
		'attribute value'         : 'str',
		'moving direction'        : 'str',
		'are other objects moving': 'str'
	}

	def enumerate_task_plans(self, task_store: TaskStore):
		for target_category in tqdm(self.metadata.categories, desc="enumerating [what move video] task"):
			attribute_dict = self.metadata.get_category_attribute_dict(target_category)
			for attribute_type, attribute_values in attribute_dict.items():
				for attribute_value in attribute_values:
					for grid_size in grid_options:
						for absolute_pos in self.grid_mappings[grid_size]:
							for target_object_moving_direction in moving_options:
								task_plan = {
									'task type'           : 'what move video',
									'grid number'             : grid_size,
									'target category'         : target_category,
									'absolute position'       : absolute_pos,
									'attribute type'          : attribute_type,
									'attribute value'         : attribute_value,
									'moving direction'        : target_object_moving_direction,
									'are other objects moving': "Yes"
								}
								task_store.add(task_plan)

								task_plan = {
									'task type'           : 'what move video',
									'grid number'             : grid_size,
									'target category'         : target_category,
									'absolute position'       : absolute_pos,
									'attribute type'          : attribute_type,
									'attribute value'         : attribute_value,
									'moving direction'        : target_object_moving_direction,
									'are other objects moving': "No"
								}
								task_store.add(task_plan)

			for grid_size in grid_options:
				for absolute_pos in self.grid_mappings[grid_size]:
					for target_object_moving_direction in moving_options:
						task_plan = {
							'task type'           : 'what move video',
							'grid number'             : grid_size,
							'target category'         : target_category,
							'absolute position'       : absolute_pos,
							'moving direction'        : target_object_moving_direction,
							'are other objects moving': "Yes"
						}
						task_store.add(task_plan)

						task_plan = {
							'task type'           : 'what move video',
							'grid number'             : grid_size,
							'target category'         : target_category,
							'absolute position'       : absolute_pos,
							'moving direction'        : target_object_moving_direction,
							'are other objects moving': "No"
						}
						task_store.add(task_plan)

	def _generate_task(self, task_plan):
		grid_size = task_plan['grid number']
		target_category = task_plan['target category']
		absolute_pos = task_plan['absolute position']
		grids = [self.grid_mappings[grid_size][absolute_pos]]
		target_object_moving_direction = task_plan['moving direction']

		if task_plan['are other objects moving'] == "Yes":
			question = f"What is the object that is moving {target_object_moving_direction} in the video?"
		else:
			question = f"What is the moving object in the video?"

		queries = [self._get_target_object_query(task_plan)]

		remaining_query = self.metadata.and_query([("category", target_category, False)])

		video_metadata = self._make_video_metadata(
			grid_size,
			grids,
			queries=queries,
			remaining_query=remaining_query,
			target_object_moving_direction=target_object_moving_direction,
			are_other_objects_moving=task_plan['are other objects moving'],
			object_size_multiplier=DEFAULT_OBJECT_SIZE_MULTIPLIER
		)

		answer = self.metadata.get_surfacename(target_category)
		negatives = [self.metadata.get_surfacename(self.metadata.sample_category_for_object(self.rng, o, target_category))
					 for o in video_metadata['objects'][1:]]
		options = self._compose_options(answer, negatives)

		return question, answer, options, video_metadata


class WhatAttributeMovementVideoGridTaskGenerator(MovementVideoGridTaskGenerator):
	schema = {
		'task type'           : 'str',
		'grid number'             : 'int',
		'target category'         : 'str',
		'absolute position'       : 'str',
		'attribute type'          : 'str',
		'attribute value'         : 'str',
		'moving direction'        : 'str',
		'are other objects moving': 'str'
	}

	def enumerate_task_plans(self, task_store: TaskStore):
		for target_category in tqdm(self.metadata.categories, desc="enumerating [what attribute move video] task"):
			attribute_dict = self.metadata.get_category_attribute_dict(target_category)
			for attribute_type, attribute_values in attribute_dict.items():
				for attribute_value in attribute_values:
					for grid_size in grid_options:
						for absolute_pos in self.grid_mappings[grid_size]:
							for target_object_moving_direction in moving_options:
								task_plan = {
									'task type'           : 'what attribute move video',
									'grid number'             : grid_size,
									'target category'         : target_category,
									'absolute position'       : absolute_pos,
									'attribute type'          : attribute_type,
									'attribute value'         : attribute_value,
									'moving direction'        : target_object_moving_direction,
									'are other objects moving': "Yes"
								}
								task_store.add(task_plan)

								task_plan = {
									'task type'           : 'what attribute move video',
									'grid number'             : grid_size,
									'target category'         : target_category,
									'absolute position'       : absolute_pos,
									'attribute type'          : attribute_type,
									'attribute value'         : attribute_value,
									'moving direction'        : target_object_moving_direction,
									'are other objects moving': "No"
								}
								task_store.add(task_plan)

	def _generate_task(self, task_plan):
		grid_size = task_plan['grid number']

		attribute_type = task_plan['attribute type']
		absolute_pos = task_plan['absolute position']
		target_object_moving_direction = task_plan['moving direction']
		grids = [self.grid_mappings[grid_size][absolute_pos]]

		queries = [self._get_target_object_query(task_plan)]
		if task_plan['are other objects moving'] == "Yes":
			question = f"What is the {attribute_type} of the object that is moving {target_object_moving_direction} in the video?"
		else:
			question = f"What is the {attribute_type} of the moving object in the video?"

		video_metadata = self._make_video_metadata(
			grid_size,
			grids,
			queries=queries,
			target_object_moving_direction=target_object_moving_direction,
			are_other_objects_moving=task_plan['are other objects moving'],
			object_size_multiplier=DEFAULT_OBJECT_SIZE_MULTIPLIER
		)

		answer = task_plan['attribute value']
		target_object = video_metadata['objects'][0]
		negative_query = self.metadata.and_query([
			(attribute_type, a, False) for a in self.metadata.query_metadata(attribute_type, self.metadata.and_query([("object", target_object, True)]))
		])
		negatives = self.metadata.sample(
			self.rng,
			NUM_OPTIONS - 1,
			attribute_type,
			query=negative_query,
		)
		options = [answer] + negatives
		return question, answer, options, video_metadata


class WhereMovementVideoGridTaskGenerator(MovementVideoGridTaskGenerator):
	schema = {
		'task type'           : 'str',
		'grid number'             : 'int',
		'target category'         : 'str',
		'absolute position'       : 'str',
		'attribute type'          : 'str',
		'attribute value'         : 'str',
		'moving direction'        : 'str',
		'are other objects moving': 'str'
	}

	def __init__(self, metadata: MetaData, seed=42):
		super().__init__(metadata, seed=seed)
		self.relative_positions = relative_positions

	def enumerate_task_plans(self, task_store: TaskStore):
		for target_category in tqdm(self.metadata.categories, desc="enumerating [where move video] task"):
			attribute_dict = self.metadata.get_category_attribute_dict(target_category)
			for attribute_type, attribute_values in attribute_dict.items():
				for attribute_value in attribute_values:
					for grid_size in grid_options:
						for absolute_pos in self.grid_mappings[grid_size]:
							for target_object_moving_direction in moving_options:
								task_plan = {
									'task type'           : 'where move video',
									'grid number'             : grid_size,
									'target category'         : target_category,
									'absolute position'       : absolute_pos,
									'attribute type'          : attribute_type,
									'attribute value'         : attribute_value,
									'moving direction'        : target_object_moving_direction,
									'are other objects moving': "Yes"
								}
								task_store.add(task_plan)

								task_plan = {
									'task type'           : 'where move video',
									'grid number'             : grid_size,
									'target category'         : target_category,
									'absolute position'       : absolute_pos,
									'attribute type'          : attribute_type,
									'attribute value'         : attribute_value,
									'moving direction'        : target_object_moving_direction,
									'are other objects moving': "No"
								}
								task_store.add(task_plan)

			for grid_size in grid_options:
				for absolute_pos in self.grid_mappings[grid_size]:
					for target_object_moving_direction in moving_options:
						task_plan = {
							'task type'           : 'where move video',
							'grid number'             : grid_size,
							'target category'         : target_category,
							'absolute position'       : absolute_pos,
							'moving direction'        : target_object_moving_direction,
							'are other objects moving': "Yes"
						}
						task_store.add(task_plan)

						task_plan = {
							'task type'           : 'where move video',
							'grid number'             : grid_size,
							'target category'         : target_category,
							'absolute position'       : absolute_pos,
							'moving direction'        : target_object_moving_direction,
							'are other objects moving': "No"
						}
						task_store.add(task_plan)

	def _generate_task(self, task_plan):
		grid_size = task_plan['grid number']

		target_category = task_plan['target category']
		categories = [target_category]
		queries = [self._get_target_object_query(task_plan)]
		absolute_pos = task_plan['absolute position']
		grids = [self.grid_mappings[grid_size][absolute_pos]]
		target_object_moving_direction = task_plan['moving direction']

		if task_plan['are other objects moving'] == "Yes":
			question = f"Where is the object that is moving {target_object_moving_direction} located in the video?"
		else:
			question = f"Where is the moving object located in the video?"
		answer = absolute_pos
		negatives = [o for o in self.grid_mappings[grid_size].keys() if o != answer]

		options = self._compose_options(answer, negatives)
		video_metadata = self._make_video_metadata(
			grid_size,
			grids,
			queries=queries,
			remaining_query=self.metadata.and_query([("category", c, False) for c in categories]),
			target_object_moving_direction=target_object_moving_direction,
			are_other_objects_moving=task_plan['are other objects moving'],
			object_size_multiplier=DEFAULT_OBJECT_SIZE_MULTIPLIER
		)

		return question, answer, options, video_metadata
