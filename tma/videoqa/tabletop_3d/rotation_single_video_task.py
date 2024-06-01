from tqdm import tqdm

from .single_video_task import GridVideoTaskGenerator
from .utils import relative_positions, reverse_relative_positions
from ..metadata import ObjaverseVideoMetaData as MetaData
from ...constant import NUM_OPTIONS, VIDEO_FPS, VIDEO_NUM_FRAMES
from ...task_store import TaskStore

grid_options = [2]
DEFAULT_ROTATION_DEGREE = 720  # rotation clockwise for 720
DEFAULT_OBJECT_SIZE_MULTIPLIER = 1.3

rotation_options = {'clockwise', 'counterclockwise'}


def direction_to_keyframes(direction, rotation_degree):
	if direction == 'clockwise':
		return [{}, {}, {}, {}, {'rotation': rotation_degree}]
	elif direction == 'counterclockwise':
		return [{}, {}, {}, {}, {'rotation': -rotation_degree}]


class RotationVideoGridTaskGenerator(GridVideoTaskGenerator):
	def _make_video_metadata(self, grid_size, grids, queries, remaining_query=..., rotation_degree=DEFAULT_ROTATION_DEGREE, target_object_rotation_direction='clockwise', are_other_objects_rotating='Yes',
							 object_size_multiplier=DEFAULT_OBJECT_SIZE_MULTIPLIER):
		objects = [self.metadata.sample(self.rng, 1, "object", q) for q in queries]
		remaining_grids = [g for g in range(grid_size ** 2) if g not in grids]
		for _ in remaining_grids:
			uid = self.metadata.sample(self.rng, 1, "object", remaining_query)
			objects.append(uid)

		keyframes = [direction_to_keyframes(target_object_rotation_direction, rotation_degree)]
		if len(grids) > 1:
			# if thers is reference object, also add keyframe for reference objects
			if are_other_objects_rotating == "Yes":
				keyframes += [direction_to_keyframes('clockwise' if target_object_rotation_direction == 'counterclockwise' else 'counterclockwise', DEFAULT_ROTATION_DEGREE)]
			else:
				keyframes += [[{} for _ in range(5)]]

		if are_other_objects_rotating == "Yes":
			remaining_keyframes = [direction_to_keyframes('clockwise' if target_object_rotation_direction == 'counterclockwise' else 'counterclockwise', DEFAULT_ROTATION_DEGREE) for _ in range(len(remaining_grids))]
		else:
			remaining_keyframes = [[{} for _ in range(5)] for _ in range(len(remaining_grids))]
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


class WhatRotationVideoGridTaskGenerator(RotationVideoGridTaskGenerator):
	schema = {
		'task type'             : 'str',
		'grid number'               : 'int',
		'target category'           : 'str',
		'absolute position'         : 'str',
		'attribute type'            : 'str',
		'attribute value'           : 'str',
		'rotating direction'        : 'str',
		'are other objects rotating': 'str'
	}

	def enumerate_task_plans(self, task_store: TaskStore):
		for target_category in tqdm(self.metadata.categories, desc="enumerating [what rotate video] task"):
			attribute_dict = self.metadata.get_category_attribute_dict(target_category)
			for attribute_type, attribute_values in attribute_dict.items():
				for attribute_value in attribute_values:
					for grid_size in grid_options:
						for absolute_pos in self.grid_mappings[grid_size]:
							for target_object_rotation_direction in rotation_options:
								task_plan = {
									'task type'             : 'what rotate video',
									'grid number'               : grid_size,
									'target category'           : target_category,
									'absolute position'         : absolute_pos,
									'attribute type'            : attribute_type,
									'attribute value'           : attribute_value,
									'rotating direction'        : target_object_rotation_direction,
									'are other objects rotating': 'Yes'
								}
								task_store.add(task_plan)

								task_plan = {
									'task type'             : 'what rotate video',
									'grid number'               : grid_size,
									'target category'           : target_category,
									'absolute position'         : absolute_pos,
									'attribute type'            : attribute_type,
									'attribute value'           : attribute_value,
									'rotating direction'        : target_object_rotation_direction,
									'are other objects rotating': 'No'
								}
								task_store.add(task_plan)

			for grid_size in grid_options:
				for absolute_pos in self.grid_mappings[grid_size]:
					for target_object_rotation_direction in rotation_options:
						task_plan = {
							'task type'             : 'what rotate video',
							'grid number'               : grid_size,
							'target category'           : target_category,
							'absolute position'         : absolute_pos,
							'rotating direction'        : target_object_rotation_direction,
							'are other objects rotating': 'Yes'
						}
						task_store.add(task_plan)

						task_plan = {
							'task type'             : 'what rotate video',
							'grid number'               : grid_size,
							'target category'           : target_category,
							'absolute position'         : absolute_pos,
							'rotating direction'        : target_object_rotation_direction,
							'are other objects rotating': 'No'
						}
						task_store.add(task_plan)

	def _generate_task(self, task_plan):
		grid_size = task_plan['grid number']
		target_category = task_plan['target category']
		absolute_pos = task_plan['absolute position']
		grids = [self.grid_mappings[grid_size][absolute_pos]]

		if task_plan['are other objects rotating'] == "Yes":
			question = f"What is the object that is rotating {task_plan['rotating direction']} in the video?"
		else:
			question = f"What is the rotating object in the video?"

		queries = [self._get_target_object_query(task_plan)]

		remaining_query = self.metadata.and_query([("category", target_category, False)])

		video_metadata = self._make_video_metadata(
			grid_size,
			grids,
			queries=queries,
			remaining_query=remaining_query,
			rotation_degree=DEFAULT_ROTATION_DEGREE,
			target_object_rotation_direction=task_plan['rotating direction'],
			are_other_objects_rotating=task_plan['are other objects rotating'],
			object_size_multiplier=DEFAULT_OBJECT_SIZE_MULTIPLIER
		)

		answer = self.metadata.get_surfacename(target_category)
		negatives = [self.metadata.get_surfacename(self.metadata.sample_category_for_object(self.rng, o, target_category))
					 for o in video_metadata['objects'][1:]]
		options = self._compose_options(answer, negatives)

		return question, answer, options, video_metadata


class WhatAttributeRotationVideoGridTaskGenerator(RotationVideoGridTaskGenerator):
	schema = {
		'task type'             : 'str',
		'grid number'               : 'int',
		'target category'           : 'str',
		'absolute position'         : 'str',
		'attribute type'            : 'str',
		'attribute value'           : 'str',
		'rotating direction'        : 'str',
		'are other objects rotating': 'str'
	}

	def enumerate_task_plans(self, task_store: TaskStore):
		for target_category in tqdm(self.metadata.categories, desc="enumerating [what attribute rotate video] task"):
			attribute_dict = self.metadata.get_category_attribute_dict(target_category)
			for attribute_type, attribute_values in attribute_dict.items():
				for attribute_value in attribute_values:
					for grid_size in grid_options:
						for absolute_pos in self.grid_mappings[grid_size]:
							for target_object_rotation_direction in rotation_options:
								task_plan = {
									'task type'             : 'what attribute rotate video',
									'grid number'               : grid_size,
									'target category'           : target_category,
									'absolute position'         : absolute_pos,
									'attribute type'            : attribute_type,
									'attribute value'           : attribute_value,
									'rotating direction'        : target_object_rotation_direction,
									'are other objects rotating': 'Yes'
								}
								task_store.add(task_plan)

								task_plan = {
									'task type'             : 'what attribute rotate video',
									'grid number'               : grid_size,
									'target category'           : target_category,
									'absolute position'         : absolute_pos,
									'attribute type'            : attribute_type,
									'attribute value'           : attribute_value,
									'rotating direction'        : target_object_rotation_direction,
									'are other objects rotating': 'No'
								}
								task_store.add(task_plan)

	def _generate_task(self, task_plan):
		grid_size = task_plan['grid number']

		attribute_type = task_plan['attribute type']
		absolute_pos = task_plan['absolute position']
		grids = [self.grid_mappings[grid_size][absolute_pos]]

		queries = [self._get_target_object_query(task_plan)]
		if task_plan['are other objects rotating'] == "Yes":
			question = f"What is the {attribute_type} of the object that is rotating {task_plan['rotating direction']} in the video?"
		else:
			question = f"What is the {attribute_type} of the rotating object in the video?"

		video_metadata = self._make_video_metadata(
			grid_size,
			grids,
			queries=queries,
			rotation_degree=DEFAULT_ROTATION_DEGREE,
			target_object_rotation_direction=task_plan['rotating direction'],
			are_other_objects_rotating=task_plan['are other objects rotating'],
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


class WhereRotationVideoGridTaskGenerator(RotationVideoGridTaskGenerator):
	schema = {
		'task type'             : 'str',
		'grid number'               : 'int',
		'target category'           : 'str',
		'absolute position'         : 'str',
		'attribute type'            : 'str',
		'attribute value'           : 'str',
		'reference category'        : 'str',
		'reference position'        : 'str',
		'target-reference order'    : 'str',
		'rotating direction'        : 'str',
		'are other objects rotating': 'str'
	}

	def __init__(self, metadata: MetaData, seed=42):
		super().__init__(metadata, seed=seed)
		self.relative_positions = relative_positions

	def enumerate_task_plans(self, task_store: TaskStore):
		for target_category in tqdm(self.metadata.categories, desc="enumerating [where rotate video] task"):
			irrelevant_categories = self.metadata.get_irrelevant_categories(target_category)
			attribute_dict = self.metadata.get_category_attribute_dict(target_category)
			for attribute_type, attribute_values in attribute_dict.items():
				for attribute_value in attribute_values:
					for grid_size in grid_options:
						for absolute_pos in self.grid_mappings[grid_size]:
							grid = self.grid_mappings[grid_size][absolute_pos]
							for target_object_rotation_direction in rotation_options:
								task_plan = {
									'task type'             : 'where rotate video',
									'grid number'               : grid_size,
									'target category'           : target_category,
									'absolute position'         : absolute_pos,
									'attribute type'            : attribute_type,
									'attribute value'           : attribute_value,
									'rotating direction'        : target_object_rotation_direction,
									'are other objects rotating': 'Yes'
								}
								task_store.add(task_plan)

								task_plan = {
									'task type'             : 'where rotate video',
									'grid number'               : grid_size,
									'target category'           : target_category,
									'absolute position'         : absolute_pos,
									'attribute type'            : attribute_type,
									'attribute value'           : attribute_value,
									'rotating direction'        : target_object_rotation_direction,
									'are other objects rotating': 'No'
								}
								task_store.add(task_plan)

								for reference_category in irrelevant_categories:
									for reference_pos in self.relative_positions:
										ref_grid = self._relative_grid(grid_size, grid, reference_pos)
										if ref_grid >= 0:
											task_plan = {
												'task type'             : 'where rotate video',
												'grid number'               : grid_size,
												'target category'           : target_category,
												'absolute position'         : absolute_pos,
												'reference category'        : reference_category,
												'reference position'        : reference_pos,
												'attribute type'            : attribute_type,
												'attribute value'           : attribute_value,
												'target-reference order'    : 'target first',
												'rotating direction'        : target_object_rotation_direction,
												'are other objects rotating': 'Yes'
											}
											task_store.add(task_plan)

											task_plan = {
												'task type'             : 'where rotate video',
												'grid number'               : grid_size,
												'target category'           : target_category,
												'absolute position'         : absolute_pos,
												'reference category'        : reference_category,
												'reference position'        : reference_pos,
												'attribute type'            : attribute_type,
												'attribute value'           : attribute_value,
												'target-reference order'    : 'reference first',
												'rotating direction'        : target_object_rotation_direction,
												'are other objects rotating': 'Yes'
											}
											task_store.add(task_plan)

											task_plan = {
												'task type'             : 'where rotate video',
												'grid number'               : grid_size,
												'target category'           : target_category,
												'absolute position'         : absolute_pos,
												'reference category'        : reference_category,
												'reference position'        : reference_pos,
												'attribute type'            : attribute_type,
												'attribute value'           : attribute_value,
												'target-reference order'    : 'target first',
												'rotating direction'        : target_object_rotation_direction,
												'are other objects rotating': 'No'
											}
											task_store.add(task_plan)

											task_plan = {
												'task type'             : 'where rotate video',
												'grid number'               : grid_size,
												'target category'           : target_category,
												'absolute position'         : absolute_pos,
												'reference category'        : reference_category,
												'reference position'        : reference_pos,
												'attribute type'            : attribute_type,
												'attribute value'           : attribute_value,
												'target-reference order'    : 'reference first',
												'rotating direction'        : target_object_rotation_direction,
												'are other objects rotating': 'No'
											}
											task_store.add(task_plan)

			for grid_size in grid_options:
				for absolute_pos in self.grid_mappings[grid_size]:
					grid = self.grid_mappings[grid_size][absolute_pos]
					for target_object_rotation_direction in rotation_options:

						task_plan = {
							'task type'             : 'where rotate video',
							'grid number'               : grid_size,
							'target category'           : target_category,
							'absolute position'         : absolute_pos,
							'rotating direction'        : target_object_rotation_direction,
							'are other objects rotating': 'Yes'
						}
						task_store.add(task_plan)

						task_plan = {
							'task type'             : 'where rotate video',
							'grid number'               : grid_size,
							'target category'           : target_category,
							'absolute position'         : absolute_pos,
							'rotating direction'        : target_object_rotation_direction,
							'are other objects rotating': 'No'
						}
						task_store.add(task_plan)

						for reference_category in irrelevant_categories:
							for reference_pos in self.relative_positions:
								ref_grid = self._relative_grid(grid_size, grid, reference_pos)
								if ref_grid >= 0:
									task_plan = {
										'task type'             : 'where rotate video',
										'grid number'               : grid_size,
										'target category'           : target_category,
										'absolute position'         : absolute_pos,
										'reference category'        : reference_category,
										'reference position'        : reference_pos,
										'target-reference order'    : 'target first',
										'rotating direction'        : target_object_rotation_direction,
										'are other objects rotating': 'Yes'
									}
									task_store.add(task_plan)

									task_plan = {
										'task type'             : 'where rotate video',
										'grid number'               : grid_size,
										'target category'           : target_category,
										'absolute position'         : absolute_pos,
										'reference category'        : reference_category,
										'reference position'        : reference_pos,
										'target-reference order'    : 'reference first',
										'rotating direction'        : target_object_rotation_direction,
										'are other objects rotating': 'Yes'
									}
									task_store.add(task_plan)
									task_plan = {
										'task type'             : 'where rotate video',
										'grid number'               : grid_size,
										'target category'           : target_category,
										'absolute position'         : absolute_pos,
										'reference category'        : reference_category,
										'reference position'        : reference_pos,
										'target-reference order'    : 'target first',
										'rotating direction'        : target_object_rotation_direction,
										'are other objects rotating': 'Yes'
									}
									task_store.add(task_plan)

									task_plan = {
										'task type'             : 'where rotate video',
										'grid number'               : grid_size,
										'target category'           : target_category,
										'absolute position'         : absolute_pos,
										'reference category'        : reference_category,
										'reference position'        : reference_pos,
										'target-reference order'    : 'reference first',
										'rotating direction'        : target_object_rotation_direction,
										'are other objects rotating': 'No'
									}
									task_store.add(task_plan)

	def _generate_task(self, task_plan):
		grid_size = task_plan['grid number']

		target_category = task_plan['target category']
		categories = [target_category]
		queries = [self._get_target_object_query(task_plan)]
		absolute_pos = task_plan['absolute position']
		grids = [self.grid_mappings[grid_size][absolute_pos]]

		if 'reference category' in task_plan:
			reference_pos = task_plan['reference position']
			reference_category = task_plan['reference category']
			categories.append(reference_category)
			queries.append(self.metadata.and_query([("category", reference_category, True)]))

			ref_grid = self._relative_grid(grid_size, grids[0], reference_pos)
			assert ref_grid >= 0, "reference grid {} not allowed".format(ref_grid)
			grids.append(ref_grid)

			if task_plan['target-reference order'] == 'target first':
				if task_plan['are other objects rotating'] == "Yes":
					question = f"Where is the object that is rotating {task_plan['rotating direction']} with respect to the {self.metadata.get_surfacename(reference_category)} in the video?"
				else:
					question = f"Where is the rotating object with respect to the {self.metadata.get_surfacename(reference_category)} in the video?"
				answer = reference_pos
			else:
				if task_plan['are other objects rotating'] == "Yes":
					question = f"Where is the {self.metadata.get_surfacename(reference_category)} with respect to the object that is rotating {task_plan['rotating direction']} in the video?"
				else:
					question = f"Where is the {self.metadata.get_surfacename(reference_category)} with respect to the rotating object in the video?"

				answer = reverse_relative_positions[reference_pos]
			negatives = [o for o in self.relative_positions if o != answer]
		else:
			if task_plan['are other objects rotating'] == "Yes":
				question = f"Where is the object that is rotating {task_plan['rotating direction']} in the video?"
			else:
				question = f"Where is the rotating object in the video?"
			answer = absolute_pos
			negatives = [o for o in self.grid_mappings[grid_size].keys() if o != answer]

		options = self._compose_options(answer, negatives)
		video_metadata = self._make_video_metadata(
			grid_size,
			grids,
			queries=queries,
			remaining_query=self.metadata.and_query([("category", c, False) for c in categories]),
			rotation_degree=DEFAULT_ROTATION_DEGREE,
			target_object_rotation_direction=task_plan['rotating direction'],
			are_other_objects_rotating=task_plan['are other objects rotating'],
			object_size_multiplier=DEFAULT_OBJECT_SIZE_MULTIPLIER
		)

		return question, answer, options, video_metadata
