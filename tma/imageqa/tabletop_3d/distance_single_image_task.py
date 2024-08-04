from tqdm import tqdm

from .single_image_task import _3DGridTaskGenerator
from .utils import relative_positions
from ..metadata import Objaverse3DMetaData
from ...constant import NUM_OPTIONS
from ...task_store import TaskStore

grid_options = [3]

relative_distance = {
	'0': [[1, 3], [4], [2, 6], [5, 7], [8]],
	'1': [[0, 4, 2], [3, 5], [7], [6, 8]],
	'2': [[1, 5], [4], [0, 8], [3, 7], [6]],
	'3': [[0, 4, 6], [1, 7], [5], [2, 8]],
	'4': [[1, 3, 5, 7], [0, 2, 6, 8]],
	'5': [[2, 4, 8], [1, 7], [3], [0, 6]],
	'6': [[3, 7], [4], [0, 8], [1, 5], [2]],
	'7': [[4, 6, 8], [3, 5], [1], [2, 0]],
	'8': [[5, 7], [4], [2, 6], [1, 3], [0]],
}


def _get_relative_distance_level(ref, target):
	for idx, level in enumerate(relative_distance[str(ref)]):
		if target in level:
			return idx


def _get_max_distance_level(ref):
	return len(relative_distance[str(ref)]) - 1


def _get_farther_grids(ref, target):
	ref_level = _get_relative_distance_level(ref, target)
	farther_grids = []
	for level in relative_distance[str(ref)][ref_level + 1:]:
		farther_grids.extend(level)
	return farther_grids


def _get_closer_grids(ref, target):
	ref_level = _get_relative_distance_level(ref, target)
	closer_grids = []
	for level in relative_distance[str(ref)][:ref_level]:
		closer_grids.extend(level)
	return closer_grids


class Distance3DGridTaskGenerator(_3DGridTaskGenerator):

	def __init__(self, metadata: Objaverse3DMetaData, max_num_distracting_object=2, seed=42):
		super().__init__(metadata, seed=seed)
		self.grid_options = grid_options
		self.max_num_distracting_object = max_num_distracting_object

	def _make_image_metadata(self, grid_size, distance_type, grids, queries, remaining_query=...):
		target_grid = grids[0]
		ref_grid = grids[1]
		objects = [self.metadata.sample(self.rng, 1, "object", q) for q in queries]
		if distance_type == 'farthest':
			possible_closer_grids = _get_closer_grids(ref_grid, target_grid)
			remaining_grids = self.rng.choice(possible_closer_grids, replace=False, size=min(self.max_num_distracting_object, len(possible_closer_grids)))
		else:
			possible_farther_grids = _get_farther_grids(ref_grid, target_grid)
			remaining_grids = self.rng.choice(possible_farther_grids, replace=False, size=min(self.max_num_distracting_object, len(possible_farther_grids)))

		remaining_grids = [int(grid) for grid in remaining_grids]  # convert numpy.int64 to int to feed into json

		for _ in remaining_grids:
			uid = self.metadata.sample(self.rng, 1, "object", remaining_query)
			objects.append(uid)

		object_path = {k: self.metadata.get_object_path(k) for k in objects}
		angles = [self.metadata.sample_object_angle(self.rng, obj) for obj in objects]

		image_metadata = {
			'grid number'   : grid_size,
			'objects'       : objects,
			'object_path'   : object_path,
			'object_angles' : angles,
			'grids'         : grids + remaining_grids,
			'blender_config': self.metadata.sample_blender_configuration(self.rng),
		}
		return image_metadata


class WhatDistance3DGridTaskGenerator(Distance3DGridTaskGenerator):
	schema = {
		'task type'     : 'str',
		'distance type'     : 'str',
		'grid number'       : 'int',
		'target category'   : 'str',
		'absolute position' : 'str',
		'attribute type'    : 'str',
		'attribute value'   : 'str',
		'reference category': 'str',
		'reference position': 'str',
	}

	def __init__(self, metadata: Objaverse3DMetaData, seed=42):
		super().__init__(metadata, seed=seed)
		self.relative_positions = relative_positions

	def enumerate_task_plans(self, task_store: TaskStore):
		for target_category in tqdm(self.metadata.categories, desc="enumerating [what distance] task"):
			irrelevant_categories = self.metadata.get_irrelevant_categories(target_category)
			attribute_dict = self.metadata.get_category_attribute_dict(target_category)
			for attribute_type, attribute_values in attribute_dict.items():
				for attribute_value in attribute_values:
					for grid_size in self.grid_options:
						for absolute_pos in self.grid_mappings[grid_size]:
							grid = self.grid_mappings[grid_size][absolute_pos]
							for reference_category in irrelevant_categories:
								for reference_pos in self.relative_positions:
									ref_grid = self._relative_grid(grid_size, grid, reference_pos)
									if ref_grid >= 0:
										if (_get_relative_distance_level(ref_grid, grid) > 0):
											task_plan = {
												'task type'     : 'what distance',
												'distance type'     : 'farthest',
												'grid number'       : grid_size,
												'target category'   : target_category,
												'absolute position' : absolute_pos,
												'reference category': reference_category,
												'reference position': reference_pos,
												'attribute type'    : attribute_type,
												'attribute value'   : attribute_value,
											}
											task_store.add(task_plan)
										if (_get_relative_distance_level(ref_grid, grid) < _get_max_distance_level(ref_grid)):
											task_plan = {
												'task type'     : 'what distance',
												'distance type'     : 'closest',
												'grid number'       : grid_size,
												'target category'   : target_category,
												'absolute position' : absolute_pos,
												'reference category': reference_category,
												'reference position': reference_pos,
												'attribute type'    : attribute_type,
												'attribute value'   : attribute_value,
											}
											task_store.add(task_plan)

			for grid_size in self.grid_options:
				for absolute_pos in self.grid_mappings[grid_size]:
					grid = self.grid_mappings[grid_size][absolute_pos]
					for reference_category in irrelevant_categories:
						for reference_pos in self.relative_positions:
							ref_grid = self._relative_grid(grid_size, grid, reference_pos)
							if ref_grid >= 0:
								if (_get_relative_distance_level(ref_grid, grid) > 0):
									task_plan = {
										'task type'     : 'what distance',
										'distance type'     : 'farthest',
										'grid number'       : grid_size,
										'target category'   : target_category,
										'absolute position' : absolute_pos,
										'reference category': reference_category,
										'reference position': reference_pos,
									}
									task_store.add(task_plan)
								if (_get_relative_distance_level(ref_grid, grid) < _get_max_distance_level(ref_grid)):
									task_plan = {
										'task type'     : 'what distance',
										'distance type'     : 'closest',
										'grid number'       : grid_size,
										'target category'   : target_category,
										'absolute position' : absolute_pos,
										'reference category': reference_category,
										'reference position': reference_pos,
									}
									task_store.add(task_plan)

	def _generate_task(self, task_plan):
		grid_size = task_plan['grid number']

		target_category = task_plan['target category']
		absolute_pos = task_plan['absolute position']
		grids = [self.grid_mappings[grid_size][absolute_pos]]
		queries = [self._get_target_object_query(task_plan)]

		remaining_query = [("category", target_category, False)]

		reference_pos = task_plan['reference position']
		reference_category = task_plan['reference category']

		queries.append(self.metadata.and_query([("category", reference_category, True)]))
		remaining_query += [("category", reference_category, False)]

		ref_grid = self._relative_grid(grid_size, grids[0], reference_pos)
		assert ref_grid >= 0, "reference grid {} not allowed".format(ref_grid)
		grids.append(ref_grid)

		if task_plan['distance type'] == 'farthest':
			question = f"What is the object that is farthest from the {self.metadata.get_surfacename(reference_category)}?"
		else:
			question = f"What is the object that is closest to the {self.metadata.get_surfacename(reference_category)}?"

		image_metadata = self._make_image_metadata(
			grid_size,
			distance_type=task_plan['distance type'],
			grids=grids,
			queries=queries,
			remaining_query=self.metadata.and_query(remaining_query)
		)

		answer = self.metadata.get_surfacename(target_category)
		negatives = [self.metadata.get_surfacename(self.metadata.sample_category_for_object(self.rng, o, target_category))
					 for o in image_metadata['objects'][1:]]
		options = self._compose_options(answer, negatives)

		return question, answer, options, image_metadata


class WhatAttributeDistance3DGridTaskGenerator(Distance3DGridTaskGenerator):
	schema = {
		'task type'     : 'str',
		'distance type'     : 'str',
		'grid number'       : 'int',
		'target category'   : 'str',
		'absolute position' : 'str',
		'attribute type'    : 'str',
		'attribute value'   : 'str',
		'reference category': 'str',
		'reference position': 'str',
	}

	def __init__(self, metadata: Objaverse3DMetaData, seed=42):
		super().__init__(metadata, seed=seed)
		self.relative_positions = relative_positions

	def enumerate_task_plans(self, task_store: TaskStore):
		for target_category in tqdm(self.metadata.categories, desc="enumerating [what attribute distance] task"):
			irrelevant_categories = self.metadata.get_irrelevant_categories(target_category)
			attribute_dict = self.metadata.get_category_attribute_dict(target_category)
			for attribute_type, attribute_values in attribute_dict.items():
				for attribute_value in attribute_values:
					for grid_size in self.grid_options:
						for absolute_pos in self.grid_mappings[grid_size]:
							grid = self.grid_mappings[grid_size][absolute_pos]
							for reference_category in irrelevant_categories:
								for reference_pos in self.relative_positions:
									ref_grid = self._relative_grid(grid_size, grid, reference_pos)
									if ref_grid >= 0:
										if (_get_relative_distance_level(ref_grid, grid) > 0):
											task_plan = {
												'task type'     : 'what attribute distance',
												'distance type'     : 'farthest',
												'grid number'       : grid_size,
												'target category'   : target_category,
												'absolute position' : absolute_pos,
												'reference category': reference_category,
												'reference position': reference_pos,
												'attribute type'    : attribute_type,
												'attribute value'   : attribute_value,
											}
											task_store.add(task_plan)
										if (_get_relative_distance_level(ref_grid, grid) < _get_max_distance_level(ref_grid)):
											task_plan = {
												'task type'     : 'what attribute distance',
												'distance type'     : 'closest',
												'grid number'       : grid_size,
												'target category'   : target_category,
												'absolute position' : absolute_pos,
												'reference category': reference_category,
												'reference position': reference_pos,
												'attribute type'    : attribute_type,
												'attribute value'   : attribute_value,
											}
											task_store.add(task_plan)

	def _generate_task(self, task_plan):
		grid_size = task_plan['grid number']

		attribute_type = task_plan['attribute type']

		absolute_pos = task_plan['absolute position']
		grids = [self.grid_mappings[grid_size][absolute_pos]]

		queries = [self._get_target_object_query(task_plan)]

		reference_pos = task_plan['reference position']
		reference_category = task_plan['reference category']
		queries.append(self.metadata.and_query([("category", reference_category, True)]))

		ref_grid = self._relative_grid(grid_size, grids[0], reference_pos)
		assert ref_grid >= 0, "reference grid {} not allowed".format(ref_grid)

		grids.append(ref_grid)
		if task_plan['distance type'] == 'farthest':
			question = f"What is the {attribute_type} of the object that is farthest from the {self.metadata.get_surfacename(reference_category)}?"
		else:
			question = f"What is the {attribute_type} of the object that is closest to the {self.metadata.get_surfacename(reference_category)}?"

		image_metadata = self._make_image_metadata(
			grid_size,
			distance_type=task_plan['distance type'],
			grids=grids,
			queries=queries,
		)

		answer = task_plan['attribute value']
		target_object = image_metadata['objects'][0]
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

		return question, answer, options, image_metadata


class WhereDistance3DGridTaskGenerator(Distance3DGridTaskGenerator):
	schema = {
		'task type'     : 'str',
		'distance type'     : 'str',
		'grid number'       : 'int',
		'target category'   : 'str',
		'absolute position' : 'str',
		'attribute type'    : 'str',
		'attribute value'   : 'str',
		'reference category': 'str',
		'reference position': 'str',
	}

	def __init__(self, metadata: Objaverse3DMetaData, seed=42):
		super().__init__(metadata, seed=seed)
		self.relative_positions = relative_positions

	def enumerate_task_plans(self, task_store: TaskStore):
		for target_category in tqdm(self.metadata.categories, desc="enumerating [where distance] task"):
			irrelevant_categories = self.metadata.get_irrelevant_categories(target_category)
			attribute_dict = self.metadata.get_category_attribute_dict(target_category)
			for attribute_type, attribute_values in attribute_dict.items():
				for attribute_value in attribute_values:
					for grid_size in self.grid_options:
						for absolute_pos in self.grid_mappings[grid_size]:
							grid = self.grid_mappings[grid_size][absolute_pos]
							for reference_category in irrelevant_categories:
								for reference_pos in self.relative_positions:
									ref_grid = self._relative_grid(grid_size, grid, reference_pos)
									if ref_grid >= 0:
										if (_get_relative_distance_level(ref_grid, grid) > 0):
											task_plan = {
												'task type'     : 'where distance',
												'distance type'     : 'farthest',
												'grid number'       : grid_size,
												'target category'   : target_category,
												'absolute position' : absolute_pos,
												'reference category': reference_category,
												'reference position': reference_pos,
												'attribute type'    : attribute_type,
												'attribute value'   : attribute_value,
											}
											task_store.add(task_plan)
										if (_get_relative_distance_level(ref_grid, grid) < _get_max_distance_level(ref_grid)):
											task_plan = {
												'task type'     : 'where distance',
												'distance type'     : 'closest',
												'grid number'       : grid_size,
												'target category'   : target_category,
												'absolute position' : absolute_pos,
												'reference category': reference_category,
												'reference position': reference_pos,
												'attribute type'    : attribute_type,
												'attribute value'   : attribute_value,
											}
											task_store.add(task_plan)

			for grid_size in self.grid_options:
				for absolute_pos in self.grid_mappings[grid_size]:
					grid = self.grid_mappings[grid_size][absolute_pos]
					for reference_category in irrelevant_categories:
						for reference_pos in self.relative_positions:
							ref_grid = self._relative_grid(grid_size, grid, reference_pos)
							if ref_grid >= 0:
								if (_get_relative_distance_level(ref_grid, grid) > 0):
									task_plan = {
										'task type'     : 'where distance',
										'distance type'     : 'farthest',
										'grid number'       : grid_size,
										'target category'   : target_category,
										'absolute position' : absolute_pos,
										'reference category': reference_category,
										'reference position': reference_pos,
									}
									task_store.add(task_plan)
								if (_get_relative_distance_level(ref_grid, grid) < _get_max_distance_level(ref_grid)):
									task_plan = {
										'task type'     : 'where distance',
										'distance type'     : 'closest',
										'grid number'       : grid_size,
										'target category'   : target_category,
										'absolute position' : absolute_pos,
										'reference category': reference_category,
										'reference position': reference_pos,
									}
									task_store.add(task_plan)

	def _generate_task(self, task_plan):
		grid_size = task_plan['grid number']

		target_category = task_plan['target category']
		categories = [target_category]
		queries = [self._get_target_object_query(task_plan)]
		absolute_pos = task_plan['absolute position']
		grids = [self.grid_mappings[grid_size][absolute_pos]]

		reference_pos = task_plan['reference position']
		reference_category = task_plan['reference category']
		categories.append(reference_category)
		queries.append(self.metadata.and_query([("category", reference_category, True)]))

		ref_grid = self._relative_grid(grid_size, grids[0], reference_pos)
		assert ref_grid >= 0, "reference grid {} not allowed".format(ref_grid)
		grids.append(ref_grid)

		if task_plan['distance type'] == 'farthest':
			question = f"Where is the object that is farthest from the {self.metadata.get_surfacename(reference_category)} in the image?"
		else:
			question = f"Where is the object that is closest to the {self.metadata.get_surfacename(reference_category)} in the image?"
		answer = absolute_pos
		negatives = [o for o in self.grid_mappings[grid_size].keys() if o != answer]

		options = self._compose_options(answer, negatives)
		image_metadata = self._make_image_metadata(
			grid_size,
			distance_type=task_plan['distance type'],
			grids=grids,
			queries=queries,
			remaining_query=self.metadata.and_query([("category", c, False) for c in categories])
		)

		return question, answer, options, image_metadata
