from tqdm import tqdm

from .single_image_task import _3DGridTaskGenerator
from .utils import relative_positions, reverse_relative_positions
from ..metadata import Objaverse3DMetaData
from ...constant import NUM_OPTIONS
from ...task_store import TaskStore

largest_size = 1.5
smallest_size = 0.5
all_size_options = set([0.5, 1.0, 1.5])
grid_options = [2]


class Size3DGridTaskGenerator(_3DGridTaskGenerator):
	def __init__(self, metadata: Objaverse3DMetaData, seed=42):
		super().__init__(metadata, seed=seed)
		self.grid_options = grid_options

	def _make_image_metadata(self, grid_size, sizes, size_options, grids, queries, remaining_query=...):
		objects = [self.metadata.sample(self.rng, 1, "object", q) for q in queries]

		remaining_grids = [g for g in range(grid_size ** 2) if g not in grids]
		for _ in remaining_grids:
			uid = self.metadata.sample(self.rng, 1, "object", remaining_query)
			objects.append(uid)
		remaining_sizes = list(self.rng.choice(size_options, replace=True, size=len(remaining_grids)))

		object_path = {k: self.metadata.get_object_path(k) for k in objects}
		angles = [self.metadata.sample_object_angle(self.rng, obj) for obj in objects]

		image_metadata = {
			'grid number'   : grid_size,
			'objects'       : objects,
			'object_path'   : object_path,
			'object_angles' : angles,
			'grids'         : grids + remaining_grids,
			'blender_config': self.metadata.sample_blender_configuration(self.rng),
			'sizes'         : sizes + remaining_sizes
		}
		return image_metadata


class WhatSize3DGridTaskGenerator(Size3DGridTaskGenerator):
	schema = {
		'task type'    : 'str',
		'size'             : 'str',
		'grid number'      : 'int',
		'target category'  : 'str',
		'absolute position': 'str',
		'attribute type'   : 'str',
		'attribute value'  : 'str',
	}

	def enumerate_task_plans(self, task_store: TaskStore):
		for target_category in tqdm(self.metadata.categories, desc="enumerating [what size] task"):
			attribute_dict = self.metadata.get_category_attribute_dict(target_category)
			for attribute_type, attribute_values in attribute_dict.items():
				for attribute_value in attribute_values:
					for grid_size in self.grid_options:
						for absolute_pos in self.grid_mappings[grid_size]:
							task_plan = {
								'task type'    : 'what size',
								'size'             : 'largest',
								'grid number'      : grid_size,
								'target category'  : target_category,
								'absolute position': absolute_pos,
								'attribute type'   : attribute_type,
								'attribute value'  : attribute_value,
							}
							task_store.add(task_plan)

							task_plan = {
								'task type'    : 'what size',
								'size'             : 'smallest',
								'grid number'      : grid_size,
								'target category'  : target_category,
								'absolute position': absolute_pos,
								'attribute type'   : attribute_type,
								'attribute value'  : attribute_value,
							}
							task_store.add(task_plan)

			for grid_size in self.grid_options:
				for absolute_pos in self.grid_mappings[grid_size]:
					task_plan = {
						'task type'    : 'what size',
						'size'             : 'largest',
						'grid number'      : grid_size,
						'target category'  : target_category,
						'absolute position': absolute_pos,
					}
					task_store.add(task_plan)

					task_plan = {
						'task type'    : 'what size',
						'size'             : 'smallest',
						'grid number'      : grid_size,
						'target category'  : target_category,
						'absolute position': absolute_pos,
					}
					task_store.add(task_plan)

	def _generate_task(self, task_plan):
		grid_size = task_plan['grid number']
		target_category = task_plan['target category']
		absolute_pos = task_plan['absolute position']
		grids = [self.grid_mappings[grid_size][absolute_pos]]

		if task_plan['size'] == 'largest':
			sizes = [largest_size]
			size_options = list(all_size_options - {largest_size})
			question = f"What is the largest object in the image?"
		else:
			sizes = [smallest_size]
			size_options = list(all_size_options - {smallest_size})
			question = f"What is the smallest object in the image?"

		queries = [self._get_target_object_query(task_plan)]

		remaining_query = self.metadata.and_query([("category", target_category, False)])

		image_metadata = self._make_image_metadata(
			grid_size,
			sizes,
			size_options,
			grids,
			queries=queries,
			remaining_query=remaining_query,
		)

		answer = self.metadata.get_surfacename(target_category)
		negatives = [self.metadata.get_surfacename(self.metadata.sample_category_for_object(self.rng, o, target_category))
					 for o in image_metadata['objects'][1:]]
		options = self._compose_options(answer, negatives)

		return question, answer, options, image_metadata


class WhatAttributeSize3DGridTaskGenerator(Size3DGridTaskGenerator):
	schema = {
		'task type'    : 'str',
		'size'             : 'str',
		'grid number'      : 'int',
		'target category'  : 'str',
		'absolute position': 'str',
		'attribute type'   : 'str',
		'attribute value'  : 'str',
	}

	def enumerate_task_plans(self, task_store: TaskStore):
		for target_category in tqdm(self.metadata.categories, desc="enumerating [what size attribute] task"):
			attribute_dict = self.metadata.get_category_attribute_dict(target_category)
			for attribute_type, attribute_values in attribute_dict.items():
				for attribute_value in attribute_values:
					for grid_size in self.grid_options:
						for absolute_pos in self.grid_mappings[grid_size]:
							task_plan = {
								'task type'    : 'what attribute size',
								'size'             : 'largest',
								'grid number'      : grid_size,
								'target category'  : target_category,
								'absolute position': absolute_pos,
								'attribute type'   : attribute_type,
								'attribute value'  : attribute_value,
							}
							task_store.add(task_plan)

							task_plan = {
								'task type'    : 'what attribute size',
								'size'             : 'smallest',
								'grid number'      : grid_size,
								'target category'  : target_category,
								'absolute position': absolute_pos,
								'attribute type'   : attribute_type,
								'attribute value'  : attribute_value,
							}
							task_store.add(task_plan)

	def _generate_task(self, task_plan):
		grid_size = task_plan['grid number']

		attribute_type = task_plan['attribute type']

		absolute_pos = task_plan['absolute position']
		grids = [self.grid_mappings[grid_size][absolute_pos]]

		queries = [self._get_target_object_query(task_plan)]

		if task_plan['size'] == 'largest':
			sizes = [largest_size]
			size_options = list(all_size_options - {largest_size})
			question = f"What is the {attribute_type} of the largest object in the image?"
		else:
			sizes = [smallest_size]
			size_options = list(all_size_options - {smallest_size})
			question = f"What is the {attribute_type} of the smallest object in the image?"

		image_metadata = self._make_image_metadata(
			grid_size,
			sizes,
			size_options,
			grids,
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


class WhereSize3DGridTaskGenerator(Size3DGridTaskGenerator):
	schema = {
		'task type'         : 'str',
		'size'                  : 'str',
		'grid number'           : 'int',
		'target category'       : 'str',
		'absolute position'     : 'str',
		'attribute type'        : 'str',
		'attribute value'       : 'str',
		'reference category'    : 'str',
		'reference position'    : 'str',
		'target-reference order': 'str'
	}

	def __init__(self, metadata: Objaverse3DMetaData, seed=42):
		super().__init__(metadata, seed=seed)
		self.relative_positions = relative_positions

	def enumerate_task_plans(self, task_store: TaskStore):
		for target_category in tqdm(self.metadata.categories, desc="enumerating [where size] task"):
			irrelevant_categories = self.metadata.get_irrelevant_categories(target_category)
			attribute_dict = self.metadata.get_category_attribute_dict(target_category)
			for attribute_type, attribute_values in attribute_dict.items():
				for attribute_value in attribute_values:
					for grid_size in self.grid_options:
						for absolute_pos in self.grid_mappings[grid_size]:
							task_plan = {
								'task type'    : 'where size',
								'size'             : 'largest',
								'grid number'      : grid_size,
								'target category'  : target_category,
								'absolute position': absolute_pos,
								'attribute type'   : attribute_type,
								'attribute value'  : attribute_value,
							}
							task_store.add(task_plan)

							task_plan = {
								'task type'    : 'where size',
								'size'             : 'smallest',
								'grid number'      : grid_size,
								'target category'  : target_category,
								'absolute position': absolute_pos,
								'attribute type'   : attribute_type,
								'attribute value'  : attribute_value,
							}
							task_store.add(task_plan)

							grid = self.grid_mappings[grid_size][absolute_pos]

							for reference_category in irrelevant_categories:
								for reference_pos in self.relative_positions:
									ref_grid = self._relative_grid(grid_size, grid, reference_pos)
									if ref_grid >= 0:
										task_plan = {
											'task type'         : 'where size',
											'size'                  : 'largest',
											'grid number'           : grid_size,
											'target category'       : target_category,
											'absolute position'     : absolute_pos,
											'reference category'    : reference_category,
											'reference position'    : reference_pos,
											'attribute type'        : attribute_type,
											'attribute value'       : attribute_value,
											'target-reference order': 'target first'
										}
										task_store.add(task_plan)

										task_plan = {
											'task type'         : 'where size',
											'size'                  : 'largest',
											'grid number'           : grid_size,
											'target category'       : target_category,
											'absolute position'     : absolute_pos,
											'reference category'    : reference_category,
											'reference position'    : reference_pos,
											'attribute type'        : attribute_type,
											'attribute value'       : attribute_value,
											'target-reference order': 'reference first'
										}
										task_store.add(task_plan)

										task_plan = {
											'task type'         : 'where size',
											'size'                  : 'smallest',
											'grid number'           : grid_size,
											'target category'       : target_category,
											'absolute position'     : absolute_pos,
											'reference category'    : reference_category,
											'reference position'    : reference_pos,
											'attribute type'        : attribute_type,
											'attribute value'       : attribute_value,
											'target-reference order': 'target first'
										}
										task_store.add(task_plan)

										task_plan = {
											'task type'         : 'where size',
											'size'                  : 'smallest',
											'grid number'           : grid_size,
											'target category'       : target_category,
											'absolute position'     : absolute_pos,
											'reference category'    : reference_category,
											'reference position'    : reference_pos,
											'attribute type'        : attribute_type,
											'attribute value'       : attribute_value,
											'target-reference order': 'reference first'
										}
										task_store.add(task_plan)

			for grid_size in self.grid_options:
				for absolute_pos in self.grid_mappings[grid_size]:
					task_plan = {
						'task type'    : 'where size',
						'size'             : 'largest',
						'grid number'      : grid_size,
						'target category'  : target_category,
						'absolute position': absolute_pos,
					}
					task_store.add(task_plan)

					task_plan = {
						'task type'    : 'where size',
						'size'             : 'smallest',
						'grid number'      : grid_size,
						'target category'  : target_category,
						'absolute position': absolute_pos,
					}
					task_store.add(task_plan)

					grid = self.grid_mappings[grid_size][absolute_pos]

					for reference_category in irrelevant_categories:
						for reference_pos in self.relative_positions:
							ref_grid = self._relative_grid(grid_size, grid, reference_pos)
							if ref_grid >= 0:
								task_plan = {
									'task type'         : 'where size',
									'size'                  : 'largest',
									'grid number'           : grid_size,
									'target category'       : target_category,
									'absolute position'     : absolute_pos,
									'reference category'    : reference_category,
									'reference position'    : reference_pos,
									'target-reference order': 'target first'
								}
								task_store.add(task_plan)

								task_plan = {
									'task type'         : 'where size',
									'size'                  : 'largest',
									'grid number'           : grid_size,
									'target category'       : target_category,
									'absolute position'     : absolute_pos,
									'reference category'    : reference_category,
									'reference position'    : reference_pos,
									'target-reference order': 'reference first'
								}
								task_store.add(task_plan)

								task_plan = {
									'task type'         : 'where size',
									'size'                  : 'smallest',
									'grid number'           : grid_size,
									'target category'       : target_category,
									'absolute position'     : absolute_pos,
									'reference category'    : reference_category,
									'reference position'    : reference_pos,
									'target-reference order': 'target first'
								}
								task_store.add(task_plan)

								task_plan = {
									'task type'         : 'where size',
									'size'                  : 'smallest',
									'grid number'           : grid_size,
									'target category'       : target_category,
									'absolute position'     : absolute_pos,
									'reference category'    : reference_category,
									'reference position'    : reference_pos,
									'target-reference order': 'reference first'
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
				if task_plan['size'] == 'largest':
					question = f"Where is the largest object in the image with respect to the {self.metadata.get_surfacename(reference_category)}?"
				else:
					question = f"Where is the smallest object in the image with respect to the {self.metadata.get_surfacename(reference_category)}?"
				answer = reference_pos
			else:
				if task_plan['size'] == 'largest':
					question = f"Where is the {self.metadata.get_surfacename(reference_category)} with respect to the largest object in the image?"
				else:
					question = f"Where is the {self.metadata.get_surfacename(reference_category)} with respect to the smallest object in the image?"
				answer = reverse_relative_positions[reference_pos]
			negatives = [o for o in self.relative_positions if o != answer]
		else:
			if task_plan['size'] == 'largest':
				question = f"Where is the largest object in the image?"
			else:
				question = f"Where is the smallest object in the image?"
			answer = absolute_pos
			negatives = [o for o in self.grid_mappings[grid_size].keys() if o != answer]

		if task_plan['size'] == 'largest':
			sizes = [largest_size]
			size_options = list(all_size_options - {largest_size})
		else:
			sizes = [smallest_size]
			size_options = list(all_size_options - {smallest_size})
		sizes += list(self.rng.choice(size_options, replace=True, size=1))

		options = self._compose_options(answer, negatives)
		image_metadata = self._make_image_metadata(
			grid_size,
			sizes,
			size_options,
			grids,
			queries=queries,
			remaining_query=self.metadata.and_query([("category", c, False) for c in categories])
		)

		return question, answer, options, image_metadata
