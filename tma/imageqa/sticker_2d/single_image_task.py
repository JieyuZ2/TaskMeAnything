from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from .utils import grid_mappings, grid_options, make_image, relative_grid, relative_position_phrase, relative_positions
from ..metadata import Objaverse2DMetaData, ObjaverseMetaData
from ...base import TaskGenerator
from ...constant import IMAGE_H, IMAGE_W, NUM_OPTIONS
from ...task_store import TaskStore


class GridTaskGenerator(TaskGenerator):
	metadata: Objaverse2DMetaData

	def __init__(self, metadata: ObjaverseMetaData, seed=42):
		super().__init__(metadata, seed=seed)
		self.grid_options = grid_options
		self.grid_mappings = grid_mappings

	def _relative_grid(self, grid_size, grid, reference_pos):
		return relative_grid(grid_size, grid, reference_pos)

	def _make_image_metadata(self, grid_size, grids, queries, remaining_query=...):
		objects, object_paths = [], []
		for q in queries:
			uid = self.metadata.sample(self.rng, 1, "object", q)
			view = self.metadata.sample_image(self.rng, uid)
			objects.append(uid)
			object_paths.append(view)

		remaining_grids = [g for g in range(grid_size ** 2) if g not in grids]
		for _ in remaining_grids:
			uid = self.metadata.sample(self.rng, 1, "object", remaining_query)
			view = self.metadata.sample_image(self.rng, uid)
			objects.append(uid)
			object_paths.append(view)

		image_metadata = {
			'grid number' : grid_size,
			'objects'     : objects,
			'object paths': object_paths,
			'grids'       : grids + remaining_grids,
		}

		return image_metadata

	def make_image(self, image_metadata):
		return make_image(image_metadata, IMAGE_H, IMAGE_W)

	def _get_target_object_query(self, task_plan):
		if 'attribute type' in task_plan:
			return self.metadata.and_query([("category", task_plan['target category'], True), (task_plan['attribute type'], task_plan['attribute value'], True)])
		else:
			return self.metadata.and_query([("category", task_plan['target category'], True)])

	def _task_plan_to_str(self, task_plan):
		t = []
		for k, v in task_plan.items():
			if self.metadata.check_category_exists(v):
				t.append(f'{k}: {self.metadata.get_surfacename(v)}')
			else:
				t.append(f'{k}: {v}')
		return '\n'.join(t)

	def _generate_task(self, task_plan) -> Tuple[str, str, List[str], Dict]:
		"(Abstract method) generate task"

	def generate(self, task_plan, return_data=True, seed=None):
		if seed is not None:
			self.rng = np.random.default_rng(seed=seed)

		question, answer, options, image_metadata = self._generate_task(task_plan)

		task = {
			'question'      : question.replace('_', ' '),
			'answer'        : answer.replace('_', ' '),
			'options'       : [o.replace('_', ' ') for o in options],
			'task_plan'     : self._task_plan_to_str(task_plan),
			'image_metadata': image_metadata,
			'image'         : self.make_image(image_metadata) if return_data else None
		}

		return task


class WhatGridTaskGenerator(GridTaskGenerator):
	schema = {
		'task type'     : 'str',
		'grid number'       : 'int',
		'target category'   : 'str',
		'absolute position' : 'str',
		'reference category': 'str',
		'reference position': 'str',
		'attribute type'    : 'str',
		'attribute value'   : 'str',
	}

	def __init__(self, metadata: ObjaverseMetaData, seed=42):
		super().__init__(metadata, seed=seed)
		self.relative_position_phrase = relative_position_phrase
		self.relative_positions = relative_positions

	def enumerate_task_plans(self, task_store: TaskStore):
		for target_category in tqdm(self.metadata.categories, desc="enumerating [what] task"):
			irrelevant_categories = self.metadata.get_irrelevant_categories(target_category)
			attribute_dict = self.metadata.get_category_attribute_dict(target_category)
			for attribute_type, attribute_values in attribute_dict.items():
				for attribute_value in attribute_values:
					for grid_size in self.grid_options:
						for absolute_pos in self.grid_mappings[grid_size]:
							task_plan = {
								'task type'    : 'what',
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
											'task type'     : 'what',
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
					task_plan = {
						'task type'    : 'what',
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
									'task type'     : 'what',
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

		if 'reference category' in task_plan:
			reference_pos = task_plan['reference position']
			reference_category = task_plan['reference category']

			queries.append(self.metadata.and_query([("category", reference_category, True)]))
			remaining_query += [("category", reference_category, False)]

			ref_grid = self._relative_grid(grid_size, grids[0], reference_pos)
			assert ref_grid >= 0, "reference grid {} not allowed".format(ref_grid)
			grids.append(ref_grid)

			question = f"What is the object {self.relative_position_phrase[reference_pos]} the {self.metadata.get_surfacename(reference_category)}?"
		else:
			question = f"What is the object in the {absolute_pos} part of the image?"

		image_metadata = self._make_image_metadata(
			grid_size,
			grids,
			queries=queries,
			remaining_query=self.metadata.and_query(remaining_query)
		)

		answer = self.metadata.get_surfacename(target_category)
		negatives = [self.metadata.get_surfacename(self.metadata.sample_category_for_object(self.rng, o, target_category))
					 for o in image_metadata['objects'][1:]]
		options = self._compose_options(answer, negatives)

		return question, answer, options, image_metadata


class WhereGridTaskGenerator(GridTaskGenerator):
	schema = {
		'task type'     : 'str',
		'grid number'       : 'int',
		'target category'   : 'str',
		'absolute position' : 'str',
		'reference category': 'str',
		'reference position': 'str',
		'attribute type'    : 'str',
		'attribute value'   : 'str',
	}

	def __init__(self, metadata: ObjaverseMetaData, seed=42):
		super().__init__(metadata, seed=seed)
		self.relative_positions = relative_positions

	def enumerate_task_plans(self, task_store: TaskStore):
		for target_category in tqdm(self.metadata.categories, desc="enumerating [where] task"):
			irrelevant_categories = self.metadata.get_irrelevant_categories(target_category)
			attribute_dict = self.metadata.get_category_attribute_dict(target_category)
			for attribute_type, attribute_values in attribute_dict.items():
				for attribute_value in attribute_values:
					for grid_size in self.grid_options:
						for absolute_pos in self.grid_mappings[grid_size]:
							task_plan = {
								'task type'    : 'where',
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
											'task type'     : 'where',
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
					task_plan = {
						'task type'    : 'where',
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
									'task type'     : 'where',
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

		target_category_name = self.metadata.get_surfacename(target_category)
		queries = [self._get_target_object_query(task_plan)]
		remaining_query = [("category", target_category, False)]

		if 'reference category' in task_plan:
			reference_pos = task_plan['reference position']
			reference_category = task_plan['reference category']
			queries.append(self.metadata.and_query([("category", reference_category, True)]))
			remaining_query += [("category", reference_category, False)]

			ref_grid = self._relative_grid(grid_size, grids[0], reference_pos)
			assert ref_grid >= 0, "reference grid {} not allowed".format(ref_grid)
			grids.append(ref_grid)

			answer = reference_pos
			negatives = [o for o in self.relative_positions if o != answer]
			question = f"Where is the {target_category_name} with respect to the {self.metadata.get_surfacename(reference_category)}?"
		else:
			answer = absolute_pos
			negatives = [o for o in self.grid_mappings[grid_size].keys() if o != answer]
			question = f"Where is the {target_category_name} in the image?"

		options = self._compose_options(answer, negatives)
		image_metadata = self._make_image_metadata(
			grid_size,
			grids,
			queries=queries,
			remaining_query=self.metadata.and_query(remaining_query)
		)

		return question, answer, options, image_metadata


class WhatAttributeGridTaskGenerator(GridTaskGenerator):
	schema = {
		'task type'     : 'str',
		'grid number'       : 'int',
		'target category'   : 'str',
		'absolute position' : 'str',
		'reference category': 'str',
		'reference position': 'str',
		'attribute type'    : 'str',
		'attribute value'   : 'str',
	}

	def __init__(self, metadata: ObjaverseMetaData, seed=42):
		super().__init__(metadata, seed=seed)
		self.relative_position_phrase = relative_position_phrase
		self.relative_positions = relative_positions

	def enumerate_task_plans(self, task_store: TaskStore):
		for target_category in tqdm(self.metadata.categories, desc="enumerating [what attribute] task"):
			irrelevant_categories = self.metadata.get_irrelevant_categories(target_category)
			attribute_dict = self.metadata.get_category_attribute_dict(target_category)
			for attribute_type, attribute_values in attribute_dict.items():
				for attribute_value in attribute_values:
					for grid_size in self.grid_options:
						for absolute_pos in self.grid_mappings[grid_size]:
							task_plan = {
								'task type'    : 'what attribute',
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
											'task type'     : 'what attribute',
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
		attribute_value = task_plan['attribute value']

		target_category = task_plan['target category']
		categories = [target_category]
		absolute_pos = task_plan['absolute position']
		grids = [self.grid_mappings[grid_size][absolute_pos]]

		queries = [self._get_target_object_query(task_plan)]
		remaining_query = []

		if 'reference category' in task_plan:
			reference_pos = task_plan['reference position']
			reference_category = task_plan['reference category']
			categories.append(reference_category)

			queries.append(self.metadata.and_query([("category", reference_category, True)]))
			remaining_query += [("category", reference_category, False)]

			ref_grid = self._relative_grid(grid_size, grids[0], reference_pos)
			assert ref_grid >= 0, "reference grid {} not allowed".format(ref_grid)
			grids.append(ref_grid)

			question = f"What is the {attribute_type} of the object {self.relative_position_phrase[reference_pos]} the {self.metadata.get_surfacename(reference_category)}?"
		else:
			question = f"What is the {attribute_type} of the object in the {absolute_pos} part of the image?"

		image_metadata = self._make_image_metadata(
			grid_size,
			grids,
			queries=queries,
			remaining_query=self.metadata.and_query([("category", target_category, False)])
		)

		answer = attribute_value
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


class WhereAttributeGridTaskGenerator(GridTaskGenerator):
	schema = {
		'task type'     : 'str',
		'grid number'       : 'int',
		'target category'   : 'str',
		'absolute position' : 'str',
		'reference category': 'str',
		'reference position': 'str',
		'attribute type'    : 'str',
		'attribute value'   : 'str',
	}

	def __init__(self, metadata: ObjaverseMetaData, seed=42):
		super().__init__(metadata, seed=seed)
		self.relative_positions = relative_positions

	def enumerate_task_plans(self, task_store: TaskStore):
		for target_category in tqdm(self.metadata.categories, desc="enumerating [where attribute] task"):
			irrelevant_categories = self.metadata.get_irrelevant_categories(target_category)
			attribute_dict = self.metadata.get_category_attribute_dict(target_category)
			for attribute_type, attribute_values in attribute_dict.items():
				for attribute_value in attribute_values:
					for grid_size in self.grid_options:
						for absolute_pos in self.grid_mappings[grid_size]:
							task_plan = {
								'task type'    : 'where attribute',
								'grid number'      : grid_size,
								'target category'  : target_category,
								'absolute position': absolute_pos,
								'attribute type'   : attribute_type,
								'attribute value'  : attribute_value,
							}
							task_store.add(task_plan)

							grid = self.grid_mappings[grid_size][absolute_pos]
							for reference_pos in self.relative_positions:
								ref_grid = self._relative_grid(grid_size, grid, reference_pos)
								if ref_grid >= 0:
									for reference_category in irrelevant_categories:
										task_plan = {
											'task type'     : 'where attribute',
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
		attribute_value = task_plan['attribute value']

		target_category = task_plan['target category']
		categories = [target_category]
		absolute_pos = task_plan['absolute position']
		grids = [self.grid_mappings[grid_size][absolute_pos]]

		queries = [self._get_target_object_query(task_plan)]
		remaining_query = [(attribute_type, attribute_value, False), (attribute_type, None, False)]

		if 'reference category' in task_plan:
			reference_pos = task_plan['reference position']
			reference_category = task_plan['reference category']
			categories.append(reference_category)

			queries.append(self.metadata.and_query([("category", reference_category, True)]))
			remaining_query.append(("category", reference_category, False))

			ref_grid = self._relative_grid(grid_size, grids[0], reference_pos)
			assert ref_grid >= 0, "reference grid {} not allowed".format(ref_grid)
			grids.append(ref_grid)

			answer = reference_pos
			negatives = [o for o in self.relative_positions if o != answer]
			question = f"Where is the {attribute_value} object with respect to the {self.metadata.get_surfacename(reference_category)}?"
		else:
			answer = absolute_pos
			negatives = [o for o in self.grid_mappings[grid_size].keys() if o != answer]
			question = f"Where is the {attribute_value} object in the image?"

		options = self._compose_options(answer, negatives)
		image_metadata = self._make_image_metadata(
			grid_size,
			grids,
			queries=queries,
			remaining_query=self.metadata.and_query(remaining_query)
		)

		return question, answer, options, image_metadata


class HowManyGridTaskGenerator(GridTaskGenerator):
	schema = {
		'task type'  : 'str',
		'grid number'    : 'int',
		'target category': 'str',
		'count'          : 'int',
		'attribute type' : 'str',
		'attribute value': 'str',
	}

	def __init__(self, metadata: ObjaverseMetaData, seed=42):
		super().__init__(metadata, seed=seed)

	def enumerate_task_plans(self, task_store: TaskStore):
		for attribute_type, attribute_values in tqdm(self.metadata.attribute_vocab.items(), desc="enumerating [how many attribute 1] task"):
			for attribute_value in attribute_values:
				for grid_size in self.grid_options:
					for count in range(1, grid_size ** 2 + 1):
						task_plan = {
							'task type'  : 'how many',
							'grid number'    : grid_size,
							'count'          : count,
							'attribute type' : attribute_type,
							'attribute value': attribute_value,
						}
						task_store.add(task_plan)

		for target_category in tqdm(self.metadata.categories, desc="enumerating [how many attribute 2] task"):
			attribute_dict = self.metadata.get_category_attribute_dict(target_category)

			for grid_size in self.grid_options:
				for count in range(1, grid_size ** 2 + 1):
					task_plan = {
						'task type'  : 'how many',
						'grid number'    : grid_size,
						'target category': target_category,
						'count'          : count,
					}
					task_store.add(task_plan)

					for attribute_type, attribute_values in attribute_dict.items():
						for attribute_value in attribute_values:
							task_plan = {
								'task type'  : 'how many',
								'grid number'    : grid_size,
								'count'          : count,
								'target category': target_category,
								'attribute type' : attribute_type,
								'attribute value': attribute_value,
							}
							task_store.add(task_plan)

	def _generate_task(self, task_plan):
		grid_size = task_plan['grid number']
		count = int(task_plan['count'])

		if 'target category' in task_plan:
			target_category = task_plan['target category']
			target_category_name = self.metadata.get_surfacename(target_category)

			if 'attribute type' in task_plan:
				attribute_type = task_plan['attribute type']
				attribute_value = task_plan['attribute value']

				query = self.metadata.and_query([("category", target_category, True), (attribute_type, attribute_value, True)])
				remaining_query1 = self.metadata.and_query([("category", target_category, False)])
				remaining_query2 = self.metadata.and_query([(attribute_type, attribute_value, False), (attribute_type, None, False)])
				remaining_query = self.metadata.or_query([remaining_query1, remaining_query2])
				question = f"How many {attribute_value} {target_category_name} are there in the image?"

			else:
				query = self.metadata.and_query([("category", target_category, True)])
				remaining_query = self.metadata.and_query([("category", target_category, False)])
				question = f"How many {target_category_name} are there in the image?"

		else:
			attribute_type = task_plan['attribute type']
			attribute_value = task_plan['attribute value']

			query = self.metadata.and_query([(attribute_type, attribute_value, True)])
			remaining_query = self.metadata.and_query([(attribute_type, attribute_value, False), (attribute_type, None, False)])
			question = f"How many {attribute_value} objects are there in the image?"

		answer = str(count)
		negatives = [str(i + 1) for i in range(grid_size ** 2) if i + 1 != count]
		options = self._compose_options(answer, negatives)

		grids = [int(i) for i in self.rng.choice(range(grid_size ** 2), count, replace=False)]
		image_metadata = self._make_image_metadata(
			grid_size,
			grids,
			queries=[query] * count,
			remaining_query=remaining_query
		)

		return question, answer, options, image_metadata
