from .utils import grid_mappings, grid_options, make_image, relative_grid, relative_position_phrase, relative_positions
from ..metadata import Objaverse3DMetaData, ObjaverseMetaData
from ..sticker_2d import GridTaskGenerator, HowManyGridTaskGenerator, WhatAttributeGridTaskGenerator, WhatGridTaskGenerator, WhereAttributeGridTaskGenerator, WhereGridTaskGenerator
from ...constant import IMAGE_H, IMAGE_W


class _3DGridTaskGenerator(GridTaskGenerator):
	metadata: Objaverse3DMetaData

	def __init__(self, metadata: ObjaverseMetaData, seed=42):
		super().__init__(metadata, seed=seed)
		self.grid_mappings = grid_mappings
		self.grid_options = grid_options
		self.relative_positions = relative_positions
		self.relative_position_phrase = relative_position_phrase

	def _make_image_metadata(self, grid_size, grids, queries, remaining_query=...):
		objects = [self.metadata.sample(self.rng, 1, "object", q) for q in queries]

		remaining_grids = [g for g in range(grid_size ** 2) if g not in grids]
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
			'blender_config': self.metadata.sample_blender_configuration(self.rng)
		}

		return image_metadata

	def make_image(self, image_metadata):
		return make_image(image_metadata, self.metadata, IMAGE_H, IMAGE_W)

	def _relative_grid(self, grid_size, grid, reference_pos):
		return relative_grid(grid_size, grid, reference_pos)


class What3DGridTaskGenerator(_3DGridTaskGenerator, WhatGridTaskGenerator):
	metadata: Objaverse3DMetaData


class Where3DGridTaskGenerator(_3DGridTaskGenerator, WhereGridTaskGenerator):
	metadata: Objaverse3DMetaData


class WhatAttribute3DGridTaskGenerator(_3DGridTaskGenerator, WhatAttributeGridTaskGenerator):
	metadata: Objaverse3DMetaData


class WhereAttribute3DGridTaskGenerator(_3DGridTaskGenerator, WhereAttributeGridTaskGenerator):
	metadata: Objaverse3DMetaData


class HowMany3DGridTaskGenerator(_3DGridTaskGenerator, HowManyGridTaskGenerator):
	metadata: Objaverse3DMetaData
