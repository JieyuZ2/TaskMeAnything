import tempfile
from typing import Dict, List, Tuple

import numpy as np

from .utils import grid_mappings, grid_options, make_video, relative_grid
from ..metadata import ObjaverseVideoMetaData
from ...base import TaskGenerator
from ...constant import VIDEO_H, VIDEO_W


def check_video(video):
	from decord import VideoReader, cpu
	with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as tmp:
		try:
			with open(tmp.name, 'wb') as file:
				file.write(video)
			with open(tmp.name, 'rb') as f:
				VideoReader(f, ctx=cpu(0))
		except Exception as e:
			return False
	return True


class GridVideoTaskGenerator(TaskGenerator):
	metadata: ObjaverseVideoMetaData

	def __init__(self, metadata: ObjaverseVideoMetaData, seed=42):
		super().__init__(metadata, seed=seed)
		self.grid_options = grid_options
		self.grid_mappings = grid_mappings

	def _relative_grid(self, grid_size, grid, reference_pos):
		return relative_grid(grid_size, grid, reference_pos)

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

	def make_video(self, video_metadata):
		return make_video(video_metadata, self.metadata, VIDEO_H, VIDEO_W)

	def _generate_task(self, task_plan) -> Tuple[str, str, List[str], Dict]:
		"(Abstract method) generate task"

	def generate(self, task_plan, return_data=True, seed=None):
		if seed is not None:
			self.rng = np.random.default_rng(seed=seed)

		retry = 0
		while True:
			question, answer, options, video_metadata = self._generate_task(task_plan)
			task = {
				'question'      : question.replace('_', ' '),
				'answer'        : answer.replace('_', ' '),
				'options'       : [o.replace('_', ' ') for o in options],
				'task_plan'     : self._task_plan_to_str(task_plan),
				'video_metadata': video_metadata,
				'video'         : self.make_video(video_metadata) if return_data else None
			}
			if return_data:
				if check_video(task['video']):
					break
				else:
					retry -= 1
					if retry <= 0:
						raise Exception("Failed to generate video")
			else:
				break

		return task
