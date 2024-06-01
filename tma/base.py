from typing import Dict

import numpy as np

from .constant import NUM_OPTIONS
from .metadata import MetaData
from .task_store import TaskStore


class TaskGenerator:
	schema = {}

	def __init__(self, metadata: MetaData, seed=42):
		self.metadata = metadata
		self.rng = np.random.default_rng(seed=seed)

	def _compose_options(self, answer, negatives):
		if len(negatives) > NUM_OPTIONS - 1:
			negatives = self.rng.choice(negatives, NUM_OPTIONS - 1, replace=False).tolist()
		options = [answer] + negatives
		return options

	def _task_plan_to_str(self, task_plan) -> str:
		"(Abstract method) convert task plan to string for task embedding"

	def enumerate_task_plans(self, task_store: TaskStore):
		"(Abstract method) enumerate task plan"

	def generate(self, task_plan, return_data=True, seed=None):
		"(Abstract method) enumerate task"


class JointTaskGenerator:
	def __init__(self, metadata: MetaData, generators: Dict, seed=42):
		self.generators = {
			k: v(metadata, seed=seed) for k, v in generators.items()
		}
		self.stats = {generator_type: 0 for generator_type in generators}
		self.schema = {}
		for generator_type, generator in self.generators.items():
			self.schema.update(generator.schema)

	def enumerate_task_plans(self, task_store: TaskStore):
		for generator_type, generator in self.generators.items():
			before = len(task_store)
			generator.enumerate_task_plans(task_store)
			self.stats[generator_type] = len(task_store) - before
			print(f"Generated [{self.stats[generator_type]}] {generator_type} tasks")
		task_store.dump()

	def generate(self, task_plan, return_data=True, seed=None):
		return self.generators[task_plan['task type']].generate(task_plan, return_data=return_data, seed=seed)
