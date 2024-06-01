from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from ..metadata import VideoSceneGraphMetaData
from ...base import TaskGenerator
from ...task_store import TaskStore


def load_mp4_video(video_path):
	with open(video_path, "rb") as file:
		mp4_data = file.read()
	return mp4_data


def enumerate_target_relation_to_possible_reference_actions(video_scene_graph, relation_type, temporal_reference_type):
	relation_to_actions = {}
	video_scene_graph_keyframes = list(video_scene_graph.keys())

	if temporal_reference_type == "before":
		for idx, keyframe_name in enumerate(video_scene_graph_keyframes[:-1]):
			next_keyframe_name = video_scene_graph_keyframes[idx + 1]
			for relation, obj in video_scene_graph[keyframe_name][relation_type].items():
				if relation not in video_scene_graph[next_keyframe_name][relation_type]:
					if relation not in relation_to_actions:
						relation_to_actions[(relation, obj)] = set()
					for after_keyframe in video_scene_graph_keyframes[idx + 1:]:
						for action in video_scene_graph[after_keyframe]['actions']:
							if action not in video_scene_graph[keyframe_name]['actions']:
								relation_to_actions[(relation, obj)].add(action)

	elif temporal_reference_type == "after":
		for idx, keyframe_name in enumerate(video_scene_graph_keyframes[1:], start=1):
			previous_keyframe_name = video_scene_graph_keyframes[idx - 1]
			for relation, obj in video_scene_graph[keyframe_name][relation_type].items():
				if relation not in video_scene_graph[previous_keyframe_name][relation_type]:
					if relation not in relation_to_actions:
						relation_to_actions[(relation, obj)] = set()
					for before_keyframe in video_scene_graph_keyframes[:idx]:
						for action in video_scene_graph[before_keyframe]['actions']:
							if action not in video_scene_graph[keyframe_name]['actions']:
								relation_to_actions[(relation, obj)].add(action)

	elif temporal_reference_type == "while":
		for idx, keyframe_name in enumerate(video_scene_graph_keyframes):
			for relation, obj in video_scene_graph[keyframe_name][relation_type].items():
				if relation not in relation_to_actions:
					relation_to_actions[(relation, obj)] = set()
				for action in video_scene_graph[keyframe_name]['actions']:
					relation_to_actions[(relation, obj)].add(action)

	# Convert sets to lists for the output
	relation_to_actions = {k: list(v) for k, v in relation_to_actions.items()}
	return relation_to_actions


def enumerate_target_action_to_possible_reference_actions(video_scene_graph, temporal_reference_type):
	action_to_actions = {}
	video_scene_graph_keyframes = list(video_scene_graph.keys())

	if temporal_reference_type == "before":
		for idx, keyframe_name in enumerate(video_scene_graph_keyframes[:-1]):
			next_keyframe_name = video_scene_graph_keyframes[idx + 1]
			for action in video_scene_graph[keyframe_name]['actions']:
				if action not in video_scene_graph[next_keyframe_name]['actions']:
					if action not in action_to_actions:
						action_to_actions[action] = set()
					for after_keyframe in video_scene_graph_keyframes[idx + 1:]:
						for reference_action in video_scene_graph[after_keyframe]['actions']:
							if reference_action not in video_scene_graph[keyframe_name]['actions'] and reference_action != action:
								action_to_actions[action].add(reference_action)

	elif temporal_reference_type == "after":
		for idx, keyframe_name in enumerate(video_scene_graph_keyframes[1:], start=1):
			previous_keyframe_name = video_scene_graph_keyframes[idx - 1]
			for action in video_scene_graph[keyframe_name]['actions']:
				if action not in video_scene_graph[previous_keyframe_name]['actions']:
					if action not in action_to_actions:
						action_to_actions[action] = set()
					for before_keyframe in video_scene_graph_keyframes[:idx]:
						for reference_action in video_scene_graph[before_keyframe]['actions']:
							if reference_action not in video_scene_graph[keyframe_name]['actions'] and reference_action != action:
								action_to_actions[action].add(reference_action)

	elif temporal_reference_type == "while":
		for idx, keyframe_name in enumerate(video_scene_graph_keyframes):
			for action in video_scene_graph[keyframe_name]['actions']:
				if action not in action_to_actions:
					action_to_actions[action] = set()
				for reference_action in video_scene_graph[keyframe_name]['actions']:
					if reference_action != action:
						action_to_actions[action].add(reference_action)

	# Convert sets to lists for the output
	action_to_actions = {k: list(v) for k, v in action_to_actions.items()}
	return action_to_actions


def get_all_spatial_relations(video_scene_graph):
	relations = set()
	for keyframe_name, keyframe in video_scene_graph.items():
		relations.update(keyframe['spatial'])
	return relations


def get_all_contact_relations(video_scene_graph):
	relations = set()
	for keyframe_name, keyframe in video_scene_graph.items():
		relations.update(keyframe['contact'])
	return relations


def get_all_objects(video_scene_graph):
	objects = set()
	for keyframe_name, keyframe in video_scene_graph.items():
		for relation in keyframe['spatial']:
			objects.add(keyframe['spatial'][relation])
		for relation in keyframe['contact']:
			objects.add(keyframe['contact'][relation])
	return objects


def get_all_actions(video_scene_graph):
	actions = set()
	for keyframe_name, keyframe in video_scene_graph.items():
		actions.update(keyframe['actions'])
	return actions


class VideoSceneGraphTaskGenerator(TaskGenerator):
	metadata: VideoSceneGraphMetaData

	embed_schema = [
		"task type",
		"object",
		"relation",
		"action",
		"reference action",
		"relation type",
		"temporal reference type",
	]

	def __init__(self, metadata: VideoSceneGraphMetaData, seed=42):
		super().__init__(metadata, seed=seed)

	def _generate_task(self, task_plan) -> Tuple[str, str, List[str], str]:
		"(Abstract method) generate task"

	def _task_plan_to_str(self, task_plan) -> str:
		t = []
		for k, v in task_plan.items():
			if k in self.embed_schema:
				assert isinstance(v, str)
				t.append(f'{k}: {v}')
		return '\n'.join(t)

	def generate(self, task_plan, return_data=True, seed=None):
		if seed is not None:
			self.rng = np.random.default_rng(seed=seed)

		question, answer, options, video_scene_graph_id = self._generate_task(task_plan)

		task = {
			"question"            : question,
			"answer"              : answer,
			"options"             : options,
			"task_plan"           : self._task_plan_to_str(task_plan),
			"video_scene_graph_id": video_scene_graph_id,
			'video'               : load_mp4_video(self.metadata.get_video_path(video_scene_graph_id)) if return_data else None
		}
		return task


class WhatObjectVideoSceneGraphTaskGenerator(VideoSceneGraphTaskGenerator):
	schema = {
		"task type"          : "str",
		"object"                 : "str",
		"relation"               : "str",
		"reference action"       : "str",
		"relation type"          : "str",
		"temporal reference type": "str",
		"video scene graph id"   : "str",
	}

	def enumerate_task_plans(self, task_store: TaskStore):
		for video_scene_graph_id, video_scene_graph in tqdm(self.metadata.video_scene_graphs.items(), desc="enumerating [what object video] task"):
			for relation_type in ["spatial", "contact"]:
				for temporal_reference_type in ["before", "after", "while"]:
					target_relation_to_possible_reference_actions = enumerate_target_relation_to_possible_reference_actions(video_scene_graph, relation_type, temporal_reference_type)
					for (target_relation, target_object), possible_reference_actions in target_relation_to_possible_reference_actions.items():
						for reference_action in possible_reference_actions:
							task_plan = {
								"task type"          : "what object video",
								"video scene graph id"   : video_scene_graph_id,
								"object"                 : self.metadata.idx2name[target_object],
								"relation"               : self.metadata.idx2name[target_relation],
								'relation type'          : relation_type,
								"reference action"       : self.metadata.idx2name[reference_action],
								"temporal reference type": temporal_reference_type,
							}
							task_store.add(task_plan)

	def _generate_task(self, task_plan):
		question = f"What is the object that the person is {task_plan['relation']} {task_plan['temporal reference type']} the person {task_plan['reference action']}?"

		answer = task_plan["object"]
		negatives = list(set(self.metadata.objects) - get_all_objects(self.metadata.video_scene_graphs[task_plan["video scene graph id"]]))
		negatives = [self.metadata.idx2name[neg] for neg in negatives]

		options = self._compose_options(answer, negatives)
		return question, answer, options, task_plan["video scene graph id"]


class WhatRelationVideoSceneGraphTaskGenerator(VideoSceneGraphTaskGenerator):
	schema = {
		"task type"          : "str",
		"object"                 : "str",
		"relation"               : "str",
		"reference action"       : "str",
		"relation type"          : "str",
		"temporal reference type": "str",
		"video scene graph id"   : "str",
	}

	def enumerate_task_plans(self, task_store: TaskStore):
		for video_scene_graph_id, video_scene_graph in tqdm(self.metadata.video_scene_graphs.items(), desc="enumerating [what relation video] task"):
			for relation_type in ["spatial", "contact"]:
				for temporal_reference_type in ["before", "after", "while"]:
					target_relation_to_possible_reference_actions = enumerate_target_relation_to_possible_reference_actions(video_scene_graph, relation_type, temporal_reference_type)
					for (target_relation, target_object), possible_reference_actions in target_relation_to_possible_reference_actions.items():
						for reference_action in possible_reference_actions:
							task_plan = {
								"task type"          : "what relation video",
								"video scene graph id"   : video_scene_graph_id,
								"object"                 : self.metadata.idx2name[target_object],
								"relation"               : self.metadata.idx2name[target_relation],
								'relation type'          : relation_type,
								"reference action"       : self.metadata.idx2name[reference_action],
								"temporal reference type": temporal_reference_type,
							}
							task_store.add(task_plan)

	def _generate_task(self, task_plan):
		if task_plan["relation type"] == "spatial":
			question = f"What is the spatial relation of the person to the {task_plan['object']} {task_plan['temporal reference type']} the person {task_plan['reference action']}?"
			negatives = list(set(self.metadata.spatial_relations) - get_all_spatial_relations(self.metadata.video_scene_graphs[task_plan["video scene graph id"]]))
		elif task_plan["relation type"] == "contact":
			question = f"What is the person doing to the {task_plan['object']} {task_plan['temporal reference type']} the person {task_plan['reference action']}?"
			negatives = list(set(self.metadata.contact_relations) - get_all_contact_relations(self.metadata.video_scene_graphs[task_plan["video scene graph id"]]))
		else:
			raise ValueError(f"Unknown relation type: {task_plan['relation type']}")

		answer = task_plan['relation']
		negatives = [self.metadata.idx2name[neg] for neg in negatives]

		options = self._compose_options(answer, negatives)
		return question, answer, options, task_plan["video scene graph id"]


class WhatActionVideoSceneGraphTaskGenerator(VideoSceneGraphTaskGenerator):
	schema = {
		"task type"          : "str",
		"action"                 : "str",
		"reference action"       : "str",
		"relation type"          : "str",
		"temporal reference type": "str",
		"video scene graph id"   : "str",
	}

	def enumerate_task_plans(self, task_store: TaskStore):
		for video_scene_graph_id, video_scene_graph in tqdm(self.metadata.video_scene_graphs.items(), desc="enumerating [what action video] task"):
			for temporal_reference_type in ["before", "after", "while"]:
				target_action_to_possible_reference_actions = enumerate_target_action_to_possible_reference_actions(video_scene_graph, temporal_reference_type)
				for target_action, possible_reference_actions in target_action_to_possible_reference_actions.items():
					for reference_action in possible_reference_actions:
						task_plan = {
							"task type"          : "what action video",
							"video scene graph id"   : video_scene_graph_id,
							"action"                 : self.metadata.idx2name[target_action],
							"reference action"       : self.metadata.idx2name[reference_action],
							"temporal reference type": temporal_reference_type,
						}
						task_store.add(task_plan)

	def _generate_task(self, task_plan):
		question = f"What action is the person doing {task_plan['temporal reference type']} the person {task_plan['reference action']}?"

		answer = task_plan["action"]
		negatives = list(set(self.metadata.actions) - get_all_actions(self.metadata.video_scene_graphs[task_plan["video scene graph id"]]))
		negatives = [self.metadata.idx2name[neg] for neg in negatives]

		options = self._compose_options(answer, negatives)
		return question, answer, options, task_plan["video scene graph id"]
