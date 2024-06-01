import json
import os
import pickle

from ..imageqa.metadata import Objaverse3DMetaData
from ..metadata import MetaData


class ObjaverseVideoMetaData(Objaverse3DMetaData):
	pass


def load_video_scene_graph(video_scene_graph_folder):
	video_folder = os.path.join(video_scene_graph_folder, "Charades_v1_480")
	scene_graphs = json.load(open(os.path.join(video_scene_graph_folder, "video_scene_graph/video_scene_graph.json")))
	idx2name = pickle.load(open(os.path.join(video_scene_graph_folder, "video_scene_graph/idx2name.pkl"), "rb"))
	objects = pickle.load(open(os.path.join(video_scene_graph_folder, "video_scene_graph/objects.pkl"), "rb"))
	actions = pickle.load(open(os.path.join(video_scene_graph_folder, "video_scene_graph/actions.pkl"), "rb"))
	spatial_relations = pickle.load(open(os.path.join(video_scene_graph_folder, "video_scene_graph/spatial_relations.pkl"), "rb"))
	contact_relations = pickle.load(open(os.path.join(video_scene_graph_folder, "video_scene_graph/contact_relations.pkl"), "rb"))
	return video_folder, scene_graphs, idx2name, objects, actions, spatial_relations, contact_relations


class VideoSceneGraphMetaData(MetaData):
	def __init__(self, path_to_metadata, video_scene_graph_folder):
		super().__init__()
		# video scene graph use idx to represent relations, objects, and actions, like r1, o1, idx_to_name is a dict to map idx to its name.
		self.image_folder, self.video_scene_graphs, self.idx2name, self.objects, self.actions, self.spatial_relations, self.contact_relations = (
			load_video_scene_graph(video_scene_graph_folder))

	def get_video_path(self, video_scene_graph_id):
		return os.path.join(self.image_folder, video_scene_graph_id + ".mp4")
