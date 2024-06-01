import json
import os
from itertools import product
from math import radians
from typing import List, Tuple

import networkx as nx
import pandas as pd

from ..metadata import CategoryMetaData

ambiguous_colors = [
	["red", "pink", "purple"],
	["yellow", "orange", "brown", "gold", "beige"],
]


def get_confusing_colors(color):
	for colors in ambiguous_colors:
		if color in colors:
			return colors
	return [color]


def remove_skip_edge(edges):
	G = nx.DiGraph()
	G.add_edges_from(edges)
	new_edges = []
	for source, target in edges:
		G.remove_edge(source, target)
		if not nx.has_path(G, source, target):
			G.add_edge(source, target)
			new_edges.append((source, target))
	return new_edges


def remove_nodes(G, nodes):
	for node in nodes:
		successors = list(G.successors(node))
		predecessors = list(G.predecessors(node))
		G.remove_node(node)
		for s in successors:
			for p in predecessors:
				G.add_edge(p, s)
	return G


def build_taxonomy(path_to_metadata, mode):
	assert mode in ['objaverse', 'scene_graph']

	cateid_to_concept = json.load(open(os.path.join(path_to_metadata, 'cateid_to_concept.json')))
	taxonomy = json.load(open(os.path.join(path_to_metadata, 'taxonomy.json')))
	edges, nodes = taxonomy['edges'], taxonomy['nodes']
	G = nx.DiGraph()
	G.add_edges_from(remove_skip_edge(edges))

	nodes_to_remove = []
	categories_with_object = set([k for k, v in cateid_to_concept.items() if len(v[mode]) > 0])
	for node in G.nodes():
		if node not in categories_with_object and len(nx.descendants(G, node) & categories_with_object) == 0:
			nodes_to_remove.append(node)
	G = remove_nodes(G, nodes_to_remove)
	G.add_nodes_from(categories_with_object)

	categories, category_info = [], {}
	for node in G.nodes():
		categories.append(node)
		if node in cateid_to_concept:
			category_info[node] = cateid_to_concept[node]
		else:
			category_info[node] = nodes[node]
	categories = sorted(categories)

	return G, categories, category_info, categories_with_object


class ObjaverseMetaData(CategoryMetaData):
	def __init__(self, path_to_metadata):
		super().__init__()

		self.taxonomy, self.categories, self.category_info, categories_with_object = \
			build_taxonomy(path_to_metadata, 'objaverse')

		cateid_to_objects = json.load(open(os.path.join(path_to_metadata, 'cateid_to_objects.json')))

		def get_category_objects(category):
			if category in cateid_to_objects:
				return list(cateid_to_objects[category].keys())
			else:
				return []

		cateid_to_objid = {}
		for cateid in self.categories:
			objs = get_category_objects(cateid)
			for c in nx.descendants(self.taxonomy, cateid) & categories_with_object:
				objs.extend(get_category_objects(c))
			cateid_to_objid[cateid] = objs
			assert len(objs) > 0
			assert len(objs) == len(set(objs))

		self.attribute_vocab, objid_to_attribute = {}, {}
		for cateid in cateid_to_objects:
			for objid in cateid_to_objects[cateid]:
				objid_to_attribute[objid] = cateid_to_objects[cateid][objid]["attributes"]
				for attr, values in cateid_to_objects[cateid][objid]["attributes"].items():
					if attr not in self.attribute_vocab:
						self.attribute_vocab[attr] = set()
					self.attribute_vocab[attr].update(values)

		data = []
		for cateid, objs in cateid_to_objid.items():
			for objid in objs:
				attribute_data = []
				for attr in self.attribute_vocab:
					values = objid_to_attribute[objid].get(attr, [])
					if len(values) == 0:
						values = [None]
					attribute_data.append(values)

				for attribute_combination in product(*attribute_data):
					data.append([objid, cateid] + list(attribute_combination))

		self.df = pd.DataFrame(data, columns=['object', 'category'] + list(self.attribute_vocab.keys()))

	def check_object_attribute(self, objid, attributes):
		for attr, values in attributes.items():
			for value in values:
				if value not in self.df[self.df['object'] == objid][attr].unique():
					return False
		return True

	def and_query(self, conditions: List[Tuple]) -> str:
		q = set()
		for k, v, i in conditions:
			# k: column name; v: value; i: is equal
			if v is None:
				if i:
					q.add(f'{k} in [None]')
				else:
					q.add(f'{k} not in [None]')
			else:
				if i:
					q.add(f'{k} == {repr(v)}')
				else:
					if k == 'category':
						# exclude all relevant categories
						for c in self.get_relevant_categories(v):
							q.add(f'{k} != {repr(c)}')
					elif k == 'color':
						# exclude all confusing colors
						for c in get_confusing_colors(v):
							q.add(f'{k} != {repr(c)}')
					else:
						q.add(f'{k} != {repr(v)}')
		return ' and '.join(q)

	def or_query(self, conditions: List[str]) -> str:
		conditions = [f'({c})' for c in conditions if len(c) > 0]
		return ' or '.join(conditions)

	def query_metadata(self, target, query: str):
		if len(query) == 0:
			return sorted(self.df[target].dropna().unique())
		else:
			return sorted(self.df.query(query)[target].dropna().unique().tolist())

	def sample(self, rng, n, target, query: str):
		if n == 1:
			return rng.choice(self.query_metadata(target, query))
		else:
			candidates = self.query_metadata(target, query)
			return rng.choice(candidates, n, replace=len(candidates) < n).tolist()

	def sample_category_for_object(self, rng, objid, exclude_category=None):
		candidates = self.query_metadata("category", self.and_query([("object", objid, True)]))
		if exclude_category is not None:
			exclude_category = self.get_relevant_categories(exclude_category)
			candidates = [c for c in candidates if c not in exclude_category]
		return rng.choice(candidates)

	def get_category_attribute_dict(self, cateid):
		attribute_dict = {}
		for attr in self.attribute_vocab:
			attribute_dict[attr] = self.query_metadata(attr, self.and_query([("category", cateid, True)]))
		return attribute_dict


class Objaverse2DMetaData(ObjaverseMetaData):
	def __init__(self, path_to_metadata, image_folder):
		super().__init__(path_to_metadata)

		self.image_folder = image_folder
		cateid_to_objects = json.load(open(os.path.join(path_to_metadata, 'cateid_to_objects.json')))

		self.objid_to_images = {}
		for cateid in cateid_to_objects:
			for objid in cateid_to_objects[cateid]:
				self.objid_to_images[objid] = [os.path.join(image_folder, cateid, objid, i)
											   for i in cateid_to_objects[cateid][objid]["images"]]

	def sample_image(self, rng, objid):
		return rng.choice(self.objid_to_images[objid])


class Objaverse3DMetaData(ObjaverseMetaData):
	def __init__(self, path_to_metadata, blender_path, assets_path, render_device='cpu', blender_cache='./blender_cache'):
		super().__init__(path_to_metadata)
		self.assets_path = assets_path
		self.blender_path = blender_path
		self.blender_cache = blender_cache
		self.render_device = render_device
		plane_dir = os.path.join(assets_path, "plane_glbs")
		self.plane_texture_path = [os.path.join(plane_dir, f) for f in os.listdir(plane_dir) if f.endswith(".glb")]
		hdri_dir = os.path.join(assets_path, "hdri")
		self.hdri_path = [os.path.join(hdri_dir, f) for f in os.listdir(hdri_dir) if f.endswith(".exr")]

		cateid_to_objects = json.load(open(os.path.join(path_to_metadata, 'cateid_to_objects.json')))
		self.object_to_angles = {objid: cateid_to_objects[cateid][objid]['angles']
								 for cateid in cateid_to_objects for objid in cateid_to_objects[cateid]}

	def get_object_path(self, objid):
		return os.path.join(self.assets_path, "objects", objid + ".glb")

	def sample_object_angle(self, rng, objid):
		angles = self.object_to_angles[objid]
		return angles[rng.choice(len(angles))]

	def sample_blender_configuration(self, rng):
		orientation = rng.choice([-1, 1])
		key_light_horizontal_angle = orientation * radians(rng.uniform(15, 45))
		fill_light_horizontal_angle = - orientation * radians(rng.uniform(15, 60))
		key_light_vertical_angle = -radians(rng.uniform(15, 45))
		fill_light_vertical_angle = -radians(rng.uniform(0, 30))

		sun_x, sun_y = radians(rng.uniform(0, 45)), radians(rng.uniform(0, 45))
		sun_energy = rng.uniform(1.0, 6.0)

		plane_texture_path = rng.choice(self.plane_texture_path)
		hdri_path = rng.choice(self.hdri_path)

		return {
			"key_light_horizontal_angle" : key_light_horizontal_angle,
			"fill_light_horizontal_angle": fill_light_horizontal_angle,
			"key_light_vertical_angle"   : key_light_vertical_angle,
			"fill_light_vertical_angle"  : fill_light_vertical_angle,
			"sun_x"                      : sun_x,
			"sun_y"                      : sun_y,
			"sun_energy"                 : sun_energy,
			"plane_texture_path"         : plane_texture_path,
			"hdri_path"                  : hdri_path
		}


def load_scene_graph(scene_graph_folder):
	image_folder = os.path.join(scene_graph_folder, "images/images")
	sg_json_folder = os.path.join(scene_graph_folder, "sceneGraphs")
	# train_scene_graphs = json.load(open(os.path.join(sg_json_folder, "train_sceneGraphs.json")))
	val_scene_graphs = json.load(open(os.path.join(sg_json_folder, "val_sceneGraphs.json")))
	scene_graphs = val_scene_graphs  # TODO: first only use val_scene_graphs
	return image_folder, scene_graphs


class SceneGraphMetaData(CategoryMetaData):
	def __init__(self, path_to_metadata, scene_graph_folder):
		super().__init__()
		self.taxonomy, self.categories, self.category_info, self.categories_with_object = \
			build_taxonomy(path_to_metadata, 'scene_graph')

		self.type_to_attribute = json.load(open(os.path.join(path_to_metadata, 'attribute_category.json')))
		self.attribute_to_type = {attr: k for k, vs in self.type_to_attribute.items() for attr in vs}

		self.image_folder, self.scene_graphs = load_scene_graph(scene_graph_folder)
		self.scene_graphs_list = list(self.scene_graphs.keys())
		self.sg_object_to_cateid = {}
		for k, v in self.category_info.items():
			if k in self.categories_with_object:
				for sg_object in v['scene_graph']:
					self.sg_object_to_cateid[sg_object] = k

		relations = set()
		for sg in self.scene_graphs.values():
			for obj in sg['objects'].values():
				for rel in obj['relations']:
					relations.add(rel['name'])
		self.relations = list(relations)

	def check_object_in_category(self, object_name):
		return object_name in self.sg_object_to_cateid

	def object_name_to_cateid(self, object_name):
		return self.sg_object_to_cateid[object_name]

	def get_attribute_type(self, attribute):
		return self.attribute_to_type.get(attribute, "other")

	def get_image_path(self, scene_graph_id):
		return os.path.join(self.image_folder, scene_graph_id + ".jpg")
