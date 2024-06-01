import argparse
import json
import math
import os
import sys
import urllib.request
from math import radians

try:
	import bpy
	from mathutils import Vector, Matrix, Quaternion, Euler
except ImportError:
	pass


def rotate(obj, degree):
	"""Rotates around the z axis by theta"""
	degree = -degree
	bpy.ops.object.select_all(action='DESELECT')
	obj.select_set(True)
	bpy.context.view_layer.objects.active = obj
	radian = radians(degree)
	bpy.context.object.rotation_mode = 'XYZ'
	rot_x, rot_y, rot_z = obj.rotation_euler
	obj.rotation_euler = Euler((rot_x, rot_y, rot_z + radian))
	freeze_transformation(obj)


def reset_scene():
	# delete everything that isn't part of a camera or a light
	bpy.ops.object.select_all(action="SELECT")
	for obj in bpy.data.objects:
		bpy.data.objects.remove(obj, do_unlink=True)
	bpy.ops.ptcache.free_bake_all()


def select_hierarchy(obj):
	"""Recursively select an object and all of its descendants."""
	obj.select_set(True)
	for child in obj.children:
		select_hierarchy(child)


def load_object(object_path: str) -> None:
	"""Loads a glb model into the scene."""
	bpy.ops.object.select_all(action='DESELECT')
	if object_path.endswith(".glb"):
		bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
	elif object_path.endswith(".fbx"):
		bpy.ops.import_scene.fbx(filepath=object_path)
	else:
		raise ValueError(f"Unsupported file type: {object_path}")

	base_name = os.path.basename(object_path)
	object_name, _ = os.path.splitext(base_name)
	bpy.context.view_layer.objects.active.name = object_name
	bpy.ops.object.select_all(action='DESELECT')

	obj = bpy.data.objects.get(object_name)
	# bpy.context.view_layer.objects.active = obj
	select_hierarchy(obj)
	bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
	meshes = [obj for obj in bpy.context.selected_objects if obj.type == "MESH"]
	non_meshes = [obj for obj in bpy.context.selected_objects if obj.type != "MESH"]
	bpy.ops.object.select_all(action="DESELECT")

	# delete non-mesh and consolidate

	for obj in non_meshes:
		obj.select_set(True)
	bpy.ops.object.delete()
	bpy.ops.object.select_all(action="DESELECT")
	for obj in meshes:
		obj.select_set(True)
	bpy.context.view_layer.objects.active = meshes[0]
	bpy.ops.object.join()
	bpy.context.view_layer.objects.active.name = object_name
	bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='BOUNDS')
	bpy.ops.object.select_all(action="DESELECT")

	return object_name


def scene_meshes():
	for obj in bpy.context.scene.objects.values():
		if isinstance(obj.data, (bpy.types.Mesh)):
			yield obj


def download_uid(uid_path, save_dir):
	return download_object(uid_path, save_dir)


def download_object(object_url, save_dir) -> str:
	"""Download the object and return the path."""
	# uid = uuid.uuid4()
	uid = object_url.split("/")[-1].split(".")[0]
	tmp_local_path = os.path.join(save_dir, f"{uid}.glb" + ".tmp")
	local_path = os.path.join(save_dir, f"{uid}.glb")
	# wget the file and put it in local_path
	os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
	urllib.request.urlretrieve(object_url, tmp_local_path)
	os.rename(tmp_local_path, local_path)
	# get the absolute path
	local_path = os.path.abspath(local_path)
	return local_path


def scene_bbox(single_obj=None, ignore_matrix=False):
	bbox_min = (math.inf,) * 3
	bbox_max = (-math.inf,) * 3
	found = False
	for obj in scene_meshes() if single_obj is None else [single_obj]:
		found = True
		for coord in obj.bound_box:
			coord = Vector(coord)
			if not ignore_matrix:
				coord = obj.matrix_world @ coord
			bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
			bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
	if not found:
		raise RuntimeError("no objects in scene to compute bounding box for")
	return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
	for obj in bpy.context.scene.objects.values():
		if not obj.parent:
			yield obj


def freeze_transformation(obj):
	bpy.context.view_layer.objects.active = obj
	obj.select_set(True)
	bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
	bpy.ops.object.select_all(action='DESELECT')


def scale(obj, scale_factor):
	bpy.ops.object.select_all(action='DESELECT')
	obj.select_set(True)
	bpy.ops.transform.resize(value=(scale_factor, scale_factor, scale_factor))
	bpy.ops.object.select_all(action='DESELECT')
	freeze_transformation(obj)


def get_3d_dimensions(obj):
	# pdb.set_trace()
	max_x, max_y, max_z = float("-inf"), float("-inf"), float("-inf")
	min_x, min_y, min_z = float("inf"), float("inf"), float("inf")

	for vertex in obj.data.vertices:
		v_world = obj.matrix_world @ vertex.co
		max_x, max_y, max_z = max(max_x, v_world.x), max(max_y, v_world.y), max(max_z, v_world.z)
		min_x, min_y, min_z = min(min_x, v_world.x), min(min_y, v_world.y), min(min_z, v_world.z)

	return (max_x - min_x, max_y - min_y, max_z - min_z)


def normalize_object(obj, factor=1.0):
	max_dimension = max(get_3d_dimensions(obj))
	scale_factor = factor * (1 / max_dimension)
	scale(obj, scale_factor)


def move_to_xy(obj, x, y):
	min_z = float('inf')
	for vertex in obj.data.vertices:
		z = obj.matrix_world @ vertex.co
		min_z = min(min_z, z.z)
	obj.location -= Vector((0, 0, min_z))
	freeze_transformation(obj)

	# move location x,y to sampled box center
	new_location = Vector((x, y, obj.location[2]))
	obj.location = new_location
	freeze_transformation(obj)


def normalize_scene():
	bbox_min, bbox_max = scene_bbox()
	scale = 1 / max(bbox_max - bbox_min)
	for obj in scene_root_objects():
		obj.scale = obj.scale * scale
	# Apply scale to matrix_world.
	bpy.context.view_layer.update()
	bbox_min, bbox_max = scene_bbox()
	offset = -(bbox_min + bbox_max) / 2
	for obj in scene_root_objects():
		obj.matrix_world.translation += offset
	bpy.ops.object.select_all(action="DESELECT")


def setup_plane_and_background(plane_texture_path, hdri_path):
	# load plane
	plane_name = load_object(plane_texture_path)
	plane = bpy.data.objects.get(plane_name)
	scale(plane, 0.5)

	# load light map
	print(f"HDRI PATH: {hdri_path}")
	bpy.ops.image.open(filepath=hdri_path)
	if bpy.data.worlds.get("World") is None:
		bpy.data.worlds.new("World")

	bpy.context.scene.world = bpy.data.worlds["World"]

	bpy.context.scene.world.use_nodes = True
	tree = bpy.context.scene.world.node_tree
	tree.nodes.clear()

	tex_env = tree.nodes.new(type="ShaderNodeTexEnvironment")
	tex_env.image = bpy.data.images[hdri_path.split('/')[-1]]  # Image name is typically the last part of the path
	background = tree.nodes.new(type="ShaderNodeBackground")
	output = tree.nodes.new(type="ShaderNodeOutputWorld")

	tree.links.new(tex_env.outputs[0], background.inputs[0])
	tree.links.new(background.outputs[0], output.inputs[0])

	return plane_texture_path + " " + hdri_path


def setup_camera_and_lights(
		sun_x,
		sun_y,
		sun_energy,
		key_light_horizontal_angle,
		fill_light_horizontal_angle,
		key_light_vertical_angle,
		fill_light_vertical_angle
):
	# for seeting up the three point lighting, we mostly follow https://courses.cs.washington.edu/courses/cse458/05au/reading/3point_lighting.pdf
	# in order to keep lights and camera on the hemisphere pointing to origin, we use a hierarchy of empties

	# create the sun

	bpy.ops.object.light_add(type="SUN")
	sun = bpy.context.active_object
	sun.rotation_euler = Euler((sun_x, sun_y, 0), "XYZ")
	sun.data.energy = sun_energy

	# create global empty

	bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
	x_rot, y_rot, z_rot = radians(90), radians(0), radians(-90)
	empty = bpy.context.scene.objects.get("Empty")

	# create camera

	# radius = random.uniform(1.8,2.2)
	radius = 2.5

	bpy.ops.object.camera_add(enter_editmode=False, align='VIEW', location=(-radius, 0, 0), rotation=Euler((x_rot, y_rot, z_rot), "XYZ"), scale=(1, 1, 1))
	cam = bpy.context.scene.objects.get("Camera")
	cam.data.lens = 35
	cam.data.sensor_width = 32
	bpy.context.scene.camera = cam

	# create camera empty

	bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
	x_rot, y_rot, z_rot = radians(90), radians(0), radians(-90)
	cam_empty = bpy.context.scene.objects.get("Empty.001")
	cam_empty.name = "camera_empty"

	# make camera empty parent of camera

	bpy.ops.object.select_all(action='DESELECT')
	cam.select_set(True)
	cam_empty.select_set(True)
	bpy.context.view_layer.objects.active = cam_empty
	bpy.ops.object.parent_set()
	bpy.ops.object.select_all(action='DESELECT')

	# make camera empty parent of global empty

	bpy.ops.object.select_all(action='DESELECT')
	cam_empty.select_set(True)
	empty.select_set(True)
	bpy.context.view_layer.objects.active = empty
	bpy.ops.object.parent_set()
	bpy.ops.object.select_all(action='DESELECT')

	light_names = ["key_light", "fill_light", "back_light"]
	light_energies = [1000., 300., 500.]

	for light_name, light_energy in zip(light_names, light_energies):
		# create light empty

		empty_name = light_name + "_empty"
		bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
		x_rot, y_rot, z_rot = radians(90), radians(0), radians(-90)
		light_empty = bpy.context.scene.objects.get("Empty.001")
		light_empty.name = empty_name

		# parent light empty to main (camera) empty

		bpy.ops.object.select_all(action='DESELECT')
		light_empty.select_set(True)
		empty.select_set(True)
		bpy.context.view_layer.objects.active = empty
		bpy.ops.object.parent_set()
		bpy.ops.object.select_all(action='DESELECT')

		# create light

		x_loc, y_loc, z_loc = -radius, 0, 0
		bpy.ops.object.light_add(type='POINT', radius=1, align='WORLD', location=(x_loc, y_loc, z_loc), rotation=Euler((x_rot, y_rot, z_rot), "XYZ"), scale=(1, 1, 1))
		bpy.data.objects["Point"].name = light_name
		light = bpy.data.objects[light_name]
		light.data.energy = light_energy
		# light.data.size = 0.5

		# parent light empty to light

		bpy.ops.object.select_all(action='DESELECT')
		light.select_set(True)
		light_empty.select_set(True)
		bpy.context.view_layer.objects.active = light_empty
		bpy.ops.object.parent_set()
		bpy.ops.object.select_all(action='DESELECT')

	# rotate camera and lights around the z-axis

	z_random_rot = radians(90)  # radians(random.uniform(0,360))
	empty.rotation_euler = Euler((0, 0, z_random_rot))

	# # raise the camera while having it point to origin

	# cam_y_random_rot = radians(random.uniform(10,50))
	# cam_empty.rotation_euler = Euler((0,cam_y_random_rot,0),"XYZ")

	bpy.context.view_layer.update()

	back_light_horizontal_angle = radians(180)
	light_horizontal_angles = [key_light_horizontal_angle, fill_light_horizontal_angle, back_light_horizontal_angle]
	for light_angle, light_name in zip(light_horizontal_angles, light_names):
		light_empty = bpy.data.objects[light_name + "_empty"]
		global_z = (light_empty.matrix_world.inverted() @ Vector((0.0, 0.0, 1.0, 0.0)))[:3]
		quat = Quaternion(global_z, light_angle)
		light_empty.rotation_euler = quat.to_euler()

	back_light_vertical_angle = 0
	light_vertical_angles = [key_light_vertical_angle, fill_light_vertical_angle, back_light_vertical_angle]
	# light_vertical_angles = [radians(-45)]*3

	for light_angle, light_name in zip(light_vertical_angles, light_names):
		light_empty = bpy.data.objects[light_name + "_empty"]
		global_x = (light_empty.matrix_world.inverted() @ Vector((1.0, 0.0, 0.0, 0.0)))[:3]
		quat = Quaternion(global_x, light_angle)
		euler_add = quat.to_euler()
		euler_current = light_empty.rotation_euler
		new_euler = Euler((euler_add[0] + euler_current[0], euler_add[1] + euler_current[1], euler_add[2] + euler_current[2]))
		light_empty.rotation_euler = new_euler

	# bpy.context.view_layer.update()

	return cam, empty


def render(fp):
	# Render image
	bpy.context.view_layer.update()
	bpy.context.scene.render.filepath = fp
	bpy.ops.render.render(write_still=True)


def setup_renderer(H, W, use_cpu=False):
	scene = bpy.context.scene
	render = bpy.context.scene.render

	render.engine = "CYCLES"
	render.image_settings.file_format = "PNG"
	render.image_settings.color_mode = "RGBA"
	render.resolution_x = W
	render.resolution_y = H
	render.resolution_percentage = 100

	scene.cycles.device = "CPU" if use_cpu else "GPU"
	scene.cycles.samples = 10 if use_cpu else 128
	scene.cycles.diffuse_bounces = 1
	scene.cycles.glossy_bounces = 1
	scene.cycles.transparent_max_bounces = 3
	scene.cycles.transmission_bounces = 3
	scene.cycles.filter_width = 0.01
	scene.cycles.use_denoising = True
	scene.render.film_transparent = False

	bpy.context.preferences.addons["cycles"].preferences.get_devices()
	# Set the device_type
	bpy.context.preferences.addons[
		"cycles"
	].preferences.compute_device_type = "METAL" if use_cpu else "CUDA"
	bpy.context.scene.view_settings.view_transform = 'Filmic'


# def randomize_camera_view(axis):
# 	euler_y = radians(random.uniform(-90, 90))
# 	euler_z = radians(random.uniform(0, 360))
# 	axis.rotation_euler = Euler((0, euler_y, euler_z))


def run_render(metadata, save_image_path, use_cpu):
	reset_scene()

	objs = []
	for uid in metadata["objects"]:
		object_path = metadata["object_path"][uid]
		objs.append(bpy.data.objects.get(load_object(object_path)))

	grid_number = metadata["grid number"]

	if grid_number == 2:
		locations = {
			0: [0.7, 0.5],
			1: [0.7, -0.5],
			2: [-0.6, 0.5],
			3: [-0.6, -0.5]
		}
		scale_factor = 1 / 2
	elif grid_number == 3:
		locations = {
			0: [0.9, 0.6],
			1: [0.9, 0],
			2: [0.9, -0.6],
			3: [0.0, 0.6],
			4: [0.0, 0.0],
			5: [0.0, -0.6],
			6: [-0.9, 0.6],
			7: [-0.9, 0.0],
			8: [-0.9, -0.6]
		}
		scale_factor = 1 / 3
	else:
		raise ValueError(f"Expected grid number to be 2 or 3 but got {grid_number}")

	# process rotate
	for idx, obj in enumerate(objs):
		rotate(obj, degree=metadata['object_angles'][idx])

	# process scale
	if "sizes" in metadata:
		for idx, obj in enumerate(objs):
			normalize_object(obj, factor=metadata['sizes'][idx] * scale_factor)
	else:
		for obj in objs:
			normalize_object(obj, factor=scale_factor)

	for pos, obj in zip(metadata["grids"], objs):
		x, y = locations[pos]
		move_to_xy(obj, x, y)

	blender_config = metadata["blender_config"]

	setup_plane_and_background(blender_config["plane_texture_path"], blender_config["hdri_path"])
	cam, axis = setup_camera_and_lights(
		blender_config["sun_x"],
		blender_config["sun_y"],
		blender_config["sun_energy"],
		blender_config["key_light_horizontal_angle"],
		blender_config["fill_light_horizontal_angle"],
		blender_config["key_light_vertical_angle"],
		blender_config["fill_light_vertical_angle"]
	)
	axis.rotation_euler = Euler((0, radians(45), 0))
	setup_renderer(H=metadata["H"], W=metadata["W"], use_cpu=use_cpu)
	render(save_image_path)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--save_local",
		type=str,
		default=""
	)
	parser.add_argument(
		"--save_image_path",
		type=str,
		default="render.png"
	)
	parser.add_argument(
		"--json_file",
		type=str,
		default="image_metadata.json"
	)

	parser.add_argument(
		"--use_cpu",
		action="store_true",
		default=False
	)

	argv = sys.argv[sys.argv.index("--") + 1:]
	args = parser.parse_args(argv)

	with open(args.json_file, "r") as f:
		metadata = json.load(f)

	run_render(metadata, args.save_image_path, args.use_cpu)
