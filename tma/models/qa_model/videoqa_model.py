import tempfile
from typing import Callable, Union

import huggingface_hub
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import os

from .base_qa_model import QAModel, QAModelInstance
from .imageqa_model import ImageQAModel

videoqa_models = {
	"video-llama2-7b" : ("VideoLLaMA2", "video-llama2-7b"),
	"video-llama2-13b": ("VideoLLaMA2", "video-llama2-13b"),
	"video-llava-7b"  : ("VideoLLaVA", "LanguageBind/Video-LLaVA-7B"),
	"chat-univi-7b"   : ("ChatUniVi", "chat-univi-7b"),
	"chat-univi-13b"  : ("ChatUniVi", "chat-univi-13b"),
	"video-chatgpt-7b": ("VideoChatGPT", "video-chatgpt-7b"),
	"video-chat2-7b"  : ("VideoChat2", "video-chat2-7b"),
	"vqamodel"        : ("VQAModel", "vqa_model_you_choose")
}


def list_videoqa_models():
	return list(videoqa_models.keys())


class VideoQAModel(QAModel):
	def __init__(
			self,
			model_name,
			prompt_name: str,
			prompt_func: Callable,
			model: QAModelInstance = None,
			torch_device: Union[int, str] = -1,
			precision=torch.bfloat16,
			choice_format='letter',
			enable_choice_search: bool = False,
	):
		super().__init__(model_name, prompt_name, prompt_func, choice_format, enable_choice_search)

		if isinstance(torch_device, str):
			torch_device = torch.device(torch_device)
		else:
			if torch_device == -1:
				torch_device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
			else:
				torch_device = torch.device(f"cuda:{torch_device}")

		if model is None:
			print(f"Loading {model_name}...")
			class_name, ckpt = videoqa_models[model_name]
			self.model_precision = precision
			self.model = eval(class_name)(ckpt, torch_device, self.model_precision)
			print(f"Finish loading {model_name}")
		else:
			print(f"Using provided self.model...")
			self.model = model

	@torch.no_grad()
	def _qa(self, data, prompt):
		if isinstance(data, str):
			return self.model.qa(data, prompt)
		else:
			with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as tmp:
				with open(tmp.name, 'wb') as file:
					file.write(data)
				video_path = tmp.name
				answer = self.model.qa(video_path, prompt)
			return answer


def sample_frames(video_path, n):
	import cv2
	# Open the video file
	cap = cv2.VideoCapture(video_path)
	if not cap.isOpened():
		print("Error: Could not open video.")
		return []

	# Calculate total number of frames and video FPS
	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

	# Calculate interval in terms of frames
	interval = max(1, total_frames // n)

	# Sample frames
	sampled_frames = []
	for i in range(0, total_frames, interval):
		# Set the current frame position
		cap.set(cv2.CAP_PROP_POS_FRAMES, i)

		# Read the frame
		ret, frame = cap.read()
		if not ret:
			print(f"Error: Could not read frame {i}.")
			break

		# Convert the frame to PIL Image
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		pil_img = Image.fromarray(frame_rgb)
		sampled_frames.append(pil_img)

		# Stop if we have collected n frames
		if len(sampled_frames) >= n:
			break

	# Release the video capture object
	cap.release()

	return sampled_frames


def get_contrasting_color(image, x, y, width, height):
	"""
	Determine a contrasting color (black or white) based on the average color of a specified area in the image.
	"""
	# Crop the relevant part of the image
	cropped_image = image.crop((x, y, x + width, y + height))
	# Convert to numpy array for analysis
	np_image = np.array(cropped_image)
	# Calculate the average color
	average_color = np.mean(np_image, axis=(0, 1))
	# Brightness calculation based on perceived luminance
	brightness = np.sqrt(0.299 * average_color[0] ** 2 + 0.587 * average_color[1] ** 2 + 0.114 * average_color[2] ** 2)
	# Return white for dark backgrounds and black for light backgrounds
	return 'white' if brightness < 128 else 'black'


def concatenate_image(images, rows, columns, separator_width=10):
	# Ensure we have the exact number of images needed
	if len(images) != rows * columns:
		raise ValueError(f"Expected {rows * columns} images, but got {len(images)}.")

	# Calculate the max width and height of images to standardize sizes
	max_width = max(img.width for img in images)
	max_height = max(img.height for img in images)

	# Resize images to the max width and height
	resized_images = [img.resize((max_width, max_height), Image.Resampling.LANCZOS) for img in images]

	# Calculate the total width and height for the combined image
	total_width = max_width * columns + separator_width * (columns - 1)
	total_height = max_height * rows + separator_width * (rows - 1)
	combined_image = Image.new('RGB', (total_width, total_height), color='white')

	# Place images in the specified grid
	x_offset = 0
	y_offset = 0
	for i, img in enumerate(resized_images):
		combined_image.paste(img, (x_offset, y_offset))
		if (i + 1) % columns == 0:  # Move to the next row after the last column
			x_offset = 0
			y_offset += img.height + separator_width
		else:  # Move to the next column
			x_offset += img.width + separator_width

	# Add numbers to each image for identification
	draw = ImageDraw.Draw(combined_image)
	try:
		font_size = (max_width + max_height) // 2 // 12
		font = ImageFont.load_default(size=font_size)
	except IOError:
		font = ImageFont.truetype("arial", 20)

	x_offset = 0
	y_offset = 0
	for i, img in enumerate(resized_images):
		text = str(i + 1)
		text_x = x_offset + 10
		text_y = y_offset + 10
		text_width, text_height = font_size, font_size
		font_color = get_contrasting_color(combined_image, text_x, text_y, text_width, text_height)
		draw.text((text_x, text_y), text, fill=font_color, font=font)
		if (i + 1) % columns == 0:
			x_offset = 0
			y_offset += img.height + separator_width
		else:
			x_offset += img.width + separator_width

	return combined_image


def video_to_concat_image(video_path, num_rows, num_columns):
	return concatenate_image(sample_frames(video_path, num_rows * num_columns), num_rows, num_columns)


class ImageQAModel4Video(VideoQAModel):
	def __init__(
			self,
			model: ImageQAModel,
			prompt_name: str,
			prompt_func: Callable,
			num_rows: int = 2,
			num_columns: int = 2,
			choice_format='letter',
			enable_choice_search: bool = False,
	):
		super(VideoQAModel, self).__init__(model.model_name, prompt_name, prompt_func, choice_format, enable_choice_search)
		self.num_rows = num_rows
		self.num_columns = num_columns
		self.num_frames = self.num_rows * self.num_columns
		self.model = model

	@torch.no_grad()
	def _qa(self, data, prompt):
		if isinstance(data, Image.Image):
			return self.model._qa(data, prompt)
		elif isinstance(data, str):
			return self.model._qa(video_to_concat_image(data, self.num_rows, self.num_columns), prompt)
		else:
			with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as tmp:
				with open(tmp.name, 'wb') as file:
					file.write(data)
				video_path = tmp.name
				answer = self.model._qa(video_to_concat_image(video_path, self.num_rows, self.num_columns), prompt)
			return answer

class VideoLLaVA(QAModelInstance):
	def __init__(self, ckpt='LanguageBind/Video-LLaVA-7B', torch_device=torch.device("cuda"), model_precision=torch.float32):
		# Environment setup# Disable certain initializations if necessary

		from .videoqa_model_library.Video_LLaVA.videollava.utils import disable_torch_init
		from .videoqa_model_library.Video_LLaVA.videollava import constants
		from .videoqa_model_library.Video_LLaVA.videollava.conversation import conv_templates, SeparatorStyle
		from .videoqa_model_library.Video_LLaVA.videollava.model.builder import load_pretrained_model
		from .videoqa_model_library.Video_LLaVA.videollava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

		self.constants = constants
		self.SeparatorStyle = SeparatorStyle
		self.tokenizer_image_token = tokenizer_image_token
		self.KeywordsStoppingCriteria = KeywordsStoppingCriteria
		self.conv_templates = conv_templates

		disable_torch_init()
		cache_dir = "cache_dir"

		self.device = torch_device
		model_name = get_model_name_from_path(ckpt)
		self.tokenizer, self.model, processor, _ = load_pretrained_model(ckpt, None, model_name, device=torch_device, cache_dir=cache_dir)
		self.video_processor = processor['video']

	def qa(self, video_path, question):
		conv_mode = "llava_v1"
		conv = self.conv_templates[conv_mode].copy()

		video_tensor = self.video_processor(video_path, return_tensors='pt')['pixel_values']
		if isinstance(video_tensor, list):
			tensor = [video.to(self.device, dtype=torch.float16) for video in video_tensor]
		else:
			tensor = video_tensor.to(self.device, dtype=torch.float16)

		question = ' '.join([self.constants.DEFAULT_IMAGE_TOKEN] * self.model.get_video_tower().config.num_frames) + '\n' + question
		conv.append_message(conv.roles[0], question)
		conv.append_message(conv.roles[1], None)

		# conv.append(question)
		prompt = conv.get_prompt()

		input_ids = self.tokenizer_image_token(prompt, self.tokenizer, self.constants.IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
		stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
		keywords = [stop_str]
		stopping_criteria = self.KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

		with torch.inference_mode():
			output_ids = self.model.generate(
				input_ids,
				images=tensor,
				do_sample=True,
				temperature=0.1,
				max_new_tokens=1024,
				use_cache=True,
				stopping_criteria=[stopping_criteria])

		outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
		if outputs.endswith(stop_str):
			outputs = outputs[:-len(stop_str)].strip()
		return outputs


class ChatUniVi(QAModelInstance):
	def __init__(self, ckpt='chat-univi-7b', torch_device=torch.device("cuda"), model_precision=torch.float32):
		# Environment setup# Disable certain initializations if necessary

		from .videoqa_model_library.Chat_UniVi.ChatUniVi import constants
		from .videoqa_model_library.Chat_UniVi.ChatUniVi.conversation import conv_templates, SeparatorStyle
		from .videoqa_model_library.Chat_UniVi.ChatUniVi.model.builder import load_pretrained_model
		from .videoqa_model_library.Chat_UniVi.ChatUniVi.utils import disable_torch_init
		from .videoqa_model_library.Chat_UniVi.ChatUniVi.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria

		from decord import VideoReader, cpu

		self.constants = constants
		self.cpu = cpu
		self.VideoReader = VideoReader
		self.conv_templates = conv_templates
		self.tokenizer_image_token = tokenizer_image_token
		self.KeywordsStoppingCriteria = KeywordsStoppingCriteria
		self.SeparatorStyle = SeparatorStyle

		disable_torch_init()
		if ckpt == 'chat-univi-7b':
			model_path = "Chat-UniVi/Chat-UniVi"
		elif ckpt == 'chat-univi-13b':
			model_path = "Chat-UniVi/Chat-UniVi-13B"
		model_name = "ChatUniVi"
		self.tokenizer, self.model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

		mm_use_im_start_end = getattr(self.model.config, "mm_use_im_start_end", False)
		mm_use_im_patch_token = getattr(self.model.config, "mm_use_im_patch_token", True)
		if mm_use_im_patch_token:
			self.tokenizer.add_tokens([self.constants.DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
		if mm_use_im_start_end:
			self.tokenizer.add_tokens([self.constants.DEFAULT_IM_START_TOKEN, self.constants.DEFAULT_IM_END_TOKEN], special_tokens=True)
		self.model.resize_token_embeddings(len(self.tokenizer))

		vision_tower = self.model.get_vision_tower()
		if not vision_tower.is_loaded:
			vision_tower.load_model()
		self.image_processor = vision_tower.image_processor

		if self.model.config.config["use_cluster"]:
			for n, m in self.model.named_modules():
				m = m.to(dtype=torch.bfloat16)

	def qa(self, video_path, question):
		# setting parameters
		max_frames = 16
		# The number of frames retained per second in the video.
		video_framerate = 4
		# Input Text
		qs = question

		# Sampling Parameter
		conv_mode = "simple"
		temperature = 0.2
		top_p = None
		num_beams = 1

		if video_path is not None:
			video_frames, slice_len = self._get_rawvideo_dec(video_path, self.image_processor, max_frames=max_frames, video_framerate=video_framerate)

			if self.model.config.mm_use_im_start_end:
				qs = self.constants.DEFAULT_IM_START_TOKEN + self.constants.DEFAULT_IMAGE_TOKEN * slice_len + self.constants.DEFAULT_IM_END_TOKEN + '\n' + qs
			else:
				qs = self.constants.DEFAULT_IMAGE_TOKEN * slice_len + '\n' + qs

			conv = self.conv_templates[conv_mode].copy()
			conv.append_message(conv.roles[0], qs)
			conv.append_message(conv.roles[1], None)
			prompt = conv.get_prompt()

			input_ids = self.tokenizer_image_token(prompt, self.tokenizer, self.constants.IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
				0).cuda()

			stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
			keywords = [stop_str]
			stopping_criteria = self.KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

			with torch.inference_mode():
				output_ids = self.model.generate(
					input_ids,
					images=video_frames.half().cuda(),
					do_sample=True,
					temperature=temperature,
					top_p=top_p,
					num_beams=num_beams,
					output_scores=True,
					return_dict_in_generate=True,
					max_new_tokens=1024,
					use_cache=True,
					stopping_criteria=[stopping_criteria])

			output_ids = output_ids.sequences
			input_token_len = input_ids.shape[1]
			n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
			if n_diff_input_output > 0:
				print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
			outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
			outputs = outputs.strip()
			if outputs.endswith(stop_str):
				outputs = outputs[:-len(stop_str)]
			outputs = outputs.strip()
			return outputs

	def _get_rawvideo_dec(self, video_path, image_processor, max_frames=None, image_resolution=224, video_framerate=1, s=None, e=None):
		# speed up video decode via decord.
		if max_frames is None:
			max_frames = 100
		if s is None:
			start_time, end_time = None, None
		else:
			start_time = int(s)
			end_time = int(e)
			start_time = start_time if start_time >= 0. else 0.
			end_time = end_time if end_time >= 0. else 0.
			if start_time > end_time:
				start_time, end_time = end_time, start_time
			elif start_time == end_time:
				end_time = start_time + 1

		if os.path.exists(video_path):
			vreader = self.VideoReader(video_path, ctx=self.cpu(0))
		else:
			print(video_path)
			raise FileNotFoundError

		fps = vreader.get_avg_fps()
		f_start = 0 if start_time is None else int(start_time * fps)
		f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1))
		num_frames = f_end - f_start + 1
		if num_frames > 0:
			# T x 3 x H x W
			sample_fps = int(video_framerate)
			t_stride = int(round(float(fps) / sample_fps))

			all_pos = list(range(f_start, f_end + 1, t_stride))
			if len(all_pos) > max_frames:
				sample_pos = [all_pos[_] for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)]
			else:
				sample_pos = all_pos

			batch = vreader.get_batch(sample_pos)
			if hasattr(batch, 'asnumpy'):
				batch_np = batch.asnumpy()
			elif hasattr(batch, 'numpy'):
				batch_np = batch.numpy()
			else:
				raise TypeError("The object does not have asnumpy or numpy methods.")
			patch_images = [Image.fromarray(f) for f in batch_np]
   
			patch_images = torch.stack([image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in patch_images])
			slice_len = patch_images.shape[0]

			return patch_images, slice_len
		else:
			print("video path: {} error.".format(video_path))


class VideoChatGPT(QAModelInstance):
	def __init__(self, ckpt, torch_device=torch.device("cuda"), model_precision=torch.float32):

		from .videoqa_model_library.Video_ChatGPT.video_chatgpt.video_conversation import conv_templates, SeparatorStyle
		from .videoqa_model_library.Video_ChatGPT.video_chatgpt.model.utils import KeywordsStoppingCriteria
		from .videoqa_model_library.Video_ChatGPT.video_chatgpt.eval.model_utils import initialize_model, load_video

		model_weights_path = huggingface_hub.snapshot_download(repo_id="weikaih/VideoChatGPT")
		model_name = os.path.join(model_weights_path, 'LLaVA-Lightning-7B-v1-1')
		projection_path = os.path.join(model_weights_path, 'video_chatgpt-7B.bin')
		self.model, self.vision_tower, self.tokenizer, self.image_processor, self.video_token_len = initialize_model(model_name, projection_path)

		self.conv_templates = conv_templates
		self.load_video = load_video
		self.KeywordsStoppingCriteria = KeywordsStoppingCriteria
		self.SeparatorStyle = SeparatorStyle

	def qa(self, video_path, question):

		video_frames = self.load_video(video_path)

		DEFAULT_VIDEO_TOKEN = "<video>"
		DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
		DEFAULT_VID_START_TOKEN = "<vid_start>"
		DEFAULT_VID_END_TOKEN = "<vid_end>"

		if self.model.get_model().vision_config.use_vid_start_end:
			qs = question + '\n' + DEFAULT_VID_START_TOKEN + DEFAULT_VIDEO_PATCH_TOKEN * self.video_token_len + DEFAULT_VID_END_TOKEN
		else:
			qs = question + '\n' + DEFAULT_VIDEO_PATCH_TOKEN * self.video_token_len

		conv_mode = 'video-chatgpt_v1'
		conv = self.conv_templates[conv_mode].copy()
		conv.append_message(conv.roles[0], qs)
		conv.append_message(conv.roles[1], None)
		prompt = conv.get_prompt()

		# Tokenize the prompt
		inputs = self.tokenizer([prompt])

		# Preprocess video frames and get image tensor
		image_tensor = self.image_processor.preprocess(video_frames, return_tensors='pt')['pixel_values']

		# Move image tensor to GPU and reduce precision to half
		image_tensor = image_tensor.half().cuda()

		# Generate video spatio-temporal features
		with torch.no_grad():
			image_forward_outs = self.vision_tower(image_tensor, output_hidden_states=True)
			frame_features = image_forward_outs.hidden_states[-2][:, 1:]  # Use second to last layer as in LLaVA
		video_spatio_temporal_features = self.get_spatio_temporal_features_torch(frame_features)

		# Move inputs to GPU
		input_ids = torch.as_tensor(inputs.input_ids).cuda()

		# Define stopping criteria for generation
		stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
		stopping_criteria = self.KeywordsStoppingCriteria([stop_str], self.tokenizer, input_ids)

		# Run model inference
		with torch.inference_mode():
			output_ids = self.model.generate(
				input_ids,
				video_spatio_temporal_features=video_spatio_temporal_features.unsqueeze(0),
				do_sample=True,
				temperature=0.2,
				max_new_tokens=1024,
				stopping_criteria=[stopping_criteria])

		# Check if output is the same as input
		n_diff_input_output = (input_ids != output_ids[:, :input_ids.shape[1]]).sum().item()
		if n_diff_input_output > 0:
			print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')

		# Decode output tokens
		outputs = self.tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

		# Clean output string
		outputs = outputs.strip().rstrip(stop_str).strip()

		return outputs

	def get_spatio_temporal_features_torch(self, features):

		# Extract the dimensions of the features
		t, s, c = features.shape

		# Compute temporal tokens as the mean along the time axis
		temporal_tokens = torch.mean(features, dim=1)

		# Padding size calculation
		padding_size = 100 - t

		# Pad temporal tokens if necessary
		if padding_size > 0:
			padding = torch.zeros(padding_size, c, device=features.device)
			temporal_tokens = torch.cat((temporal_tokens, padding), dim=0)

		# Compute spatial tokens as the mean along the spatial axis
		spatial_tokens = torch.mean(features, dim=0)

		# Concatenate temporal and spatial tokens and cast to half precision
		concat_tokens = torch.cat([temporal_tokens, spatial_tokens], dim=0).half()

		return concat_tokens


class VideoLLaMA2(QAModelInstance):
	def __init__(self, ckpt='video-llama2-7b', torch_device=torch.device("cuda"), model_precision=torch.float32):
		from .videoqa_model_library.Video_LLaMA.video_llama import inference

		if ckpt == 'video-llama2-7b':
			model_weights_path = huggingface_hub.snapshot_download(repo_id="DAMO-NLP-SG/Video-LLaMA-2-7B-Finetuned")
			llama_model_path = os.path.join(model_weights_path, "llama-2-7b-chat-hf")
			vl_model_path = os.path.join(model_weights_path, "VL_LLaMA_2_7B_Finetuned.pth")

		elif ckpt == 'video-llama2-13b':
			model_weights_path = huggingface_hub.snapshot_download(repo_id="DAMO-NLP-SG/Video-LLaMA-2-13B-Finetuned")
			llama_model_path = os.path.join(model_weights_path, "llama-2-13b-chat-hf")
			vl_model_path = os.path.join(model_weights_path, "VL_LLaMA_2_13B_Finetuned.pth")

		# we modify the path
		self.chatbot = inference.ChatBot(cfg_path="ignore.yaml", llama_model_path=llama_model_path, vl_model_path=vl_model_path, model_type='llama_v2', torch_device=torch_device)

	def qa(self, video_path, question):
		self.chatbot.upload(up_video=video_path, audio_flag=False)
		llm_message = self.chatbot.ask_answer(user_message=question)
		self.chatbot.reset()
		return llm_message


class VideoChat2(QAModelInstance):
	def __init__(self, ckpt='video-chat2-7b', torch_device=torch.device("cuda"), model_precision=torch.float32):
		from .videoqa_model_library.Video_Chat.video_chat2 import inference

		# we modify the path
		model_weights_path = huggingface_hub.snapshot_download(repo_id="weikaih/VideoChat2")
		llama_model_path = os.path.join(model_weights_path, "vicuna-7b-v0")
		vit_blip_path = os.path.join(model_weights_path, "umt_l16_qformer.pth")
		videochat2_model_stage2_path = os.path.join(model_weights_path, "videochat2_7b_stage2.pth")
		videochat2_model_stage3_path = os.path.join(model_weights_path, "videochat2_7b_stage3.pth")
		self.chatbot = inference.ChatBot(llama_model_path, vit_blip_path, videochat2_model_stage2_path, videochat2_model_stage3_path)
		self.num_frames = 16

	def qa(self, video_path, question):
		self.chatbot.upload(video_path, num_frames=self.num_frames)
		llm_message = self.chatbot.ask_answer(user_message=question)
		self.chatbot.reset()
		return llm_message