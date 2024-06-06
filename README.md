# TaskMeAnything




<p align="center">
    <img src="jy.png" width="150" style="margin-bottom: 0.2;"/>
<p>
<h2 align="center"> <a href="">Task Me Anything:</a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for latest update.  </h2>


<!-- **🌐 Website**]()  -->
<h5 align="center">

<!-- [**🤗 Graphical Interface**]() | [**🤗 1.0 Random Dataset**]() | [**📖 Paper**]() -->

</h5>


## 🔔News

 **🔥[2024-06-01]: Code released!**

## What's TaskMeAnything?
TaskMeAnything is a scalable and controlable programmatic benchmark generation engine designed to generate tailored ImageQA and VideoQA questions based on user query (e.g Which Model is better at material recognition: GPT4o or LLaVA-Next?), and also provide query approximation algorithms to estimate the query results under certain budgets. (e.g. Know the better model over 5 millions questions related to material recognition within 500 inferences per models.)


Current we release the following resources: 
1. [**TaskMeAnything-v1**](todo): the first version of TaskMeAnything, includes 28 difference task generators which can generate over 5 millions task instances.
2. [**TaskMeAnything-v1-Random**](todo): A randomly selected from TaskMeAnything-v1, including 5,700 ImageQA and 1,800 VideoQA questions.
3. [**TaskMeAnything-DB**](todo): A database for TaskMeAnything, which stores the results of query, releases for developing more effective and efficient query approximation algorithms.
4. [**TaskMeAnything-UI**](todo): An interactive graphical interface for TaskMeAnything 1.0, which allows users to interact with the performance of models on TaskMeAnything1.0 in a intuitve way.

   

## TaskMeAnything-v1

### Installation
You can easily download the repo and setup the environments via:
```
git clone https://github.com/JieyuZ2/TaskMeAnything.git
cd ./TaskMeAnything

# if you want to test ImageQA models.
pip install -r requirements.txt
# if you want to test VideoQA models.
pip install -r requirements-video.txt
```
Notice: if you wanna render 3D images, videos by `Blender` locally or use `Internvl-chat-v1.5-24B` that required `flash-attn` which hard to install by pip, you can use the docker image we provide. You can pull the docker image from DockerHub which includes all the dependencies like `Blender`, `flash-attn`, `cuda driver`, `nvcc`, etc.
```
docker push weikaih/ubuntu20.4_internvl_blender_v1.2:latest
docker run --gpus all -it weikaih/ubuntu20.4_internvl_blender_v1.2:latest /bin/bash # run the docker image with GPU support

git clone https://github.com/JieyuZ2/TaskMeAnything.git
cd ./TaskMeAnything

# if you want to test ImageQA models.
pip install -r requirements.txt
# if you want to test VideoQA models.
pip install -r requirements-video.txt
```
### Source data
Source data is stored in [HuggingFace](https://huggingface.co/datasets/jieyuz2/TaskMeAnything-v1-source). It includes `3d_assets`, `agqa_video`, and `object_images`.

For real image scenarios, please download the images and scene graphs from the following links: [SceneGraph](https://downloads.cs.stanford.edu/nlp/data/gqa/sceneGraphs.zip), [Image](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip). After downloading, move the scene graphs and images into the source data folder, and arrange them as format below.
```
/TaskMeAnything-v1-source/vg/sceneGraphs: move scene graphs files to this folder (e.g. /TaskMeAnything-v1-source/vg/sceneGraphs/train_sceneGraphs.json).
/TaskMeAnything-v1-source/vg/images/images: move all the images to this folder (e.g. /TaskMeAnything-v1-source/vg/images/images/2323739.jpg).
```


### Task Generator
Currently we have 28 task generators in TaskMeAnything-v1, across 5 Scenarios:
1. `2D Sticker Image`: grid-how-many, grid-what, grid-where, grid-what-attribute, grid-where-attribute
2. `3D Tabletop Image`: 3d-what, 3d-where, 3d-what-attribute, 3d-where-attribute, 3d-how-many, 3d-what-size, 3d-where-size, 3d-what-attribute-size, 3d-what-distance, 3d-where-distance, 3d-what-attribute-distance
3. `3D Tabletop Video`: video-3d-what-move, video-3d-where-move, video-3d-what-attribute-move, video-3d-what-rotate, video-3d-where-rotate, video-3d-what-attribute-rotate
4. `Real Images`: sg-what-object, sg-what-relation, sg-what-attribute
5. `Real Videos`: video-sg-what-object, video-sg-what-relation, video-sg-what-action

### Tested Models 
We support the following ImageQA and VideoQA models: 
- `ImageQA`: qwenvl-chat, qwenvl, llavav1.5-7b, llavav1.5-13b, instructblip-vicuna7b, instructblip-vicuna13b, internvl-chat-v1.5, gemini-vision-pro, qwen-vl-max, gpt4v, gpt4o
- `VideoQA`: video-llama2-7b, video-llama2-13b, video-llava-7b, chat-univi-7b, chat-univi-13b, video-chatgpt-7b, video-chat2-7b
### Demo
Stay tuned! We will release the demo before June, 2024.

## TaskMeAnything-v1-Random
* **[TaskMeAnything-v1-imageqa-random](https://huggingface.co/datasets/weikaih/TaskMeAnything-v1-imageqa-random) is a dataset randomly selected from TaskMeAnything-v1, including 5,700 ImageQA. (more details)**
* **[TaskMeAnything-v1-videoqa-random](https://huggingface.co/datasets/weikaih/TaskMeAnything-v1-videoqa-random) is a dataset randomly selected from TaskMeAnything-v1, including 1,800 VideoQA questions. (more details)**

### Load TaskMeAnything-v1-Random ImageQA Dataset
```
import datasets

dataset_name = 'weikaih/TaskMeAnything-v1-imageqa-random'
dataset = datasets.load_dataset(dataset_name, split = TASK_GENERATOR_SPLIT)
```
where `TASK_GENERATOR_SPLIT` is one of the task generators, eg, `2d_how_many`.


### Load TaskMeAnything-v1-Random VideoQA Dataset and Convert Video Binary Stream to mp4
* Since Huggingface does not support saving .mp4 files in datasets, we save videos in the format of binary streams. After loading, you can convert the video binary stream to .mp4 using the following method.
```
import datasets

dataset_name = 'weikaih/TaskMeAnything-v1-videoqa-random'
dataset = datasets.load_dataset(dataset_name, split = TASK_GENERATOR_SPLIT)

# example: convert binary stream in dataset to .mp4 files
video_binary = dataset[0]['video']
with open('/path/save/video.mp4', 'wb') as f:
    f.write(video_binary)
```

## TaskMeAnything-DB
**TaskMeAnything-DB** are stored in [HuggingFace](https://huggingface.co/datasets/jieyuz2/TaskMeAnything-v1-db)

## TaskMeAnything-UI
**TaskMeAnything-UI** are hosted in [HuggingFace](todo), check out our interactive interface to explore the performance of models on TaskMeAnything1.0 in your own way!

## Disclaimers
**TaskMeAnything** and its associated resources are provided for research and educational purposes only. The authors and contributors make no warranties regarding the accuracy or reliability of the data and software. Users are responsible for ensuring their use complies with applicable laws and regulations. The project is not liable for any damages or losses resulting from the use of these resources.


## Contact

- Jieyu Zhang: jieyuz2@cs.washington.edu

## Citation

**BibTeX:**

```bibtex

```

