# TaskMeAnything

## 🔔News

 **🔥[2024-06-01]: Code released!**

## What's TaskMeAnything?

## TaskMeAnything-v1

### Source data
Source data are stored in [HuggingFace](https://huggingface.co/datasets/jieyuz2/TaskMeAnything-v1-source)

### Demo

### TaskMeAnything-v1-Random

#### Load TaskMeAnything-v1-Random ImageQA Dataset
```
import datasets

dataset_name = 'TaskMeAnything-v1-random-imageqa'
dataset = datasets.load_dataset(dataset_name, split = TASK_GENERATOR_SPLIT)
```
where `TASK_GENERATOR_SPLIT` is one of the task generators, eg, `2d_how_many`.

#### Load TaskMeAnything-v1-Random VideoQA Dataset and Convert Video Binary Stream to mp4
* Since Huggingface does not support saving .mp4 files in datasets, we save videos in the format of binary streams. After loading, you can convert the video binary stream to .mp4 using the following method.
```
import datasets

dataset_name = 'TaskMeAnything-v1-random-videoqa'
dataset = datasets.load_dataset(dataset_name, split = TASK_GENERATOR_SPLIT)

# example: convert binary stream in dataset to .mp4 files
video_binary = dataset[0]['video']
with open('/path/save/video.mp4', 'wb') as f:
    f.write(video_binary)
```

### TaskMeAnything-DB

**TaskMeAnything-DB** are stored in [HuggingFace](https://huggingface.co/datasets/jieyuz2/TaskMeAnything-v1-db)

### TaskMeAnything-UI

## Disclaimers

## Contact

- Jieyu Zhang: jieyuz2@cs.washington.edu

## Citation

**BibTeX:**

```bibtex

```
