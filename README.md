# TaskMeAnything: VideoQA Branch
This branch is for evaluating videoqa models that are not support by huggingface. For imageqa models, check out the [main branch](https://github.com/JieyuZ2/TaskMeAnything)

### Intro
Since current video models are not well support by huggingface, we need to import these video models as submodules of TaskMeAnything model. To keep the codebase of main branch clear, we use this video_models branch to support evaluating on video models. 

### Installation
First, pip the requirements for videoqa models (python3.10): 
```bash
pip install -r requirements-video.txt
```
Then exec these command to add the videoqa models as submodules:
```bash
git submodule add https://github.com/weikaih04/Video-LLaVA.git tma/models/qa_model/videoqa_model_library/Video_LLaVA

git submodule add https://github.com/weikaih04/Video-LLaMA.git tma/models/qa_model/videoqa_model_library/Video_LLaMA

git submodule add https://github.com/weikaih04/Video-ChatGPT.git tma/models/qa_model/videoqa_model_library/Video_ChatGPT

git submodule add https://github.com/weikaih04/Ask-Anything.git tma/models/qa_model/videoqa_model_library/Video_Chat

git submodule add https://github.com/weikaih04/Chat-UniVi.git tma/models/qa_model/videoqa_model_library/Chat_UniVi
```

Notice: Please use the TMA's fork model repository (e.g., /weikaih04/Video-LLaVA) instead of the original model repository. This is necessary because we need to modify the original model code to adapt to our codebase and ensure that each model processes the same `16 frames` of video for benchmarking purposes.


### Usage
You can evaluate videoqa models using exactly the same method in notebooks in `demo`.

If you wanna use our videoqa models for inference. You can use the following code snippet:
```python
from tma.models.qa_model import VideoQAModel
# from tma.models.qa_model.prompt import succinct_prompt
from tma.models.qa_model.prompt import detailed_videoqa_prompt


model = VideoQAModel(
    model_name= "video-llava-7b",
    prompt_name= "detailed",
    prompt_func= detailed_videoqa_prompt
)

video = './path/to/video.mp4'
question = "Describe the video."

model.qa(video, question)
```

