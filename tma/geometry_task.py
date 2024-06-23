from typing import Dict, List, Tuple

import numpy as np 
from tqdm import tqdm

from ..base import TaskGenerator
from ..task_store import TaskStore
from utils import *
import json
import enum
from metadata import MathTemplateMetaData
# PERI_SIDE_ONE_RANGE, PERI_SIDE_TWO_RANGE, PERI_SIDE_THREE_RANGE, 
#     make_single_prod, make_pair_prod, make_triplet_prod

class Template(enum):
    PERIMETER_TEMPLATES = 'MathVerse/math_anotations/perimeter_templates.json'

class GeoPlanGenerator(TaskGenerator):
    metadata: MathTemplateMetaData

    def __init__(self, metadata: MathTemplateMetaData, seed=42):
        super().__init__(metadata, seed=seed)
    
    def _task_plan_to_str(self, task_plan) -> str:
        "(Abstract method) task plan to string"

    def _generate_task(self, task_plan) -> Tuple[str, str, List[str], Dict]:
        "(Abstract method) generate task"
        # TODO: COME BACK IN FILL THIS IN AS NOT A ABSTRACT METHOD
        
    
    def generate(self, task_plan, return_data=True, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)

        question, answer, math_metadata = self._generate_task(task_plan)

        task = {
            'question'  : question.replace('_', ' '),
            'answer'    : answer.replace('_', ' '),
            'task_plan' : self._task_plan_to_str(task_plan),
            'math_metadata' : math_metadata,
            
        }

        return task

    
class PerimeterGenerator(GeoPlanGenerator):
    schema = {
        'question_template' : 'str',
        'side_one'          : 'float',
        'side_two'          : 'float',
        'side_three'        : 'float'
    }

    def __init__(self, metadata: MathTemplateMetaData, seed=42, side_one_range=PERI_SIDE_ONE_RANGE, side_two_range=PERI_SIDE_TWO_RANGE, side_three_range=PERI_SIDE_THREE_RANGE, num_splices=100):
        super.__init__(metadata, seed=seed)
        self.side_one_range = side_one_range
        self.side_two_range = side_two_range
        self.side_three_range = side_three_range
        self.int_to_peri_list = int_to_peri_list
        self.num_splices = num_splices

    def enumerate_task_plans(self, task_store: TaskStore):
        single = make_single_prod(self.side_one_range, self.num_splices)
        pairs = make_pair_prod(self.side_one_range, self.side_three_range, self.num_splices)
        triplets = make_triplet_prod(self.side_one_range, self.side_two_range, self.side_three_range, self.num_splices)
        
        template_breakdown = self.metadata.templates_by_num_params
        
        for param_count, templates in template_breakdown.items():
            peri_list = locals()[self.int_to_peri_list[param_count]]
            
            for template in tqdm(templates, desc=f"Enumerating templates with {param_count} params"):
                for group in peri_list:
                    params = [param if param is not None else None for param in group]
                    while len(params) < 3:
                        params.append(None)

                    task_plan = {
                        'question_template': template,
                        'side_one': params[0],
                        'side_two': params[1],
                        'side_three': params[2]
                    }
                    
                    task_store.add(task_plan)
                
            
    def _generate_task(self, task_plan) -> Tuple[str | List[str] | Dict]:
        question = None
        answer = None

        template = task_plan['question_template']
        side_one = task_plan['side_one']
        side_two = task_plan['side_two']
        side_three = task_plan['side_three']

        if side_two is None:
            question = template.format(side_one) # format is single param
            answer = 3 * side_one
            
        elif side_three is None:
            question = template.format(side_one, side_two) # format is double param
            answer = 2 * side_one + side_two
        
        else:
            question = template.format(side_one, side_two, side_three) # format is triple param
            answer = side_one + side_two + side_three
            

        return question, answer, self.metadata
    
# class AreaGenerator(GeoPlanGenerator):
#     schema = {

#     }

#     def __init__(self, seed=42):
#         super.__init__(seed=seed)
    
#     def enumerate_task_plans(self, task_store: TaskStore):
#         return super().enumerate_task_plans(task_store)
    
#     def _generate_task(self, task_plan) -> Tuple[str | List[str] | Dict]:
#         return super()._generate_task(task_plan)


