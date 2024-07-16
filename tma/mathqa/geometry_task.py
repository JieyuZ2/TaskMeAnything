import os
import sys

from typing import Dict, List, Tuple

import numpy as np 
from tqdm import tqdm

from baseMath import TaskGenerator
from task_storeMath import TaskStore
from utils import *
import json
import enum
from metadataMath import MathTemplateMetaData
# PERI_SIDE_ONE_RANGE, PERI_SIDE_TWO_RANGE, PERI_SIDE_THREE_RANGE, 
#     make_single_prod, make_pair_prod, make_triplet_prod

class Template(enum.Enum):
    PERIMETER_TEMPLATES = 'MathVerse/math_anotations/perimeter_templates.json'
    MIDPOINT_TEMPLATES = 'annotations/math_annotations/midpoint.json'

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

    
class MidpointGenerator(GeoPlanGenerator):
    X_COORDINATE_RANGE = (0, 100)
    Y_COORDINATE_RANGE = (0, 100)
    Z_COORDINATE_RANGE = (0, 100)

    schema = {
        'question_template': 'str',
        'point1': 'str',
        'point2': 'str'
    }

    def __init__(self, metadata: MathTemplateMetaData, seed=42, x_range=X_COORDINATE_RANGE, y_range=Y_COORDINATE_RANGE, z_range=Z_COORDINATE_RANGE, num_splices=5):
        super().__init__(metadata, seed=seed)
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.num_splices = num_splices



    def enumerate_task_plans(self, task_store: TaskStore):
        x_values = np.linspace(self.x_range[0], self.x_range[1], self.num_splices)
        y_values = np.linspace(self.y_range[0], self.y_range[1], self.num_splices)
        z_values = np.linspace(self.z_range[0], self.z_range[1], self.num_splices)

        template_breakdown = self.metadata.templates_by_num_params

        for templates in template_breakdown.items():
            for template in tqdm(templates, desc="Enumerating templates for midpoint tasks"):
                for x1 in x_values:
                    for y1 in y_values:
                        for z1 in z_values:
                            for x2 in x_values:
                                for y2 in y_values:
                                    for z2 in z_values:
                                        task_plan = {
                                            'question_template': template,
                                            'point1': (x1, y1, z1),
                                            'point2': (x2, y2, z2)
                                        }
                                        task_store.add(task_plan)
                
    def _generate_task(self, task_plan) -> Tuple[str, str, List[str], Dict]:
        template = task_plan['question_template']
        point1 = eval(task_plan['point1'])
        point2 = eval(task_plan['point2'])

        midpoint = "({}, {}, {})".format(
            (point1[0] + point2[0]) / 2,
            (point1[1] + point2[1]) / 2,
            (point1[2] + point2[2]) / 2
        )
        
        question = template.format(
            param1 = point1,
            param2 = point2)
        answer = str(midpoint)
        
        return question, answer, self.metadata
    


class IntersectionGenerator(GeoPlanGenerator):
    X_COORDINATE_RANGE = (0, 10) 
    Y_COORDINATE_RANGE = (0, 10)

    schema = {
        'question_template': 'str',
        'vector1_start': 'str',
        'vector1_end': 'str',
        'vector2_start': 'str',
        'vector2_end': 'str'
    }

    def __init__(self, metadata: MathTemplateMetaData, seed=42, x_range=X_COORDINATE_RANGE, y_range=Y_COORDINATE_RANGE, num_splices=4):
        super().__init__(metadata, seed=seed)
        self.x_range = x_range
        self.y_range = y_range
        self.num_splices = num_splices

    def enumerate_task_plans(self, task_store: TaskStore):
        x_values = np.linspace(self.x_range[0], self.x_range[1], self.num_splices)
        y_values = np.linspace(self.y_range[0], self.y_range[1], self.num_splices)

        template_breakdown = self.metadata.templates_by_num_params

        for param_count, templates in template_breakdown.items():
            for template in tqdm(templates, desc="Enumerating templates for intersection tasks"):
                for x1 in x_values:
                    for y1 in y_values:
                        for x2 in x_values:
                            for y2 in y_values:
                                for x3 in x_values:
                                    for y3 in y_values:
                                        for x4 in x_values:
                                            for y4 in y_values:
                                                task_plan = {
                                                    'question_template': template,
                                                    'vector1_start': str((x1, y1)),
                                                    'vector1_end': str((x2, y2)),
                                                    'vector2_start': str((x3, y3)),
                                                    'vector2_end': str((x4, y4))
                                                }
                                                task_store.add(task_plan)

    def _generate_task(self, task_plan) -> Tuple[str, str, List[str], Dict]:
        template = task_plan['question_template']
        vector1_start = eval(task_plan['vector1_start'])
        vector1_end = eval(task_plan['vector1_end'])
        vector2_start = eval(task_plan['vector2_start'])
        vector2_end = eval(task_plan['vector2_end'])

        vector1 = np.array(vector1_end) - np.array(vector1_start)
        vector2 = np.array(vector2_end) - np.array(vector2_start)

        if np.cross(vector1, vector2) == 0:
            answer = "null"
        else:
            # Compute intersection
            A = np.array([[vector1[0], -vector2[0]], [vector1[1], -vector2[1]]])
            b = np.array([vector2_start[0] - vector1_start[0], vector2_start[1] - vector1_start[1]])
            t = np.linalg.solve(A, b)
            intersection = vector1_start + t[0] * vector1
            answer = str(tuple(intersection))

        question = template.format(
            param1=vector1_start,
            param2=vector1_end,
            param3=vector2_start,
            param4=vector2_end
        )
        
        return question, answer, self.metadata



class CircleGenerator(GeoPlanGenerator):
    schema = {
        'question_template': 'str',
        'radius': 'str',          # Store float as string
        'circumference': 'str',   # Store float as string
        'area': 'str'
    }

    def __init__(self, metadata: MathTemplateMetaData, seed=42, radius_range=(1, 100), num_tasks=100):
        super().__init__(metadata, seed=seed)
        self.radius_range = radius_range
        self.num_tasks = num_tasks
        self.all_tasks = set()

    def _task_plan_to_str(self, task_plan) -> str:
        return json.dumps(task_plan)

    def enumerate_task_plans(self, task_store: TaskStore):
        # This method generates all possible tasks in this range
        all_radius = np.linspace(self.radius_range[0], self.radius_range[1], self.num_tasks)
        template_breakdown = self.metadata.templates_by_num_params
        for num_paras, texts in template_breakdown.items():
            for text in texts:
                for radius in tqdm(all_radius, desc=f"Enumerating templates with {num_paras} params"):
                    circumference = 2 * np.pi * radius
                    area = np.pi * radius**2
                    result = (radius, circumference, area)
                    if result not in self.all_tasks:
                        task_plan = {
                            'question_template': text,
                            'radius': str(radius),
                            'circumference': str(circumference),
                            'area': str(area)
                        }
                        task_store.add(task_plan)
                        self.all_tasks.add(result)

    def _generate_task(self, task_plan) -> Tuple[str, str, Dict]:
        # Generate one task at a time
        answer = None
        question = None
        template = task_plan['question_template']
        radius = float(task_plan['radius'])
        circumference = float(task_plan['circumference'])
        area = float(task_plan['area'])
        if 'circumference' in template or 'perimeter' in template:
            if 'radius' in template:
                question = template.format(param1=radius)
                answer = circumference
            else:
                question = template.format(param1=circumference)
                answer = radius
        elif 'area' in template:
            if 'radius' in template:
                question = template.format(param1=radius)
                answer = area
            else:
                question = template.format(param1=area)
                answer = radius
        else:
            raise ValueError("Template must specify either perimeter, area, or radius.")
        return question, str(answer), self.metadata


# continue working on the angle class
class AngleGenerator(GeoPlanGenerator):
    schema = {
        'question_template': 'str',
        'angles': 'list'  # Store list of angles
    }

    def __init__(self, metadata: MathTemplateMetaData, seed=42, num_tasks=100):
        super().__init__(metadata, seed=seed)
        self.num_tasks = num_tasks
        self.all_tasks = set()
    
    def _task_plan_to_str(self, task_plan) -> str:
        return json.dumps(task_plan)
    
    def enumerate_task_plans(self, task_store: TaskStore):
        # store tasks in the task store
        template_breakdown = self.metadata.templates_by_num_params
        shapes = {
            'triangle': [60, 60, 60],
            'rectangle': [90, 90, 90, 90],
            'pentagon': [108, 108, 108, 108, 108],
            'hexagon': [120, 120, 120, 120, 120, 120]
        }
        for num_params, templates in template_breakdown.items():
            for template_text in templates:
                for shape, angles in shapes.items():
                    if len(angles) == num_params:
                        task_plan = {
                            'question_template': template_text,
                            'angles': angles
                        }
                        task_store.add(task_plan)
                        self.all_tasks.add(tuple(angles))
        
    def _generate_task(self, task_plan) -> Tuple[str, str, Dict]:
        # generate the single task
        question = None
        answer = None

        template = task_plan['question_template']
        angles = task_plan['angles']

        if len(angles) == 3:
            answer = "triangle"
        elif len(angles) == 4:
            answer = "rectangle"
        elif len(angles) == 5:
            answer = "pentagon"
        elif len(angles) == 6:
            answer = "hexagon"
        else:
            raise ValueError("Unsupported shape")

        question = template.format(*angles)
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


