import json
import numpy
from ..metadata import MetaData

def handle_templates(template_path):
        # iterate through json file and seperate into different number of params
        templates_by_num_params = {}

        with open(template_path, 'r') as file:
            templates_data = json.load(file)

        # Iterate over each template
        for template_name, template_info in templates_data.items():
            num_params = template_info["num_params"]
            template_text = template_info["text"]

            # Add the template text to the corresponding list based on the number of parameters
            if num_params not in templates_by_num_params:
                templates_by_num_params[num_params] = []
            templates_by_num_params[num_params].append(template_text)

        return templates_by_num_params

class MathTemplateMetaData(MetaData):
    def __init__(self, path_to_metadata, template_path):
        super.__init__()
        self.templates_by_num_params = handle_templates(template_path=template_path)
        
    def get_params(self):
        if len(self.templates_by_num_params) == 0:
            raise AssertionError("The templates_by_num_params dictionary must contain at least one element.")
        
        return self.templates_by_num_params.keys
    
    def get_templates(self, param_count):
        if len(self.templates_by_num_params) == 0:
            raise AssertionError("The templates_by_num_params dictionary must contain at least one element.")
        
        return self.templates_by_num_params[param_count]
    