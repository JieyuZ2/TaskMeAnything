import random
from geometry_task import MathTemplateMetaData, PerimeterGenerator, MidpointGenerator, IntersectionGenerator, CircleGenerator, AngleGenerator, TaskStore

def main():
    # Define the path to templates using a raw string
    Circle_template_path = r'C:\Users\jiata\OneDrive\Desktop\Raivn\TaskMeAnyMath\annotations\math_annotations\circle_templates.json'
    Angle_template_path = r'C:\Users\jiata\OneDrive\Desktop\Raivn\TaskMeAnyMath\annotations\math_annotations\angle_template.json'
    Perimeter_template_path = r'C:\Users\jiata\OneDrive\Desktop\Raivn\TaskMeAnyMath\annotations\math_annotations\perimeter_templates.json'
    Midpoint_template_path = r'C:\Users\jiata\OneDrive\Desktop\Raivn\TaskMeAnyMath\annotations\math_annotations\midpoint_templates.json'
    Intersection_template_path = r'C:\Users\jiata\OneDrive\Desktop\Raivn\TaskMeAnyMath\annotations\math_annotations\intersection_templates.json'

    Circle_metadata = MathTemplateMetaData(template_path=Circle_template_path)

    Circle_task_store = TaskStore(schema = CircleGenerator.schema)

    # metadata = MathTemplateMetaData(template_path=Circle_template_path)
    metadata = MathTemplateMetaData(template_path=Circle_template_path)

    Circle_generator = CircleGenerator(metadata=Circle_metadata)
    Circle_generator.enumerate_task_plans(Circle_task_store)
 
    # Select and print a random task
    Circle_tasks = list(Circle_task_store)
    if Circle_tasks:
        random_Circle_task = random.choice(Circle_tasks)
        Circle_task = Circle_generator.generate(random_Circle_task)
        print("Random Circle Task:")
        print("Question:", Circle_task['question'])
        print("Answer:", Circle_task['answer'])
if __name__ == "__main__":
    main()