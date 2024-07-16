from midPoint_tasks import MidpointGenerator
from metadataMath import MathTemplateMetaData
from task_storeMath import TaskStore
from geometry_task import GeoPlanGenerator

def main():
    # Define absolute paths to metadata and templates using raw strings
    path_to_metadata = r'C:\Users\jiata\OneDrive\Desktop\Raivn\TaskMeAnyMath\metadata.json'  # Adjust this path as needed
    template_path = r'C:\Users\jiata\OneDrive\Desktop\Raivn\TaskMeAnyMath\annotations\math_annotations\midpoint.json'

    metadata = MathTemplateMetaData(path_to_metadata=path_to_metadata, template_path=template_path)
    task_store = TaskStore()

    # Initialize and run the MidpointGenerator
    midpoint_generator = MidpointGenerator(metadata)
    midpoint_generator.enumerate_task_plans(task_store)

    # Generate and print tasks
    for task_plan in task_store.tasks:
        task = midpoint_generator.generate(task_plan)
        print(task)

if __name__ == "__main__":
    main()