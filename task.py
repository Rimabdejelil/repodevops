task_list = []

def add_task(task):
    task_list.append(task)

def display_tasks():
    for task in task_list:
        print(task)

# task_manager.py (dans la branche feature/marquer-tache-terminee)

def mark_task_as_completed(index):
    if 0 <= index < len(task_list):
        task_list[index]['completed'] = True


