task_list = []

def add_task(task):
    task_list.append(task)



def mark_task_as_completed(index):
    if 0 <= index < len(task_list):
        task_list[index]['completed'] = True


def mark_task_as_in_progress(index):
    if 0 <= index < len(task_list):
        task_list[index]['status'] = 'En cours'



