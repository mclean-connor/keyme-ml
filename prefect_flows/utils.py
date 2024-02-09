import subprocess

def run_subprocess(command, task_name=""):
    print(f"Running {task_name}...")
    result = subprocess.run(command, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"Error in {task_name}: {result.stderr}")
    if result.returncode != 0:
        raise Exception(f"{task_name} failed")
