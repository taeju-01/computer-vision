import os
from roboflow import Roboflow

def main():
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY가 설정되지 않았습니다. 예: export ROBOFLOW_API_KEY=...")

    workspace = os.getenv("WORKSPACE", "joseph-nelson")
    project_name = os.getenv("PROJECT", "hard-hat-workers")
    version_num = int(os.getenv("VERSION", "1"))
    fmt = os.getenv("FORMAT", "yolov8")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version = project.version(version_num)
    dataset = version.download(fmt)

    print("Downloaded dataset to:", dataset.location)

if __name__ == "__main__":
    main()
