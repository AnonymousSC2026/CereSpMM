import json
import os
import tarfile
import shutil
import numpy as np

from cerebras.sdk.client import SdkLauncher

with open("artifact_path.json", "r", encoding="utf8") as f:
    artifact_path = json.load(f)["artifact_path"]

def extract_for_inspect(artifact_path: str) -> str:
    if os.path.exists("out"):
        shutil.rmtree("out")
    os.makedirs("out", exist_ok=True)

    if artifact_path.endswith(".tar.gz"):
        with tarfile.open(artifact_path, "r:gz") as tar:
            tar.extractall("out")
        subs = [s for s in os.listdir("out") if os.path.isdir(os.path.join("out", s))]
        if len(subs) != 1:
            raise RuntimeError(f"Expect 1 subdir in ./out after extract, got {subs}")
        top = os.path.join("out", subs[0])             # out/<hash>
        cand = os.path.join(top, "out")                # out/<hash>/out
        if not os.path.exists(os.path.join(cand, "out.json")):
            raise FileNotFoundError(f"out.json not found in {cand}")
        return cand
    else:
        
        if os.path.isdir(artifact_path):
            cand1 = os.path.join(artifact_path, "out.json")
            if os.path.exists(cand1):
                return artifact_path
            cand2 = os.path.join(artifact_path, "out")
            if os.path.exists(os.path.join(cand2, "out.json")):
                return cand2
        raise RuntimeError(f"Invalid artifact_path: {artifact_path}")


app_dir_for_inspect = extract_for_inspect(artifact_path)
print(f"[INFO] Inspect dir: {app_dir_for_inspect}")

with open(os.path.join(app_dir_for_inspect, "out.json"), "r", encoding="utf8") as f:
    compile_data = json.load(f)

with SdkLauncher(artifact_path, simulator=False,  disable_version_check=True) as launcher:

    # Transfer an additional file to the appliance,
    # then write contents to stdout on appliance
    launcher.stage("run.py")
    #launcher.stage("tmp.csv")
    launcher.stage("tmp_val_pad.csv")
    launcher.stage("tmp_x_pad.csv")
    launcher.stage("tmp_y_pad.csv")
    #response = launcher.run(
        #"echo \"ABOUT TO RUN IN THE APPLIANCE\"",
        #"cat additional_artifact.txt",
    #)
    #print("Test response: ", response)

    # Run the original host code as-is on the appliance,
    # using the same cmd as when using the Singularity container
    response = launcher.run("cs_python run.py --name out --cmaddr %CMADDR%")
    print("Host code execution response: ", response)

    # Fetch files from the appliance
    #launcher.download_artifact("sim.log", "./output_dir/sim.log")
