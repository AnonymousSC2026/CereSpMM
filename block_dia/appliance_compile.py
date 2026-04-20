import json
from cerebras.sdk.client import SdkCompiler

compiler = SdkCompiler()

options = (
    "--arch=wse3 "
    "--fabric-dims=762,1172 "
    "--fabric-offsets=4,1 "
    "--params=h:1170,w:755,Mt:4,Kt:5,Nt:30,MAX_NUM_DIAGS:11,MAX_VALS_PER_PX:20 "
    "--memcpy --channels=1 "
    "-o out"
)
with SdkCompiler(disable_version_check=True) as compiler:
    artifact_path = compiler.compile(
        ".",         
        "layout.csl",
        options,        
        "."          
    )

with open("artifact_path.json", "w", encoding="utf8") as f:
    json.dump({"artifact_path": artifact_path}, f)
