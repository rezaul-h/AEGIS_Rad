import json,platform,subprocess,sys
from pathlib import Path
import torch

def main():
    out={'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,
         'cuda_available':torch.cuda.is_available(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    try:out['pip_freeze']=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines()
    except Exception as e:out['pip_freeze_error']=str(e)
    Path('environment_snapshot.json').write_text(json.dumps(out,indent=2)); print(json.dumps({k:v for k,v in out.items() if k!='pip_freeze'},indent=2))
if __name__=='__main__':main()
