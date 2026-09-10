import argparse,subprocess,sys
from pathlib import Path
from aegis_rad.config import load_config

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--tokenizer',required=True); ap.add_argument('--output-root',required=True); a=ap.parse_args(); cfg=load_config(a.config)
    for seed in cfg.training.seeds:
        out=Path(a.output_root)/f'seed_{seed}'
        subprocess.run([sys.executable,str(Path(__file__).with_name('train.py')),'--config',a.config,'--tokenizer',a.tokenizer,'--output-dir',str(out),'--seed',str(seed)],check=True)
if __name__=='__main__':main()
