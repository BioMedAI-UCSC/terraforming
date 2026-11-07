"""Download the ICLR run tree, pod logs and source/environment provenance."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'apps/nautalis'))
from download import main

if __name__=='__main__':
    main(default_job='iclr-all',default_path='/work/outputs/iclr_experiments',
         default_output='outputs/iclr_experiments')
