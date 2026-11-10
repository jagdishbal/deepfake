import os
import random
import shutil
from pathlib import Path

random.seed(42)
root = Path('D:/deepfake_college_project/Deepfake-Analysis/dataset')

for mod in ['image', 'video']:
    for cls in ['real', 'fake']:
        train_dir = root / mod / 'train' / cls
        val_dir = root / mod / 'validation' / cls
        val_dir.mkdir(parents=True, exist_ok=True)
        
        files = list(train_dir.glob('*.*'))
        num_val = int(len(files) * 0.2)
        if num_val > 0:
            val_files = random.sample(files, num_val)
            for f in val_files:
                shutil.move(str(f), str(val_dir / f.name))
            print(f'Moved {len(val_files)} files to {mod}/validation/{cls}')
        else:
            print(f'No files to move for {mod}/train/{cls}')
