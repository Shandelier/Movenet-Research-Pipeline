conda activate tf
python ./main.py --pose 0 --input 'input\Synthetic-HDR-26s\Straight-Synthetic-HDR-26s'
python ./main.py --pose 1 --input 'input\Synthetic-HDR-26s\Slouch-Synthetic-HDR-26s'
python ./main.py --pose 0 --input 'input\Synthetic-noHDR-26s\Straight-Synthetic-noHDR-26s'
python ./main.py --pose 1 --input 'input\Synthetic-noHDR-26s\Slouch-Synthetic-noHDR-26s'
