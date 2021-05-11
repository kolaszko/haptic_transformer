#second config
python kfold_train.py --dataset-path=/media/titan/hdd/Datasets/haptic_anymal/rokubimini_lighter.pickle --projection-dim 16 --nheads 4 --num-encoder-layers 4 --feed-forward 256 --dropout 0.1 --lr 1e-3 --weight-decay 1e-4 --batch-size 512
python kfold_train.py --dataset-path=/media/titan/hdd/Datasets/haptic_anymal/rokubimini_lighter.pickle --projection-dim 16 --nheads 4 --num-encoder-layers 4 --feed-forward 256 --dropout 0.2 --lr 1e-3 --weight-decay 1e-4 --batch-size 1024
python kfold_train.py --dataset-path=/media/titan/hdd/Datasets/haptic_anymal/rokubimini_lighter.pickle --projection-dim 16 --nheads 4 --num-encoder-layers 4 --feed-forward 256 --dropout 0.3 --lr 1e-3 --weight-decay 1e-4 --batch-size 512
python kfold_train.py --dataset-path=/media/titan/hdd/Datasets/haptic_anymal/rokubimini_lighter.pickle --projection-dim 16 --nheads 4 --num-encoder-layers 4 --feed-forward 256 --dropout 0.4 --lr 1e-3 --weight-decay 1e-4 --batch-size 512
#third config
python kfold_train.py --dataset-path=/media/titan/hdd/Datasets/haptic_anymal/rokubimini_lighter.pickle --projection-dim 16 --nheads 8 --num-encoder-layers 8 --feed-forward 512 --dropout 0.1 --lr 1e-3 --weight-decay 1e-4 --batch-size 128
python kfold_train.py --dataset-path=/media/titan/hdd/Datasets/haptic_anymal/rokubimini_lighter.pickle --projection-dim 16 --nheads 8 --num-encoder-layers 8 --feed-forward 512 --dropout 0.2 --lr 1e-3 --weight-decay 1e-4 --batch-size 128
python kfold_train.py --dataset-path=/media/titan/hdd/Datasets/haptic_anymal/rokubimini_lighter.pickle --projection-dim 16 --nheads 8 --num-encoder-layers 8 --feed-forward 512 --dropout 0.3 --lr 1e-3 --weight-decay 1e-4 --batch-size 128
python kfold_train.py --dataset-path=/media/titan/hdd/Datasets/haptic_anymal/rokubimini_lighter.pickle --projection-dim 16 --nheads 8 --num-encoder-layers 8 --feed-forward 512 --dropout 0.4 --lr 1e-3 --weight-decay 1e-4 --batch-size 128

#first config
python kfold_train.py --dataset-path=/media/titan/hdd/Datasets/haptic_anymal/rokubimini_lighter.pickle --projection-dim 16 --nheads 2 --num-encoder-layers 1 --feed-forward 128 --dropout 0.1 --lr 1e-3 --weight-decay 1e-4 --batch-size 1024
python kfold_train.py --dataset-path=/media/titan/hdd/Datasets/haptic_anymal/rokubimini_lighter.pickle --projection-dim 16 --nheads 2 --num-encoder-layers 1 --feed-forward 128 --dropout 0.2 --lr 1e-3 --weight-decay 1e-4 --batch-size 1024
python kfold_train.py --dataset-path=/media/titan/hdd/Datasets/haptic_anymal/rokubimini_lighter.pickle --projection-dim 16 --nheads 2 --num-encoder-layers 1 --feed-forward 128 --dropout 0.3 --lr 1e-3 --weight-decay 1e-4 --batch-size 1024
python kfold_train.py --dataset-path=/media/titan/hdd/Datasets/haptic_anymal/rokubimini_lighter.pickle --projection-dim 16 --nheads 2 --num-encoder-layers 1 --feed-forward 128 --dropout 0.4 --lr 1e-3 --weight-decay 1e-4 --batch-size 1024