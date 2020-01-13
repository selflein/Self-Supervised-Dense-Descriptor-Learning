## Visual Descriptor Learning

### Environment setup
```
pip install pipenv
pipenv install
pipenv shell
```
Then training can be started.


### Start training
```
python visual_descriptor/train.py --name test --dataset_path /storage/remote/atcremers51/w0020/visual_descriptor/frames --batch_size 4 --cuda
```

### Visualize training curve
```
tensorboard --logdir logs/{name of the experiment above}
```