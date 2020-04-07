# reinforcement-learning-DQN
Implementation of the vanilla Deep Q-Learning(DQN)

## Deployment
Follow the following steps to run the code: 

```shell script
$ git clone https://github.com/khordoo/reinforcement-learning-cross-entropy-Pytorch.git
$ cd reinforcement-learning-cross-entropy-Pytorch
$ python3 -m venv venv 
$ source /venv/bin/activate
$ pip3 install -r requirements.txt
$ python3 cartploe-cross-entropy.py 
```
To view loss and rewards charts in Tensorboard: 
```shell script
$ tensorboard --logdir runs/
```
This will run a local server on your machine ,Click on the displayed link to open  the tensorboard in your browser and view the charts

### Results
Here is an animation showing a trained agent playing the game.

![Cartpole-trained](images/cartpole-trained.gif)