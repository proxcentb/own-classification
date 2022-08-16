white = 255
black = 0
red = (255, 0, 0)
fps = 480
w, h = 640, 640
pw, ph = 64, 64
pws, phs = w // pw, h // ph

val_split = 0.2
trainPath = './processed/train'
testPath = './processed/test'

batch_size = 10
momentum_value = 0.8
epochs = 4
learning_rate = 0.1
use_cuda = True
device = "cuda"