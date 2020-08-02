import os

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(BASE_PATH, 'src')

EX_IM_PATH = os.path.join(os.path.join(BASE_PATH, 'data'), 'input-example.jpg')
STYLE_IM_PATH = os.path.join(os.path.join(BASE_PATH, 'data'), 'style-example.jpg')
# EX_OUT_PATH = os.path.join(os.path.join(BASE_PATH, 'output'), 'output-example.jpg')
EX_OUT_PATH = os.path.join(BASE_PATH, 'output')


CONTENT_LAYER_NAME = 'block4_conv2'
STYLE_LAYER_NAMES = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1',
]
STYLE_WEIGHTS = [0.2, 0.2, 0.2, 0.2, 0.2]
ALPHA = 1
BETA = 10000
EPOCHS = 3000
