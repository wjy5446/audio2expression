import tensorflow as tf

from model import *

if __name__ == '__main__':
    model = A2E()
    model.build_model()

    path_audio = 'data/do_1.wav'
    output = model.get_expression_from_audio(path_audio, 3)
    print(output)