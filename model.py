import copy
import math
import json
import resampy

import numpy as np
import tensorflow as tf
from scipy.io import wavfile

from python_speech_features import mfcc


class A2E(object):
    def __init__(self, path_model_ds="model/deepspeech.pb", path_model_a2e="model/a2e.pb", path_3dmm_info="data/facecap"):
        self.path_model_ds = path_model_ds
        self.path_model_a2e = path_model_a2e
        self.path_3dmm_info = path_3dmm_info

        self.audio_handler = AudioHandler(path_model_ds)

    def build_model(self):
        self.audio_handler.build_model()

        with tf.io.gfile.GFile("model/a2e.pb", "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="a2e")

        # build session
        self.graph = tf.get_default_graph()
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(graph=self.graph, config=config)

        self.input_a2e = self.graph.get_tensor_by_name('a2e/audio:0')
        self.output_a2e = self.graph.get_tensor_by_name('a2e/output:0')

    def smooth(self, audio, window_len=3, window='hanning'):
        s = np.r_[audio[window_len-1:0:-1], audio, audio[-2:-window_len-1:-1]]

        if window == 'flat':  # moving average
            w = np.ones(window_len, 'd')
        else:
            w = eval('np.'+window+'(window_len)')

        audio_smooth = np.convolve(w/w.sum(), s, mode='valid')
        edge = window_len // 2
        return audio_smooth[edge:-(edge)]

    def get_expression_from_audio(self, path_audio, smooth=0):
        input_audio = self.audio_handler.process_audio(path_audio)
        output = self.sess.run(self.output_a2e, feed_dict={
                               self.input_a2e: input_audio})[0]
        output[output < 0] = 0

        if smooth > 0:
            for i in range(51):
                output[:, i] = self.smooth(output[:, i], window_len=smooth)

        output = output.tolist()

        return output


class AudioHandler(object):
    def __init__(self, path_model="model/deepspeech.pb", n_mfcc=26, n_context=9, fps=30, win_size=16, win_stride=1):
        self.path_model = path_model
        self.n_mfcc = n_mfcc
        self.n_context = n_context
        self.fps = fps
        self.win_size = win_size
        self.win_stride = win_stride

    def build_model(self):
        with tf.io.gfile.GFile(self.path_model, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="deepspeech")

        # build session
        self.graph = tf.get_default_graph()
        config = tf.ConfigProto(log_device_placement=False)
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(graph=self.graph, config=config)

        # tensor
        self.input_ds = self.graph.get_tensor_by_name(
            'deepspeech/input_node:0')
        self.seq_length_ds = self.graph.get_tensor_by_name(
            'deepspeech/input_lengths:0')
        self.output_ds = self.graph.get_tensor_by_name('deepspeech/logits:0')

    def process_audio(self, path_audio):
        sr, audio = wavfile.read(path_audio)

        audio_len_s = float(audio.shape[0]) / sr
        n_frames = int(math.ceil(audio_len_s * self.fps))

        if audio.ndim != 1:
            audio = audio[:, 0]

        audio_copy = copy.deepcopy(audio)
        audio_resample = resampy.resample(audio_copy.astype(float), sr, 16000)

        audio_mfcc = self.convert_mfcc(audio_resample)
        audio_ds_logit = self.get_deepspeech_logit(audio_mfcc)

        audio_inter = self.interpolate_feature(
            audio_ds_logit, 50, self.fps, n_frames)
        return self.make_window(audio_inter, self.win_size, self.win_stride)

    def convert_mfcc(self, audio):
        audio_mfcc = mfcc(audio, samplerate=16000, numcep=self.n_mfcc)
        audio_mfcc = audio_mfcc[::2]
        n_strides = len(audio_mfcc)

        empty_context = np.zeros(
            (self.n_context, self.n_mfcc), dtype=audio_mfcc.dtype)
        audio_mfcc = np.concatenate((empty_context, audio_mfcc, empty_context))

        window_size = 2 * self.n_context + 1
        audio_mfcc_window = np.lib.stride_tricks.as_strided(
            audio_mfcc,
            (n_strides, window_size, self.n_mfcc),
            (audio_mfcc.strides[0], audio_mfcc.strides[0],
             audio_mfcc.strides[1]),
            writeable=False)

        audio_mfcc_window = np.reshape(audio_mfcc_window, [n_strides, -1])
        audio_mfcc_window = np.copy(audio_mfcc_window)
        audio_mfcc_window = (
            audio_mfcc_window - np.mean(audio_mfcc_window)) / np.std(audio_mfcc_window)

        return audio_mfcc_window

    def get_deepspeech_logit(self, audio_mfcc):
        output = self.sess.run(self.output_ds,
                               feed_dict={self.input_ds: audio_mfcc[np.newaxis, ...],
                                          self.seq_length_ds: [audio_mfcc.shape[0]]})
        return output

    def interpolate_feature(self, input_features, input_rate, output_rate, n_frames):
        n_features = input_features[:, 0].shape[1]
        input_len = input_features[:, 0].shape[0]
        seq_len = input_len / float(input_rate)
        output_len = n_frames

        input_timestamps = np.arange(input_len) / float(input_rate)
        output_timestamps = np.arange(output_len) / float(output_rate)
        output_features = np.zeros((output_len, n_features))

        for feat in range(n_features):
            output_features[:, feat] = np.interp(
                output_timestamps, input_timestamps, input_features[:, 0][:, feat])

        return output_features

    def make_window(self, input_features, win_size, win_stride):
        # make_window
        zero_pad = np.zeros((int(win_size / 2), input_features.shape[1]))
        output = np.concatenate((zero_pad, input_features, zero_pad), axis=0)
        windows = []

        for win_index in range(0, output.shape[0] - win_size, win_stride):
            windows.append(output[win_index:win_index+win_size])

        return np.array(windows)
