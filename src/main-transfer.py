import argparse
import os
import tensorflow as tf
import numpy as np
from PIL import Image

import constants as c


def parse_f():
    parser = argparse.ArgumentParser(description='Neural Style Transfer')
    parser.add_argument('--content_im_path', default=c.EX_IM_PATH, help='path of input image, including filename')
    parser.add_argument('--output_im_path', default=c.EX_OUT_PATH, help='path of the folder in which the output image will be stored')
    return parser


def layer_dims_f(model, layer_names):
    num_kernels = np.zeros(len(layer_names), dtype='int32')
    dim_kernels = np.zeros(len(layer_names), dtype='int32')
    for i, l_name in enumerate(layer_names):
        num_kernels[i] = model.get_layer(l_name).output_shape[3]
        dim_kernels[i] = model.get_weights()[0].shape[0] ** 2
    return num_kernels, dim_kernels


def loss_f(model_vgg, content_ref_fmap, style_ref_grams, num_kernels, dim_kernels, gen_im):

    def _loss_f():
        gen_im_preproc = tf.keras.applications.vgg19.preprocess_input(gen_im)

        content_gen_fmap = content_loss_f(model_vgg, gen_im_preproc)
        content_loss = 0.5 * tf.math.reduce_sum(tf.math.square(content_ref_fmap - content_gen_fmap))

        style_gen_grams = style_loss_f(model_vgg, gen_im_preproc)
        diff_styles = [tf.math.square(a - b) for a, b in zip(style_ref_grams, style_gen_grams)]
        diff_styles_reduced = tf.TensorArray(dtype='float32', size=len(diff_styles))
        for i, diff_style in enumerate(diff_styles):
            diff_styles_reduced = diff_styles_reduced.write(i, tf.math.reduce_sum(diff_style))
        diff_styles_reduced = diff_styles_reduced.stack()

        style_loss = 1. / ((2 * num_kernels * dim_kernels) ** 2) * diff_styles_reduced
        style_loss = tf.tensordot(c.STYLE_WEIGHTS, style_loss, axes=1)

        total_loss = c.ALPHA * content_loss + c.BETA * style_loss
        return total_loss

    return _loss_f


def content_loss_f(model_vgg, im_preproc):
    content_ext_model = tf.keras.Model(inputs=model_vgg.inputs,
                                       outputs=model_vgg.get_layer(c.CONTENT_LAYER_NAME).output)
    content_fmap = content_ext_model(im_preproc)
    return content_fmap


def style_loss_f(model_vgg, im_preproc):
    style_ext_models = [tf.keras.Model(inputs=model_vgg.inputs, outputs=model_vgg.get_layer(layer).output) \
                        for layer in c.STYLE_LAYER_NAMES]
    style_ref_fmaps = [style_model(im_preproc) for style_model in style_ext_models]

    def _gram_matrix_f(kernels):
        kernels = kernels[0, :, :, :]
        num_kernels = kernels.shape[2]
        flattened_kernels = tf.reshape(kernels, (num_kernels, -1))
        gram_matrix = tf.tensordot(flattened_kernels, tf.transpose(flattened_kernels), axes=1)
        return gram_matrix

    style_ref_grams = [_gram_matrix_f(f_maps) for f_maps in style_ref_fmaps]
    return style_ref_grams


def load_image_tf(path):
    im = Image.open(path)
    im = np.array(im)
    im = np.expand_dims(im, 0)
    im = tf.Variable(im, dtype='float32')
    return im


def main():

    # Load content image
    content_im = load_image_tf(c.EX_IM_PATH)
    content_im_preproc = tf.keras.applications.vgg19.preprocess_input(content_im)

    # VGG model
    _, width, height, channels = content_im.shape
    model_vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_tensor=None,
                                            input_shape=(width, height, 3), pooling='max')
    model_vgg.trainable = False
    num_kernels, dim_kernels = layer_dims_f(model_vgg, c.STYLE_LAYER_NAMES)

    # Content loss
    content_ref_fmap = content_loss_f(model_vgg, content_im_preproc)

    # Load style image
    style_im = load_image_tf(c.STYLE_IM_PATH)
    style_im_preproc = tf.keras.applications.vgg19.preprocess_input(style_im)

    # Style loss
    style_ref_grams = style_loss_f(model_vgg, style_im_preproc)

    # Create noise image
    # gen_im = np.random.randint(0, 256, (1, width, height, channels))
    # gen_im = np.zeros((1, width, height, channels))
    # gen_im = tf.Variable(np.array(gen_im), dtype='float32')

    # Start iterating from original image
    gen_im = tf.Variable(content_im)

    # Optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=1)

    for epoch in range(c.EPOCHS):
        opt.minimize(loss=loss_f(model_vgg, content_ref_fmap, style_ref_grams, num_kernels, dim_kernels, gen_im), var_list=[gen_im])
        if epoch % 5 == 0:
            save_im = gen_im.numpy()[0, :, :, :]
            save_im = np.where(save_im <= 255, save_im, 255)
            save_im = np.where(save_im >= 0, save_im, 0)
            save_im = save_im.astype(np.uint8)
            save_im = Image.fromarray(save_im)
            save_im.save(os.path.join(c.EX_OUT_PATH, 'output-example' + str(epoch) + '.jpg'))


if __name__ == '__main__':
    # args = parse_f().parse_args()
    # args_dict = args.__dict__
    # args_dict['content_layer_name'] = c.CONTENT_LAYER_NAME
    # args_dict['style_layer_names'] = c.STYLE_LAYER_NAMES
    # args_dict['style_weights'] = c.STYLE_WEIGHTS
    # args_dict['alpha'] = c.ALPHA
    # args_dict['beta'] = c.BETA
    # main(**args_dict)
    main()
