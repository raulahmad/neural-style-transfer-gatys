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
    parser.add_argument('--content_layer', default=c.CONTENT_LAYER_NAME, help='content layer name whose feature maps will be used for loss computation. Read Readme for further information')
    parser.add_argument('--style_layer', default=c.STYLE_LAYER_NAMES, help='style layer name(s) whose feature maps will be used for loss computation. Read Readme for further information')
    parser.add_argument('--style_layer_weights', default=c.STYLE_WEIGHTS, help='weights for each style layer')
    parser.add_argument('--content_loss_weight', default=c.ALPHA, help='weight of content loss on loss function')
    parser.add_argument('--style_loss_weight', default=c.BETA, help='weight of style loss on loss function')
    parser.add_argument('--epochs', default=c.EPOCHS, help='number of iterations when training')
    return parser


def layer_dims_f(model, layer_names):
    num_kernels = np.zeros(len(layer_names), dtype='int32')
    dim_kernels = np.zeros(len(layer_names), dtype='int32')
    for i, l_name in enumerate(layer_names):
        num_kernels[i] = model.get_layer(l_name).output_shape[3]
        dim_kernels[i] = model.get_weights()[0].shape[0] ** 2
    return num_kernels, dim_kernels


def loss_f(content_ext_model, style_ext_models, content_ref_fmap, style_ref_grams, num_kernels, dim_kernels, gen_im,
           style_layer_weights, content_loss_weight, style_loss_weight):

    def _loss_f():
        gen_im_preproc = tf.keras.applications.vgg19.preprocess_input(gen_im)

        content_gen_fmap = content_ext_model(gen_im_preproc)
        content_loss = 0.5 * tf.math.reduce_sum(tf.math.square(content_ref_fmap - content_gen_fmap))

        style_gen_grams = style_grams_f(style_ext_models, gen_im_preproc)
        diff_styles = [tf.math.square(a - b) for a, b in zip(style_ref_grams, style_gen_grams)]
        diff_styles_reduced = tf.TensorArray(dtype='float32', size=len(diff_styles))
        for i, diff_style in enumerate(diff_styles):
            diff_styles_reduced = diff_styles_reduced.write(i, tf.math.reduce_sum(diff_style))
        diff_styles_reduced = diff_styles_reduced.stack()

        style_loss = 1. / ((2 * num_kernels * dim_kernels) ** 2) * diff_styles_reduced
        style_loss = tf.tensordot(style_layer_weights, style_loss, axes=1)

        total_loss = content_loss_weight * content_loss + style_loss_weight * style_loss
        return total_loss

    return _loss_f


def style_grams_f(style_ext_models, im_preproc):
    style_ref_fmaps = [style_model(im_preproc) for style_model in style_ext_models]

    def _gram_matrix_f(kernels):
        kernels = kernels[0, :, :, :]
        num_kernels = kernels.shape[2]
        flattened_kernels = tf.reshape(kernels, (num_kernels, -1))
        gram_matrix = tf.tensordot(flattened_kernels, tf.transpose(flattened_kernels), axes=1)
        return gram_matrix

    style_ref_grams = [_gram_matrix_f(f_maps) for f_maps in style_ref_fmaps]
    return style_ref_grams


def load_image_tf_f(path):
    im = Image.open(path)
    im = np.array(im)
    im = np.expand_dims(im, 0)
    im = tf.Variable(im, dtype='float32')
    return im


def main(**kwargs):

    # Load content image
    content_im = load_image_tf_f(kwargs['content_im_path'])
    content_im_preproc = tf.keras.applications.vgg19.preprocess_input(content_im)

    # VGG model
    _, width, height, channels = content_im.shape
    model_vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_tensor=None,
                                            input_shape=(width, height, 3), pooling='max')
    model_vgg.trainable = False
    num_kernels, dim_kernels = layer_dims_f(model_vgg, kwargs['style_layer'])

    # Content loss: model to compute feature maps
    content_ext_model = tf.keras.Model(inputs=model_vgg.inputs,
                                       outputs=model_vgg.get_layer(kwargs['content_layer']).output)

    # Content loss: feature maps of input (reference) image
    content_ref_fmap = content_ext_model(content_im_preproc)

    # Style loss: load style image
    style_im = load_image_tf_f(c.STYLE_IM_PATH)
    style_im_preproc = tf.keras.applications.vgg19.preprocess_input(style_im)

    # Style loss: gram matrix of input (reference) image
    style_ext_models = [tf.keras.Model(inputs=model_vgg.inputs, outputs=model_vgg.get_layer(layer).output) \
                        for layer in kwargs['style_layer']]
    style_ref_grams = style_grams_f(style_ext_models, style_im_preproc)

    # Create noise image
    # gen_im = np.random.randint(0, 256, (1, width, height, channels))
    # gen_im = np.zeros((1, width, height, channels))
    # gen_im = tf.Variable(np.array(gen_im), dtype='float32')

    # Start iterating from original image
    gen_im = tf.Variable(content_im)

    # Optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=1)

    for epoch in range(kwargs['epochs']):
        opt.minimize(
            loss=loss_f(content_ext_model, style_ext_models, content_ref_fmap, style_ref_grams, num_kernels, dim_kernels, gen_im,
                        kwargs['style_layer_weights'], kwargs['content_loss_weight'], kwargs['style_loss_weight']),
            var_list=[gen_im])
        if epoch % 5 == 0:
            save_im = gen_im.numpy()[0, :, :, :]
            save_im = np.where(save_im <= 255, save_im, 255)
            save_im = np.where(save_im >= 0, save_im, 0)
            save_im = save_im.astype(np.uint8)
            save_im = Image.fromarray(save_im)
            save_im.save(os.path.join(kwargs['output_im_path'], 'output-example' + str(epoch) + '.jpg'))


if __name__ == '__main__':
    args = parse_f().parse_args()
    kwargs = args.__dict__
    main(**kwargs)
