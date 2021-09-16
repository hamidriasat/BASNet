import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
from resnet import basic_block, make_layer

from loss import iou_loss, bce_loss, ssim_loss


"""
apply Convolution+BN+Relu layer
"""
def conv_bn_relu(x_in, planes, dilation=1, stride=1, padding="same"):
    x = layers.Conv2D(filters=planes, kernel_size=(3, 3), strides=stride, padding=padding, dilation_rate=dilation)(x_in)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


"""
Creates a convolution block with three consecutive 3*3 conv's
"""
def conv_block(x_in, planes, dilation=1):
    x = conv_bn_relu(x_in=x_in, planes=planes, dilation=dilation)
    x = conv_bn_relu(x_in=x, planes=planes, dilation=dilation)
    x = conv_bn_relu(x_in=x, planes=planes, dilation=dilation)

    return x


"""
Resize input tensor using bilinear upsampling strategy
factor: resize input with given factor
"""
def resize_bilinear(x_in, factor=2):
    h = tf.keras.backend.int_shape(x_in)[1] * factor
    w = tf.keras.backend.int_shape(x_in)[2] * factor
    return tf.image.resize(x_in, [h, w])


"""
Segmentation head uses convolution to map each stage output to final 
number of classes, if final size is not null then also resizes to given size.
x_in: input tensor
out_planes: # output classes
final_size: size to which each stage output will be resized 
"""
def segmentation_head(x_in, out_planes, final_size):
    x = layers.Conv2D(out_planes, kernel_size=(3, 3), strides=1, padding="same",)(x_in)

    if final_size is not None:
        x = tf.image.resize(x, size=final_size)

    return x


"""
BASNet prediction module, it outputs coarse label map
input_shape: shape of model input tensor
layers_arg: ResNet-34 block sizes at different layer
num_classes: # of output classes
planes: filter size used through the model architecture
"""
def BASNet_Predict(input_shape, layers_arg, num_classes, planes):

    x_in = layers.Input(input_shape)
    downsample = None

    """-------------Encoder--------------"""
    x = layers.Conv2D(planes, kernel_size=(3, 3),strides=1, padding='same')(x_in)

    "encoder stage/layer 1"
    e_stage_1 = make_layer(x, basic_block, planes, planes, layers_arg[0], expansion=1)
    x = layers.Activation("relu")(e_stage_1)

    "encoder stage/layer 2"
    #  1 --> 1/2
    e_stage_2 = make_layer(x, basic_block, planes, planes*2, layers_arg[1], stride=2,expansion=1)
    x = layers.Activation("relu")(e_stage_2)

    "encoder stage/layer 3"
    #  1/2 --> 1/4
    e_stage_3 = make_layer(x, basic_block, planes*2, planes * 4, layers_arg[2], stride=2, expansion=1)
    x = layers.Activation("relu")(e_stage_3)

    "encoder stage/layer 4"
    #  1/4 --> 1/8
    e_stage_4 = make_layer(x, basic_block, planes * 4, planes * 8, layers_arg[3], stride=2, expansion=1)
    x = layers.Activation("relu")(e_stage_4)

    "encoder stage 5"
    #  1/8 --> 1/16
    x = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(x)
    x = basic_block(x, planes * 8, downsample=downsample)
    x = basic_block(x, planes * 8, downsample=downsample)
    e_stage_5 = basic_block(x, planes * 8, downsample=downsample)

    "encoder stage 6"
    #  1/16 --> 1/32
    x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(e_stage_5)
    x = basic_block(x, planes * 8, downsample=downsample)
    x = basic_block(x, planes * 8, downsample=downsample)
    e_stage_6 = basic_block(x, planes * 8, downsample=downsample)

    """-------------Bridge--------------"""

    bridge = conv_block(e_stage_6, planes * 8, dilation=2)

    """-------------Decoder--------------"""

    "decoder stage 6"
    # dilation difference
    x = layers.concatenate([e_stage_6, bridge], axis=-1)
    d_stage_6 = conv_block(x, planes * 8)

    "decoder stage 5"
    #  1/32 --> 1/16
    x = resize_bilinear(d_stage_6,)
    x = layers.concatenate([e_stage_5, x], axis=-1)
    d_stage_5 = conv_block(x, planes * 8)

    "decoder stage 4"
    #  1/16 --> 1/8
    x = resize_bilinear(d_stage_5,)
    x = layers.concatenate([e_stage_4, x], axis=-1)
    d_stage_4 = conv_block(x, planes * 8)

    "decoder stage 3"
    #  1/8 --> 1/4
    x = resize_bilinear(d_stage_4,)
    x = layers.concatenate([e_stage_3, x], axis=-1)
    d_stage_3 = conv_block(x, planes * 8)

    "decoder stage 2"
    #  1/4 --> 1/2
    x = resize_bilinear(d_stage_3,)
    x = layers.concatenate([e_stage_2, x], axis=-1)
    d_stage_2 = conv_block(x, planes * 8)

    "decoder stage 1"
    #  1/2 --> 1
    x = resize_bilinear(d_stage_2,)
    x = layers.concatenate([e_stage_1, x], axis=-1)
    d_stage_1 = conv_block(x, planes * 8)

    """-------------Side Output--------------"""
    d_stage_1 = segmentation_head(d_stage_1, num_classes, None)  # 1
    d_stage_2 = segmentation_head(d_stage_2, num_classes, input_shape[:2])  # 1/2
    d_stage_3 = segmentation_head(d_stage_3, num_classes, input_shape[:2])  # 1/4
    d_stage_4 = segmentation_head(d_stage_4, num_classes, input_shape[:2])  # 1/8
    d_stage_5 = segmentation_head(d_stage_5, num_classes, input_shape[:2])  # 1/16
    d_stage_6 = segmentation_head(d_stage_6, num_classes, input_shape[:2])  # 1/32
    bridge = segmentation_head(bridge, num_classes, input_shape[:2])        # 1/32

    model = models.Model(inputs=[x_in], outputs=[d_stage_1, d_stage_2, d_stage_3, d_stage_4, d_stage_5, d_stage_6, bridge])

    # set weight initializers
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel_initializer = tf.keras.initializers.he_normal()
        if hasattr(layer, 'depthwise_initializer'):
            layer.depthwise_initializer = tf.keras.initializers.he_normal()

    return model


"""
BASNet Residual Refinement Module(RRM) module, it outputs fine label map
base_model: base prediction model, which has multiple outputs, from which first will be refined
planes: filter size used through the model architecture
num_classes: # of output classes
"""
def BASNet_RRM(base_model, planes, num_classes):
    base_model_input = base_model.input
    x_in = base_model.output[0]

    """-------------Encoder--------------"""
    x = layers.Conv2D(planes, kernel_size=(3, 3), strides=1, padding='same')(x_in)

    "encoder stage 1"
    e_stage_1 = conv_bn_relu(x, planes,)
    #  1 --> 1/2
    x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(e_stage_1)

    "encoder stage 2"
    e_stage_2 = conv_bn_relu(x, planes, )
    #  1/2 --> 1/4
    x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(e_stage_2)

    "encoder stage 3"
    e_stage_3 = conv_bn_relu(x, planes, )
    #  1/4 --> 1/8
    x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(e_stage_3)

    "encoder stage 4"
    e_stage_4 = conv_bn_relu(x, planes, )
    #  1/8 --> 1/16
    x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(e_stage_4)

    """-------------Bridge--------------"""

    bridge = conv_bn_relu(x, planes, )
    #  1/16 --> 1/8
    bridge = resize_bilinear(bridge, )

    """-------------Decoder--------------"""

    "decoder stage 4"
    x = layers.concatenate([e_stage_4, bridge], axis=-1)
    d_stage_4 = conv_bn_relu(x, planes)

    "decoder stage 3"
    #  1/8 --> 1/4
    x = resize_bilinear(d_stage_4, )
    x = layers.concatenate([e_stage_3, x], axis=-1)
    d_stage_3 = conv_bn_relu(x, planes)

    "decoder stage 2"
    #  1/4 --> 1/2
    x = resize_bilinear(d_stage_3, )
    x = layers.concatenate([e_stage_2, x], axis=-1)
    d_stage_2 = conv_bn_relu(x, planes)

    "decoder stage 1"
    #  1/2 --> 1
    x = resize_bilinear(d_stage_2, )
    x = layers.concatenate([e_stage_1, x], axis=-1)
    d_stage_1 = conv_bn_relu(x, planes)

    # segmentation head
    d_stage_1 = segmentation_head(d_stage_1, num_classes, None)  # 1
    
    # add prediction + refinement output
    d_stage_1 = x_in + d_stage_1

    model = models.Model(inputs=[base_model_input], outputs=[d_stage_1])

    # set weight initializers
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel_initializer = tf.keras.initializers.he_normal()
        if hasattr(layer, 'depthwise_initializer'):
            layer.depthwise_initializer = tf.keras.initializers.he_normal()

    return model


"""
BASNet, its a combination of two modules, prediction module and Residual Refinement Module(RRM)
input_shape: shape of model input tensor
num_classes: # of output classes
layers_arg: ResNet-34 block sizes at different layer
planes: filter size used through the model architecture
"""
def BASNet(input_shape=[256,256,3], num_classes=1,layers_arg=[3, 4, 6, 3], planes=64,):

    # create prediction model
    predict_model = BASNet_Predict(input_shape, layers_arg, num_classes, planes,)

    # create refinement model
    refine_model = BASNet_RRM(predict_model, planes, num_classes)

    # apply final activation on both models
    rrm_output, pred1_output, pred2_output, pred3_output,\
    pred4_output, pred5_output, pred6_output, pred7_output=\
        tf.nn.sigmoid(refine_model.output), tf.nn.sigmoid(predict_model.output[0]), \
        tf.nn.sigmoid(predict_model.output[1]), tf.nn.sigmoid(predict_model.output[2]),\
        tf.nn.sigmoid(predict_model.output[3]), tf.nn.sigmoid(predict_model.output[4]), \
        tf.nn.sigmoid(predict_model.output[5]), tf.nn.sigmoid(predict_model.output[6])

    final_model = models.Model(inputs=[predict_model.input], outputs=[
        rrm_output, pred1_output, pred2_output, pred3_output,pred4_output, pred5_output, pred6_output, pred7_output
    ])

    return final_model


if __name__ == "__main__":
    """## Model Compilation"""
    INPUT_SHAPE = [256, 256, 3]
    OUTPUT_CHANNELS = 1

    with tf.device("cpu:0"):
        # create model
        basnet_model = BASNet( input_shape =INPUT_SHAPE, num_classes=OUTPUT_CHANNELS,)
        optimizer = tf.keras.optimizers.SGD(momentum=0.9, lr=0.045)
        # compile model
        basnet_model.compile(
            loss=[iou_loss, bce_loss, ssim_loss],
            optimizer=optimizer,
            metrics=['accuracy'])
        # show model summary in output
        basnet_model.summary()

        # # save model architecture as png
        # tf.keras.utils.plot_model(basnet_model, show_layer_names=True, show_shapes=True)
        # # save model
        # basnet_model.save("basnet.hdf5")





