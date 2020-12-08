import tensorflow as tf


# jieduan chu ziji xuyao de moxing
# model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)



basemodel = tf.keras.applications.VGG16(weights = 'imagenet')
# take out all layers in Model
print([layer.output for layer in basemodel.layers])

# feat_extraction_model = keras.Model(inputs=vgg19.input, outputs=features_list)
