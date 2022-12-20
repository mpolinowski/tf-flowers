import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt

# download the MobileNet without the classification head
MobileNet_feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"
MobileNet_feature_extractor_layer = hub.KerasLayer(MobileNet_feature_extractor_url, input_shape=(224, 224, 3))
# freeze the feature extraction layer from mobilenet
MobileNet_feature_extractor_layer.trainable = False

# download and process training dataset
# specify path of the flowers dataset
flowers_data_url = tf.keras.utils.get_file(
  'flower_photos','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
   untar=True)
# pre-processing
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
flowers_data = image_generator.flow_from_directory(str(flowers_data_url), target_size=(224,224), batch_size = 64, shuffle = True)

# create input batches from flower data
for flowers_data_input_batch, flowers_data_label_batch in flowers_data:
  break

# build new model out of feature extraction layer
# and a fresh dense layer for us to train
model = tf.keras.Sequential([
  MobileNet_feature_extractor_layer,
  tf.keras.layers.Dense(flowers_data.num_classes, activation='softmax')
])

# build the model
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
# and train it with flower dataset
history = model.fit_generator(flowers_data, epochs=50)


# evaluate the model
## get classification labels from dataset
class_names = sorted(flowers_data.class_indices.items(), key=lambda pair:pair[1])
class_names = np.array([key.title() for key, value in class_names])
print(class_names)


# now we can feed a fresh batch of images to the trained model
predicted_batch = model.predict(flowers_data_input_batch)
# predict labels for each image inside the batch
predicted_id = np.argmax(predicted_batch, axis=-1)
predicted_label_batch = class_names[predicted_id]

label_id = np.argmax(flowers_data_input_batch, axis=-1)

# show images with labels
plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)

for n in range(64):
  plt.subplot(8,8,n+1)
  plt.tight_layout()
  plt.imshow(flowers_data_input_batch[n])
  ## use predicted label as title and use green/red to indicate if correct
  color = "green" if predicted_id[n].any() == label_id[n].any() else "red"
  plt.title(predicted_label_batch[n].title(), color=color)
  plt.axis('off')

plt.show()