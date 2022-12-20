import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import matplotlib.pyplot as plt

# import mobile net with trained weights
Trained_MobileNet_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"

Trained_MobileNet = tf.keras.Sequential([
    hub.KerasLayer(Trained_MobileNet_url, input_shape=(224,224,3))])




# # test the raw mobilenet model
# sample_image = tf.keras.preprocessing.image.load_img(r'./test_images/kitten.jpg', target_size = (224, 224))

# # pre-process image according to model requirements
# sample_image = np.array(sample_image)/255.0
# predicted_class = Trained_MobileNet.predict(np.expand_dims(sample_image, axis = 0))
# predicted_class = np.argmax(predicted_class)

# download the imagenet labels = mobilenet classes
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

# # show image with predicted class
# plt.imshow(sample_image)
# predicted_class_name = imagenet_labels[predicted_class]
# plt.title("Prediction: " + predicted_class_name.title())
# plt.show()


# download and process training dataset
# specify path of the flowers dataset
flowers_data_url = tf.keras.utils.get_file(
  'flower_photos','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
   untar=True)
# pre-processing
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
flowers_data = image_generator.flow_from_directory(str(flowers_data_url), target_size=(224,224), batch_size = 64, shuffle = True)


for flowers_data_input_batch, flowers_data_label_batch in flowers_data:
  # check how many images / classes => Found 3670 images belonging to 5 classes.
  # print("Image batch shape: ", flowers_data_input_batch.shape)
  # print("Label batch shape: ", flowers_data_label_batch.shape)
  break


predictions_batch = Trained_MobileNet.predict(flowers_data_input_batch)
# We have a batch size of 64 with 1000 classes
# print(predictions_batch.shape)
# get classnames for imagenet classes
predicted_class_names = imagenet_labels[np.argmax(predictions_batch, axis=-1)]
# print(predicted_class_names)

# print all 64 images
# and add predicted class
plt.figure(figsize=(15,15))

for n in range(64):
  plt.subplot(8,8,n+1)
  plt.tight_layout()
  plt.imshow(flowers_data_input_batch[n])
  plt.title(predicted_class_names[n])
  plt.axis('off')
# performance is poor without training
plt.show()