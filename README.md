# Construct GAN via configuration

This repository is to implement an easy way to build your own GAN just by editing configuration file.

There are some examples in [templates](templates) folder, including MNIST, face generation.

### Prerequisites:
[Keras](https://keras.io), [TensorFlow](https://www.tensorflow.org)

### How to use?

Takes generate MNIST as an example, the NN is defined in [mnist.json](./templates/mnist.json).
 (Please refer to the [template file format](templates/README.md).)

```
import tensorflow as tf

# import GAN
from GAN import GAN

# construct GAN
mnistGAN = GAN("mnist.json")

# compile GAN model
mnistGAN.compile()

# get session and initialize global variables
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# train GAN model
for epoch in range(20000):
    # prepare batch_images for each iteration.
    # in the example, discriminator train twice per iteration.

    loss_g, loss_d = mnistGAN.batch_evaluate(sess, batch_images, iter_dis=2)

    # You can display some information during training to monitor progress.
    # For example, by displaying generated images, you can see how good it is.
    # You can also display generator and discriminator losses.
    # Displaying discriminator outputs distribution for both real and generated images may help for performance tuning.

# save model (this is keras json format) if you want.
mnistGAN.save_generator_model("mnist_generator.json")
mnistGAN.save_discriminator_model("mnist_discriminator.json")

# save model weights
mnistGAN.save_genenerator_weights("mnist_generator.h5")
mnistGAN.save_discriminator_weights("mnist_discriminatorr.h5")

# generate new images
# 1. pick some random vectors with the same dimension as "encode_size" in mnist.json.
CODE_SIZE = 2
num_images = 9
codes = np.random.normal(size=(num_images, CODE_SIZE)).astype('float32')

# 2. generate these images
fake_images = mnistGAN.getGenerator().predict(codes)
```

You can load trained model and weights, and generate images later.

```
# construct empty model
gan = GAN()

# load model
gan.load_model("mnist_generator.json", "mnist_discriminator.json")

# load generator weights
gan.load_generator_weights("mnist_generator.h5")

# generate new images
CODE_SIZE = 2
num_images = 9
codes = np.random.normal(size=(num_images, CODE_SIZE)).astype('float32')
images = gan.getGenerator().predict(codes)
```
