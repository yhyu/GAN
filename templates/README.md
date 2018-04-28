### template file format

A template file looks like:
```
{
    "image_shape": [36, 36, 3],
    "encode_size": 256,
    "generator": {
        "optimizer": {
            "name": "Adam",
            "learning_rate": 0.0001
        },
        "layers":[
        ...
        ]
    },
    "discriminator": [
        "optimizer": {
            "name": "SGD",
            "learning_rate": 0.001
        },
        "layers": [
        ...
        ]
    }
}
```

"image_shape": images height, width, and depth.

"encode_size": generator input vector dimentions.

"generator": defines generator NN architecture.

"discriminator": defines discriminator NN architecture.


Sub-items in "generator" and "discriminator":

"optimizer": defines optimizer for training steps, only supports "Adam" and "SGD"

"layers": defines a list of layers to construct NN.

**Notes**: You don't need define input layer and discriminator classfication layer.

Supports layer types are "Reshape", "Dense", "Flatten", "BatchNormalization", "Dropout",
 "Conv2D", "MaxPooling2D", "AveragePooling2D", and "Conv2DTranspose".
