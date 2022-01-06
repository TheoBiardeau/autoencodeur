import gzip
import numpy as np


# Fonction de décompression du format zip
def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(28 * 28 * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(num_images, 28, 28)


# Décompression et chargement des images
x_train = extract_data('train−images−idx3−ubyte.gz ', 60000)
x_test = extract_data('t10k-images-idx3-ubyte.gz', 10000)

#Normalisation des images
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

#vectorisation des images 28x28 en vecteur de 784 coeffs
x_train = x_train.reshape((len(x_train)), np.prod(x_train.shape[1:]))
x_test = x_test.reshape((len(x_test)), np.prod(x_test.shape[1:]))

print(x_train.shape)
print(x_test.shape)