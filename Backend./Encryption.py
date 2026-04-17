import numpy as np
import cv2
from tensorflow import keras
from tensorflow.keras import layers
import logging

# 1. CHAOTIC MAP FOR SEED GENERATION 
class ChaoticSeedGenerator:
    """Generate chaotic sequences for U-Net key generation"""

    @staticmethod
    def logistic_map(x0=0.5, r=3.99, n=10000):
        sequence = np.zeros(n, dtype=np.float32)
        sequence[0] = x0
        for i in range(1, n):
            sequence[i] = r * sequence[i-1] * (1 - sequence[i-1])
        return sequence

    @staticmethod
    def henon_map(x0=0.1, y0=0.1, a=1.4, b=0.3, n=10000):
        x = np.zeros(n, dtype=np.float32)
        y = np.zeros(n, dtype=np.float32)
        x[0], y[0] = x0, y0
        for i in range(1, n):
            x[i] = np.clip(1 - a * x[i-1]**2 + y[i-1], -10, 10)
            y[i] = b * x[i-1]
        return x, y

    @staticmethod
    def create_chaotic_seed_image(x0=0.1, y0=0.1, r0=0.5, size=64):
        # Generate chaotic sequences
        logistic_seq = ChaoticSeedGenerator.logistic_map(r0, n=size*size)
        henon_x, henon_y = ChaoticSeedGenerator.henon_map(x0, y0, n=size*size)

        combined = (logistic_seq + henon_x + henon_y) / 3.0

        # normalize
        cmin, cmax = combined.min(), combined.max()
        if cmax - cmin > 0:
            combined = (combined - cmin) / (cmax - cmin)
        else:
            combined = np.random.rand(size * size).astype(np.float32)

        seed_image = combined.reshape(size, size).astype(np.float32)
        return seed_image

# 2. U-NET KEY GENERATOR MODEL
class UNetKeyGenerator:
    def __init__(self, input_size=64, output_size=256):
        self.input_size = input_size
        self.output_size = output_size
        self.model = self.build_unet()

    def conv_block(self, inputs, filters, kernel_size=3):
        x = layers.Conv2D(filters, kernel_size, activation='relu',
                          padding='same', kernel_initializer='he_normal')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, kernel_size, activation='relu',
                          padding='same', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        return x

    def build_unet(self):
        inputs = layers.Input(shape=(self.input_size, self.input_size, 1))

        conv1 = self.conv_block(inputs, 64)
        pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = self.conv_block(pool1, 128)
        pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = self.conv_block(pool2, 256)
        pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = self.conv_block(pool3, 512)
        pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

        bottleneck = self.conv_block(pool4, 1024)

        up5 = layers.UpSampling2D(size=(2, 2))(bottleneck)
        up5 = layers.Conv2D(512, 2, activation='relu', padding='same')(up5)
        merge5 = layers.concatenate([conv4, up5], axis=3)
        conv5 = self.conv_block(merge5, 512)

        up6 = layers.UpSampling2D(size=(2, 2))(conv5)
        up6 = layers.Conv2D(256, 2, activation='relu', padding='same')(up6)
        merge6 = layers.concatenate([conv3, up6], axis=3)
        conv6 = self.conv_block(merge6, 256)

        up7 = layers.UpSampling2D(size=(2, 2))(conv6)
        up7 = layers.Conv2D(128, 2, activation='relu', padding='same')(up7)
        merge7 = layers.concatenate([conv2, up7], axis=3)
        conv7 = self.conv_block(merge7, 128)

        up8 = layers.UpSampling2D(size=(2, 2))(conv7)
        up8 = layers.Conv2D(64, 2, activation='relu', padding='same')(up8)
        merge8 = layers.concatenate([conv1, up8], axis=3)
        conv8 = self.conv_block(merge8, 64)

        up9 = layers.UpSampling2D(size=(2, 2))(conv8)
        conv9 = self.conv_block(up9, 32)

        up10 = layers.UpSampling2D(size=(2, 2))(conv9)
        conv10 = self.conv_block(up10, 16)

        outputs = layers.Conv2D(1, 1, activation='tanh', kernel_initializer='he_normal')(conv10)
        model = keras.Model(inputs=inputs, outputs=outputs, name='UNet_KeyGen')
        return model

    def compile_model(self, learning_rate=0.0001):
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss='mse',
            metrics=['mae']
        )

    def train(self, chaotic_seeds, target_keys, epochs=100, batch_size=8):
        self.compile_model()
        history = self.model.fit(
            chaotic_seeds,
            target_keys,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.15,
            verbose=1
        )
        return history

    def generate_key(self, chaotic_seed):
        # ensure shape (1, H, W, 1)
        if len(chaotic_seed.shape) == 2:
            chaotic_seed = chaotic_seed[..., np.newaxis]
        if len(chaotic_seed.shape) == 3:
            chaotic_seed = chaotic_seed[np.newaxis, ...]
        chaotic_seed = chaotic_seed.astype(np.float32)
        key_matrix = self.model.predict(chaotic_seed, verbose=0)
        return key_matrix[0, :, :, 0]

    def save_model(self, filepath):
        self.model.save(filepath)

    def load_model(self, filepath):
        self.model = keras.models.load_model(filepath)

# 3. ENCRYPTION USING U-NET GENERATED KEY
class ImageEncryptor:
    def __init__(self, key_generator: UNetKeyGenerator):
        self.key_generator = key_generator

    def permute_pixels(self, image, key_matrix):
        h, w = image.shape[:2]
        key_sum = int(np.sum(np.abs(key_matrix)) * 1e6) % (2**32 - 1)
        np.random.seed(key_sum)
        perm_indices = np.random.permutation(h * w)
        return perm_indices

    def xor_diffusion(self, image, key_matrix):
        h, w, c = image.shape
        if key_matrix.shape != (h, w):
            key_resized = cv2.resize(key_matrix, (w, h))
        else:
            key_resized = key_matrix
        key_uint8 = ((key_resized + 1) * 127.5).astype(np.uint8)
        encrypted = np.zeros_like(image, dtype=np.uint8)
        for channel in range(c):
            encrypted[:, :, channel] = np.bitwise_xor(image[:, :, channel].astype(np.uint8), key_uint8)
        return encrypted

    def encrypt(self, image, chaotic_params=(0.1, 0.1, 0.5)):
        x0, y0, r0 = chaotic_params
        h, w, c = image.shape
        seed_image = ChaoticSeedGenerator.create_chaotic_seed_image(x0, y0, r0, size=64)
        key_matrix = self.key_generator.generate_key(seed_image)
        key_resized = cv2.resize(key_matrix, (w, h))
        perm_indices = self.permute_pixels(image, key_resized)
        encrypted = np.zeros_like(image, dtype=np.uint8)
        for channel in range(c):
            flat = image[:, :, channel].flatten()
            confused = flat[perm_indices]
            encrypted[:, :, channel] = confused.reshape(h, w)
        encrypted = self.xor_diffusion(encrypted, key_resized)
        return encrypted, key_matrix, perm_indices, chaotic_params

# 4. DECRYPTION
class ImageDecryptor:
    def __init__(self, key_generator: UNetKeyGenerator):
        self.key_generator = key_generator

    def decrypt(self, encrypted_image, chaotic_params, perm_indices):
        x0, y0, r0 = chaotic_params
        h, w, c = encrypted_image.shape
        seed_image = ChaoticSeedGenerator.create_chaotic_seed_image(x0, y0, r0, size=64)
        key_matrix = self.key_generator.generate_key(seed_image)
        key_resized = cv2.resize(key_matrix, (w, h))
        key_uint8 = ((key_resized + 1) * 127.5).astype(np.uint8)
        de_diffused = np.zeros_like(encrypted_image, dtype=np.uint8)
        for channel in range(c):
            de_diffused[:, :, channel] = np.bitwise_xor(encrypted_image[:, :, channel].astype(np.uint8), key_uint8)
        decrypted = np.zeros_like(de_diffused, dtype=np.uint8)
        total_pixels = h * w
        for channel in range(c):
            flat_diffused = de_diffused[:, :, channel].flatten()
            de_confused = np.zeros(total_pixels, dtype=np.uint8)
            de_confused[perm_indices] = flat_diffused
            decrypted[:, :, channel] = de_confused.reshape(h, w)
        return decrypted

# 5. UTILS (entropy/npcr/uaci)
def calculate_entropy(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    return float(entropy)

def calculate_npcr(img1, img2):
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    diff = np.sum(img1 != img2)
    npcr = (diff / img1.size) * 100
    return float(npcr)

def calculate_uaci(img1, img2):
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    diff = np.abs(img1.astype(float) - img2.astype(float))
    uaci = np.mean(diff) / 255 * 100
    return float(uaci)
