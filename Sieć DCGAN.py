import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def convert_to_grayscale(x):
    x = tf.image.rgb_to_grayscale(x)
    return x / 255.0

dataset = keras.preprocessing.image_dataset_from_directory(
    directory="oct_crop/oct", label_mode=None, image_size=(256, 256), batch_size=64,
).map(convert_to_grayscale)

discriminator = keras.Sequential(
    [
        keras.Input(shape=(256, 256, 1)),
        layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Conv2D(256, kernel_size=4, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ],
    name="discriminator",
)
print(discriminator.summary())

latent_dim = 256
generator = keras.Sequential(
    [
        keras.Input(shape=(latent_dim,)),
        layers.Dense(16 * 16 * 256),
        layers.Reshape((16, 16, 256)),
        layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(alpha=0.2),
        layers.Conv2DTranspose(1, kernel_size=4, strides=2, padding="same", activation="sigmoid"),
    ],
    name="generator",
)
generator.summary()

opt_gen = keras.optimizers.Adam(1e-4)
opt_disc = keras.optimizers.Adam(1e-4)
loss_fn = keras.losses.BinaryCrossentropy()

log_dir = "logs/"
writer = tf.summary.create_file_writer(log_dir)


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=opt_gen,
                                 discriminator_optimizer=opt_disc,
                                 generator=generator,
                                 discriminator=discriminator)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

checkpoint.restore(manager.latest_checkpoint)
if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
else:
    print("Initializing from scratch.")


for epoch in range(10000):
    for idx, (real) in enumerate(tqdm(dataset)):
        batch_size = real.shape[0]
        with tf.GradientTape() as gen_tape:
            random_latent_vectors = tf.random.normal(shape = (batch_size, latent_dim))
            fake = generator(random_latent_vectors)

        if epoch % 20 == 0 and idx % 2 == 0:
            img = keras.preprocessing.image.array_to_img(fake[0])
            img.save("gen_images/generated_img_%03d_%d.png" % (epoch, idx))

        with tf.GradientTape() as disc_tape:
            loss_disc_real = loss_fn(tf.ones((batch_size, 1)), discriminator(real))
            loss_disc_fake = loss_fn(tf.zeros((batch_size, 1)), discriminator(fake))
            loss_disc = (loss_disc_real + loss_disc_fake)/2

        grads = disc_tape.gradient(loss_disc, discriminator.trainable_weights)
        opt_disc.apply_gradients(
            zip(grads, discriminator.trainable_weights)
        )

        with tf.GradientTape() as gen_tape:
            fake = generator(random_latent_vectors)
            output = discriminator(fake)
            loss_gen = loss_fn(tf.ones(batch_size, 1), output)

        grads = gen_tape.gradient(loss_gen, generator.trainable_weights)
        opt_gen.apply_gradients(zip(grads, generator.trainable_weights))

        if idx % 400 == 0:
            save_path = manager.save()
            print(f"Saved checkpoint for epoch {epoch+1}: {save_path}")

        with writer.as_default():
            tf.summary.scalar('loss_gen', loss_gen, step=epoch)
            tf.summary.scalar('loss_disc', loss_disc, step=epoch)
            writer.flush()




