import os

# specify the shape of the inputs for our network
IMG_SHAPE = (105, 105, 1)

# specify the batch size and number of epochs
BATCH_SIZE = 64
EPOCHS = 1000
LEARNING_RATE = 6e-5

# define the path to the base output directory
BASE_OUTPUT = "model"

# use the base output path to derive the path to the serialized
# model along with training history plot
MODEL_PATH = os.path.sep.join([BASE_OUTPUT, "siamese_model"])