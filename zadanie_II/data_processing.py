import os
import numpy as np
import cv2
import imutils
import matplotlib.pyplot as plt
import sys
import random

def imshow(image):
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

def read_imgs_from_dir(path_to_dir):
    images = []
    for filename in os.listdir(path_to_dir):
        image_path = os.path.join(path_to_dir, filename)
        image = cv2.imread(image_path, 0)
        images.append(image)

    return images

def load_images(path):
    X = []
    y = []
    curr_y = 0

    # read all entities in data folder
    for entity in os.listdir(path):
        images_path = os.path.join(path, entity)

        # read all the images of each entity
        for filename in os.listdir(images_path):
            image_path = os.path.join(images_path, filename)
            image = cv2.imread(image_path, 0)
            X.append(image)
            y.append(curr_y)
                 
        curr_y += 1
    
    y = np.vstack(y)
    X = np.stack(X)

    return X, y.ravel()

def make_pairs(images, labels, negatives=8):
	# initialize two empty lists to hold the (image, image) pairs and
	# labels to indicate if a pair is positive or negative
	pair_images = []
	pair_labels = []
	
	numClasses = len(np.unique(labels))

	# create list of arrays with same label
	pos_label_list = [np.where(labels == i)[0] for i in range(0, numClasses)]

	# loop over all images
	for image in range(len(images)):
		currentImage = images[image]
		label = labels[image]

		"""
		positive pair
		"""
		# take all images with same label
		for same_label in pos_label_list[label]:
			image_pos = images[same_label]

			# append positive pair and update label to 1
			pair_images.append([currentImage, image_pos])
			pair_labels.append([1])

		"""
		negative pair
		"""
		neg_label_list = np.where(labels != label)[0]
		diff_labels = random.choices(neg_label_list, k=negatives) 
		
		for diff_label in diff_labels:
			image_neg = images[diff_label]

			# append negative pair and update label to 0
			pair_images.append([currentImage, image_neg])
			pair_labels.append([0])

	return (np.array(pair_images), np.array(pair_labels))

def stack_pairs(pairTrain, labelTrain):
    images = []

    for i in np.random.choice(np.arange(0, len(pairTrain)), size=(49,)):
        # grab the current image pair and label
        imageA = pairTrain[i][0]
        imageB = pairTrain[i][1]
        label = labelTrain[i]

        padding = 4
        w, h = imageA.shape

        output = np.zeros((w+2*padding, 2*w+padding), dtype="uint8")
        pair = np.hstack([imageA, imageB])
        output[padding:w+padding, 0:2*w] = pair

        text = "-" if label[0] == 0 else "+"
        color = (0, 0, 255) if label[0] == 0 else (0, 255, 0)

        # create a 3-channel RGB image from the grayscale pair,
        vis = cv2.merge([output] * 3)
        cv2.putText(vis, text, (4, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
            color, 3)
        images.append(vis)
    
    return images

def pairs_montage(X_train, y_train):
    montage = imutils.build_montages(stack_pairs(X_train, y_train), (192, 102), (7, 7))[0]
    imshow(montage)

def draw_evaluation(sample, s_class, similarity):
    plt.figure(figsize=(4, 6))
    plt.title("Similarity: {:.2f}".format(similarity))
    pair = np.hstack((sample, s_class))
    plt.imshow(pair, cmap=plt.cm.gray)
    plt.show()

def evaluate_data(model, path_classes, path_to_sample):
    # read sample
    sample = image = cv2.imread(path_to_sample, 0)
    if sample is None:
        sys.exit("Could not read the image.")

    sample_orig = sample.copy()

    # expand image dimensions, add 1 channel and batch dimension
    sample = np.expand_dims(sample, axis=-1)
    sample = np.expand_dims(sample, axis=0)

    # read classes we will compare our sample to
    classes = read_imgs_from_dir(path_classes)

    for single_class in classes:
        single_class_orig = single_class.copy()

        single_class = np.expand_dims(single_class, axis=-1)
        single_class = np.expand_dims(single_class, axis=0)

        # evaluate similarity
        prediction = model.predict([sample, single_class])
        similarity = prediction[0][0]
        draw_evaluation(sample_orig, single_class_orig, similarity)
        print(similarity)

def show_positive_train(pairs, labels, marker):
    num_labels = len(labels)
    counter = 0
    for i in range(num_labels):
        if labels[i] == marker:
            pair = np.hstack((pairs[i][0], pairs[i][1]))

            counter += 1
            plt.figure(figsize=(4, 6))
            plt.title(f"{counter}.")
            plt.imshow(pair, cmap=plt.cm.gray)