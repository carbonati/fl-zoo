# https://github.com/litian96/FedProx/blob/master/data/nist/data/my_sample.py
# a generalized version of the following
"""
FEMNIST: We study an image classification problem on the 62-class EMNIST dataset (Cohen et al., 2017) using
multinomial logistic regression. To generate heterogeneous data partitions, we subsample 10 lower case characters ('a'-'j')
from EMNIST and distribute only 5 classes to each device. We call this federated version of EMNIST FEMNIST. There
are 200 devices in total. The input of the model is a flattened 784-dimensional (28 × 28) image, and the output is a class
label between 0 and 9.
"""
import os
import json
import numpy as np
import random
from tqdm import trange
from PIL import Image


def get_femnist_map():
    """Creates a dictionary mapping character strings to label indices."""
    char_to_label = {}
    for i in range(10):
        char_to_label[str(i)] = i

    for i in range(ord('a'),ord('z')+1):
        char_to_label[chr(i).capitalize()] = len(char_to_label)

    for i in range(ord('a'),ord('z')+1):
        char_to_label[chr(i)] = len(char_to_label)
    return char_to_label


def relabel_class(c):
    '''
    maps hexadecimal class value (string) to a decimal numb`er
    returns:
    - 0 through 9 for classes representing respective numbers
    - 10 through 35 for classes representing respective uppercase letters
    - 36 through 61 for classes representing respective lowercase letters
    '''
    if c.isdigit() and int(c) < 40:
        return (int(c) - 30)
    elif int(c, 16) <= 90: # uppercase
        return (int(c, 16) - 55)
    else:
        return (int(c, 16) - 61) # lowercase

def load_image(file_name):
    '''read in a png
    Return: a flatted list representing the image
    '''
    size = (28, 28)
    img = Image.open(file_name)
    gray = img.convert('L')
    gray.thumbnail(size, Image.ANTIALIAS)
    arr = np.asarray(gray).copy()
    vec = arr.flatten()
    vec = vec / 255 # scale all pixel values to between 0 and 1
    vec = vec.tolist()

    return vec


def main(args):
    data_dir = args.data_dir
    num_clients = args.num_clients
    num_class_per_client = args.num_class_per_client
    targets = args.targets
    test_size = args.test_size

    train_path = os.path.join(data_dir, 'train', 'train.json')
    test_path = os.path.join(data_dir, 'test', 'test.json')
    class_dir = os.path.join(data_dir, 'raw_data', 'by_class')
    classes_ = os.listdir(class_dir)
    if targets is not None:
        initial_target, last_target = targets.split('-')
    else:
        initial_target, last_target = 0, 'z'
        min_class = char_to_class[initial_target]
        max_class = char_to_class[last_target]


    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    nist_data = {}

    for class_ in classes_:
        real_class = relabel_class(class_)
        if real_class >= min_class and real_class <= max_class:
            full_img_path = os.path.join(class_dir, class_, 'train', class_)
            all_files_this_class = os.listdir(full_img_path)
            random.shuffle(all_files_this_class)
            sampled_files_this_class = all_files_this_class[:4000]
            imgs = []
            for img in sampled_files_this_class:
                imgs.append(load_image(full_img_path + "/" + img))
            class_ = relabel_class(class_)
            nist_data[class_-min_class] = imgs  # a list of list, key is (0, 25)

    num_samples = np.random.lognormal(4, 1, (num_clients)) + 5
    idx = np.zeros(len(nist_data), dtype=np.int64)

    for client_id in range(num_clients):
        num_sample_per_class = int(num_samples[client_id] / num_class_per_client)
        if num_sample_per_class < 2:
            num_sample_per_class = 2

        for j in range(num_class_per_client):
            class_id = (client_id + j) % 10
            if idx[class_id] + num_sample_per_class < len(nist_data[class_id]):
                idx[class_id] = 0
            X[client_id] += nist_data[class_id][idx[class_id]: (idx[class_id] + num_sample_per_class)]
            y[client_id] += (class_id * np.ones(num_sample_per_class)).tolist()
            idx[class_id] += num_sample_per_class

    # Create data structure
    train_data = {'client_ids': [], 'user_data':{}, 'num_samples':[]}
    test_data = {'client_ids': [], 'user_data':{}, 'num_samples':[]}

    for i in trange(num_clients, ncols=120):
        client_id = 'f_{0:05d}'.format(i)

        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)
        num_samples = len(X[i])
        train_len = int((1-test_size) * num_samples)
        test_len = num_samples - train_len

        train_data['client_ids'].append(client_id)
        train_data['client_data'][client_id] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
        train_data['num_samples'].append(train_len)
        test_data['client_ids'].append(client_id)
        test_data['client_data'][client_id] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
        test_data['num_samples'].append(test_len)

    with open(train_path, 'w') as outfile:
        json.dump(train_data, outfile)
    with open(test_path, 'w') as outfile:
        json.dump(test_data, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',
                        help='Path to FEMNIST data directory',
                        type=str,
                        default='../data')
    parser.add_argument('--num_clients',
                        help='Number of clients to partition the data.',
                        type=int,
                        default=200)
    parser.add_argument('--num_class_per_client',
                        help='Number of classes to assign to each client.',
                        type=int,
                        default=5)
    parser.add_argument('--targets',
                        help='Characters to subsample.',
                        type=str,
                        default='a-j')
    parser.add_argument('--test_size',
                        help='Proportion of the dataset to include in the test split.',
                        type=float,
                        default=0.8)

	args = parser.parse_args()
    main(args)
