import json
import os
import tempfile

import numpy as np
import requests

import seeds_classification.model_utils as utils
from seeds_classification import my_utils

homedir = os.path.dirname(os.path.realpath(__file__))

# Loading label names and label info files
synsets = np.genfromtxt(os.path.join(homedir, '..', 'data', 'data_splits', 'synsets.txt'), dtype='str', delimiter='/n')
try:
    synsets_info = np.genfromtxt(os.path.join(homedir, 'model_files', 'data', 'info.txt'), dtype='str', delimiter='/n')
except:
    synsets_info = np.array(['']*len(synsets))
assert synsets.shape == synsets_info.shape, """
Your info file should have the same size as the synsets file.
Blank spaces corresponding to labels with no info should be filled with some string (eg '-').
You can also choose to remove the info file."""

# Load training info
info_files = os.listdir(os.path.join(homedir, 'training_info'))
info_file_name = [i for i in info_files if i.endswith('.json')][0]
info_file = os.path.join(homedir, 'training_info', info_file_name)
with open(info_file) as datafile:
    train_info = json.load(datafile)
mean_RGB = train_info['augmentation_params']['mean_RGB']
output_dim = train_info['training_params']['output_dim']

# Load net weights
weights_files = os.listdir(os.path.join(homedir, 'training_weights'))
weights_file_name = [i for i in weights_files if i.endswith('.npz')][0]
test_func = utils.load_model(os.path.join(homedir, 'training_weights', weights_file_name), output_dim=output_dim)

def catch_error(f):
    def wrap(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            raise e
            return {"status": "error",
                    "predictions": []}
    return wrap


@catch_error
def predict_url(urls, test_func=test_func):
    if not isinstance(urls, list):
        urls = [urls]

    pred_lab, pred_prob = my_utils.single_prediction(test_func,
                                                     im_list=urls,
                                                     aug_params={'mean_RGB': mean_RGB,
                                                                 'filemode':'url'})
    return format_prediction(pred_lab, pred_prob)


@catch_error
def predict_file(filenames, test_func=test_func):
    if not isinstance(filenames, list):
        filenames = [filenames]

    pred_lab, pred_prob = my_utils.single_prediction(test_func,
                                                    im_list=filenames,
                                                    aug_params={'mean_RGB':
                                                                mean_RGB, 'filemode':'local'})
    return format_prediction(pred_lab, pred_prob)


@catch_error
def predict_data(images, test_func=test_func):
    if not isinstance(images, list):
        images = [images]

    filenames = []
    for image in images:
        f = tempfile.NamedTemporaryFile(delete=False)
        f.write(image)
        f.close()
        filenames.append(f.name)

    try:
        pred_lab, pred_prob = my_utils.single_prediction(test_func,
                                                        im_list=filenames,
                                                        aug_params={'mean_RGB':
                                                                    mean_RGB, 'filemode':'local'})
    except Exception as e:
        raise e
    finally:
        for f in filenames:
            os.remove(f)
    return format_prediction(pred_lab, pred_prob)


def format_prediction(labels, probabilities):
    d = {
        "status": "ok",
         "predictions": [],
    }

    for label_id, prob in zip(labels, probabilities):
        name = synsets[label_id]

        pred = {
            "label_id": label_id,
            "label": name,
            "probability": float(prob),
            "info": {
                "links": [{"link": 'Google images', "url": image_link(name)},
                          {"link": 'Wikipedia', "url": wikipedia_link(name)}],
                'metadata': synsets_info[label_id],
            },
        }
        d["predictions"].append(pred)
    return d


def image_link(pred_lab):
    """
    Return link to Google images
    """
    base_url = 'https://www.google.es/search?'
    params = {'tbm':'isch','q':pred_lab}
    link = base_url + requests.compat.urlencode(params)
    return link


def wikipedia_link(pred_lab):
    """
    Return link to wikipedia webpage
    """
    base_url = 'https://en.wikipedia.org/wiki/'
    link = base_url + pred_lab.replace(' ', '_')
    return link


def metadata():
    d = {
        "author": None,
        "description": None,
        "url": None,
        "license": None,
        "version": None,
    }
    return d
