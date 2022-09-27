from __future__ import print_function

import math

try:
    raw_input
except:
    raw_input = input

import numpy as np
from matplotlib import gridspec
import pickle
import time
import datetime
import os
from PIL import Image
import json
import sys
import argparse
import imageio

imsave = imageio.imsave
import itertools
import copy
import pywt
from pywt import dwt2, wavedec2
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions

RESNET_MEAN = np.array([103.939, 116.779, 123.68])


def preprocess(sample_path):
    sample_path = 'imagenet/' + sample_path
    img = image.load_img(sample_path, target_size=(224, 224))
    x = image.img_to_array(img)
    return x


def preprocess_target(sample_path):
    img = image.load_img(sample_path, target_size=(224, 224))
    x = image.img_to_array(img)
    return x


def image_process_for_save(x):
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def SLIA(model,
                  sample,
                  clip_max=1,
                  clip_min=0,
                  constraint='l2',
                  num_iterations=76,
                  gamma=1.0,
                  target_label=None,
                  target_sample=None,
                  max_num_evals=1e4,
                  init_num_evals=100,
                  queries=0,
                  verbose=True):
    data_model = 'imagenet+resnet50'
    params = {'clip_max': clip_max, 'clip_min': clip_min,
              'target_label': target_label,
              'constraint': constraint,
              'num_iterations': num_iterations,
              'gamma': gamma,
              'd': int(np.prod(sample.shape)) / 4,
              'max_num_evals': max_num_evals,
              'init_num_evals': init_num_evals,
              'queries': queries,
              'verbose': verbose,
              }
    # Set binary search threshold.
    if params['constraint'] == 'l2':
        params['theta'] = params['gamma'] / params['d']
    else:
        params['theta'] = params['gamma'] / (params['d'] ** 2)

    params['sample'] = sample.astype('float32') / 255.0
    params['shape'] = sample.shape
    if target_sample is not None:
        params['target_sample'] = target_sample.astype('float32') / 255.0
    else:
        params['target_sample'] = None
    sample = np.expand_dims(sample, axis=0)
    sample = preprocess_input(sample)  # RGB to GBR.
    params['original_label'] = np.argmax(model.predict(sample))  # in order to get the original_label.
   
    # Initialize.
    if params['target_sample'] is None:
       perturbed = initialize(model, params['sample'], params)
    else:
       perturbed = params['target_sample']
    dist = compute_distance(perturbed, params['sample'], constraint)
    print(dist)
    # Project the initialization to the boundary.
    perturbed, dist_post_update = binary_search_batch(params['sample'],                                                      np.expand_dims(perturbed, 0),
                                                      model,
                                                      params)

    dist = compute_distance(perturbed, params['sample'], constraint)
    print(dist)
    for j in np.arange(params['num_iterations']):
        params['cur_iter'] = j + 1

        # Choose delta.
        delta = select_delta(params, dist_post_update)

        # Choose number of evaluations.
        num_evals = int(params['init_num_evals'] * np.power(j + 1, 1/4))

        # approximate gradient.
        gradf = approximate_gradient_DWT(model, perturbed, num_evals,
                                          delta, params)
        if params['constraint'] == 'linf':
            update = np.sign(gradf)
        else:
            update = gradf

        # search step size.
        epsilon = geometric_progression_for_stepsize_DWT(perturbed,
                                                     update, dist, model, params)

        # Update the sample.
        perturbed = clip_image(params['adversarial_example'],
                               clip_min, clip_max)

        # Binary search to return to the boundary.
        perturbed, dist_post_update = binary_search_batch(params['sample'],
                                                          perturbed[None], model, params)

        # compute new distance.
        dist = compute_distance(perturbed, params['sample'], constraint)
        if verbose:
            print('iteration: {:d}, l2 distance {:.4E},queries: {:d}'.format(j + 1, dist, params['queries']))
            perturbed_image2 = np.copy(perturbed) * 255
            imsave('{}/DWT/{}.jpg'.format(data_model, j+1), perturbed_image2.astype(np.uint8))

def approximate_gradient_DWT(model, sample, num_evals, delta, params):
    clip_max, clip_min = params['clip_max'], params['clip_min']
    # sample_points' shape = (100, 224 ,224, 3)
    sample_points_shape = [num_evals] + list((224, 224, 3))
    sample_points = np.random.randn(*sample_points_shape)
    # low_fre_perturbs' shape = (100, 112 ,112, 3)
    low_fre_perturbs_shape = [num_evals] + list((112, 112, 3))  
    low_fre_perturbs = np.random.randn(*low_fre_perturbs_shape)
    for i in range(3):

        coeffs = dwt2(sample[:, :, i], 'haar')
        cA, (cH, cV, cD) = coeffs
        noise_shape = [num_evals] + list(cA.shape)
        if params['constraint'] == 'l2':
            rv = np.random.randn(*noise_shape)
        elif params['constraint'] == 'linf':
            rv = np.random.uniform(low=-1 * np.mean(cA), high=1 * np.mean(cA), size=noise_shape)
        rv = rv / np.sqrt(np.sum(rv ** 2, axis=(1, 2), keepdims=True))
        # rv's shape = (100, 112 ,112)
        t = rv * np.mean(cA)
        new_cA = cA + t
        low_fre_perturbs[:, :, :, i] = rv
        for j in range(num_evals):       
            coeffs = new_cA[j], (cH, cV, cD)
            sample_points[j, :, :, i] = pywt.idwt2(coeffs, 'haar')

    # query the model.
    decisions = decision_function(model, sample_points, params)
    params['queries'] += num_evals - 1
    decision_shape = [len(decisions)] + [1] * len(params['shape'])
    fval = 2 * decisions.astype(float).reshape(decision_shape) - 1.0

    # Baseline subtraction (when fval differs)
    if np.mean(fval) == 1.0:  # label changes.
        gradf = np.mean(low_fre_perturbs, axis=0)
    elif np.mean(fval) == -1.0:  # label not change.
        gradf = - np.mean(low_fre_perturbs, axis=0)
    else:
        fval -= np.mean(fval)
        gradf = np.mean(fval * low_fre_perturbs, axis=0)

    # Get the gradient direction.
    gradf = gradf / np.linalg.norm(gradf)

    return gradf


def decision_function(model, image, params):
    """
    Decision function output 1 on the desired side of the boundary,
    0 otherwise.
    """
    params['queries'] += 1
    image = clip_image(image, params['clip_min'], params['clip_max'])
    image = image.astype('float') * 255.0
    image = preprocess_input(image)
    prob = model.predict(image)
    if params['target_label'] is None:
        return np.argmax(prob, axis=1) != params['original_label']
    else:
        return np.argmax(prob, axis=1) == params['target_label']


def clip_image(image, clip_min, clip_max):
    # Clip an image, or an image batch, with upper and lower threshold.
    return np.minimum(np.maximum(clip_min, image), clip_max)


def compute_distance(x1, x2, constraint='l2'):
    # Compute the distance between two images.
    if constraint == 'l2':
        return np.linalg.norm(x1 - x2)
    elif constraint == 'linf':
        return np.max(abs(x1 - x2))


def project(original_image, perturbed_images, alphas, params):
    alphas_shape = [len(alphas)] + [1] * len(params['shape'])
    alphas = alphas.reshape(alphas_shape)
    if params['constraint'] == 'l2':
        return (1 - alphas) * original_image + alphas * perturbed_images
    elif params['constraint'] == 'linf':
        out_images = clip_image(
            perturbed_images,
            original_image - alphas,
            original_image + alphas
        )
        return out_images


def binary_search_batch(original_image, perturbed_images, model, params):
    """ Binary search to approach the boundar. """

    # Compute distance between each of perturbed image and original image.
    dists_post_update = np.array([
        compute_distance(
            original_image,
            perturbed_image,
            params['constraint']
        )
        for perturbed_image in perturbed_images])

    # Choose upper thresholds in binary searchs based on constraint.
    if params['constraint'] == 'linf':
        highs = dists_post_update
        # Stopping criteria.
        thresholds = np.minimum(dists_post_update * params['theta'], params['theta'])
    else:
        highs = np.ones(len(perturbed_images))
        thresholds = params['theta']

    lows = np.zeros(len(perturbed_images))

    # Call recursive function.
    while np.max((highs - lows) / thresholds) > 1:
        # projection to mids.
        mids = (highs + lows) / 2.0
        mid_images = project(original_image, perturbed_images, mids, params)

        # Update highs and lows based on model decisions.
        mid_images = np.squeeze(mid_images)
        decisions = decision_function(model, mid_images[None], params)

        lows = np.where(decisions == 0, mids, lows)
        highs = np.where(decisions == 1, mids, highs)

    out_images = project(original_image, perturbed_images, highs, params)

    # Compute distance of the output image to select the best choice.
    # (only used when stepsize_search is grid_search.)
    dists = np.array([
        compute_distance(
            original_image,
            out_image,
            params['constraint']
        )
        for out_image in out_images])
    idx = np.argmin(dists)

    dist = dists_post_update[idx]
    out_image = out_images[idx]
    return out_image, dist


def initialize(model, sample, params):
    sample_point = np.zeros_like(sample)
    low_fre_perturb_shape = list((112, 112)) 
    while(1):
      for i in range(3):
          coeffs = dwt2(sample[:, :, i], 'haar')
          cA, (cH, cV, cD) = coeffs
          new_cA = np.random.uniform(cA.min(), cA.max(), size=low_fre_perturb_shape)
          coeffs = new_cA, (cH, cV, cD)
          sample_point[:, :, i] = pywt.idwt2(coeffs, 'haar')

      # query the model.
      decision = decision_function(model, sample_point[None], params)
      if decision:
          print('adversarial..........')
          break;
    dist = compute_distance(sample_point, sample, params['constraint'])
    print(dist)

    low = 0.0
    high = 1.0
    while high - low > 0.001:
        mid = (high + low) / 2.0
        blended = (1 - mid) * sample + mid * sample_point
        success = decision_function(model, blended[None], params)
        if success:
            high = mid
        else:
            low = mid
    perturbed_image = (1 - high) * sample + high * sample_point
    dist = compute_distance(perturbed_image, sample, params['constraint'])
    perturbed_image2 = np.copy(perturbed_image) * 255
    imsave('{}/DWT/{}.jpg'.format(data_model, 0), perturbed_image2.astype(np.uint8))
    print('iteration: {:d}, {:s} distance {:.4E},queries: {:d}'.format(1, params['constraint'], dist, params['queries']))
    return perturbed_image

def geometric_progression_for_stepsize_DWT(x, update, dist, model, params):
    """
	Geometric progression to search for stepsize.
	Keep decreasing stepsize by half until reaching
	the desired side of the boundary,
	"""
    epsilon = dist / np.sqrt(params['cur_iter']) * 4

    def phi(epsilon):
        # new = x + epsilon * update
        new_sample = np.zeros_like(x)
        for i in range(3):

            cA, (cH, cV, cD) = dwt2(x[:, :, i], 'haar')

            # coeffs = wavedec2(x[:, :, i], 'haar', level=2)
            # cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs
            new_cA = cA + epsilon * update[:, :, i]
            
            coeffs = new_cA, (cH, cV, cD)
            new_sample[:, :, i] = pywt.idwt2(coeffs, 'haar')

        success = decision_function(model, new_sample[None], params)
        params['adversarial_example'] = new_sample
        return success

    while not phi(epsilon):
        epsilon /= 2.0

    return epsilon


def select_delta(params, dist_post_update):
    """
    Choose the delta at the scale of distance
    between x and perturbed sample.

    """
    if params['cur_iter'] == 1:
        delta = 0.1 * (params['clip_max'] - params['clip_min'])
    else:
        if params['constraint'] == 'l2':
            delta = np.sqrt(params['d']) * params['theta'] * dist_post_update * 4
        elif params['constraint'] == 'linf':
            delta = params['d'] * params['theta'] * dist_post_update * 4

    return delta


if __name__ == "__main__":
    model = ResNet50(weights='imagenet')
    target_sample = preprocess_target('images/original/290.00002646.jpg')
    temp_sample = np.copy(target_sample)
    temp_sample = np.expand_dims(temp_sample, axis=0)
    temp_sample = preprocess_input(temp_sample)  # RGB to GBR
    target_label = np.argmax(model.predict(temp_sample))  # get the label of the target image.
    filepath = os.listdir('imagenet')
    for i, sample in enumerate(filepath):
        sample = preprocess(sample)

        parser = argparse.ArgumentParser()

        parser.add_argument('--constraint', type=str,
                            choices=['l2', 'linf'],
                            default='l2')

        parser.add_argument('--attack_type', type=str,
                            choices=['targeted', 'untargeted'],
                            default='untargeted')

        parser.add_argument('--num_iterations', type=int,
                            default=76)

        args = parser.parse_args()
        dict_a = vars(args)
        if args.attack_type == 'targeted':
            target_label = target_label
            target_sample = target_sample
        else:
            target_label = None
            target_sample = None
        print('attacking the {}th sample...'.format(i))
        SLIA(model,
                      sample,
                      clip_max=1,
                      clip_min=0,
                      constraint=args.constraint,
                      num_iterations=args.num_iterations,
                      gamma=1.0,
                      target_label=target_label,
                      target_sample=target_sample,
                      max_num_evals=1e4,
                      init_num_evals=100,
                      queries=0)

