import os
import math
import pickle
import random
import numpy as np
import glob
import torch
import cv2
import numbers
import copy
from sklearn.metrics import auc

####################
# Files & IO
####################

###################### get image path list ######################
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']
LANDMARK_EXTENSIONS = ['.txt']

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def is_landmark_file(filename):
    return any(filename.endswith(extension) for extension in LANDMARK_EXTENSIONS)


def load_txt_file(file_path):
    '''
    load data or string from text file.
    '''
    with open(file_path, 'r') as cfile:
        content = cfile.readlines()
    cfile.close()
    content = [x.strip() for x in content]
    num_lines = len(content)
    return content, num_lines


def anno_parser(anno_path, num_pts):
    '''
    parse the annotation for 300W dataset, which has a fixed format for .pts file
    return:
    pts: 3 x num_pts (x, y, oculusion)
    '''
    data, num_lines = load_txt_file(anno_path)
    n_points = num_pts

    # read points coordinate
    pts = np.zeros((3, n_points), dtype='float32')
    line_offset = 0    # first point starts
    point_set = set()
    for point_index in range(n_points):
        try:
            pts_list = data[point_index + line_offset].split(',')       # x y format
            # if len(pts_list) > 2:    # handle edge case where additional whitespace exists after point coordinates
            #     pts_list = remove_item_from_list(pts_list, '')
            pts[0, point_index] = float(pts_list[0])
            pts[1, point_index] = float(pts_list[1])
            pts[2, point_index] = float(1) # oculusion flag, 0: oculuded, 1: visible. We use 1 for all points since no visibility is provided by 300-W
            point_set.add( point_index )
        except ValueError:
            print('error in loadin g points in %s' % anno_path)
    return pts, point_set


## change the vector to heatmap
def generate_label_map(pts, height, width, sigma, downsample, ctype):
    ## pts = 3 * N numpy array; points location is based on the image with size (height*downsample, width*downsample)
    #if isinstance(pts, numbers.Number):
    # this image does not provide the annotation, pts is a int number representing the number of points
    #return np.zeros((height,width,pts+1), dtype='float32'), np.ones((1,1,1+pts), dtype='float32')
    # nopoints == True means this image does not provide the annotation, pts is a int number representing the number of points

    assert isinstance(pts, np.ndarray) and len(pts.shape) == 2 and pts.shape[0] == 3, 'The shape of points : {}'.format(pts.shape)
    if isinstance(sigma, numbers.Number):
        sigma = np.zeros((pts.shape[1])) + sigma
    assert isinstance(sigma, np.ndarray) and len(sigma.shape) == 1 and sigma.shape[0] == pts.shape[1], 'The shape of sigma : {}'.format(sigma.shape)

    offset = downsample / 2.0 - 0.5
    num_points, threshold = pts.shape[1], 0.01

    visiable = pts[2, :].astype('bool')
    transformed_label = np.fromfunction( lambda y, x, pid : ((offset + x*downsample - pts[0,pid])**2 \
                                                        + (offset + y*downsample - pts[1,pid])**2) \
                                                          / -2.0 / sigma[pid] / sigma[pid],
                                                          (height, width, num_points), dtype=int)

    mask_heatmap      = np.ones((1, 1, num_points+1), dtype='float32')
    mask_heatmap[0, 0, :num_points] = visiable

    if ctype == 'laplacian':
        transformed_label = (1+transformed_label) * np.exp(transformed_label)
    elif ctype == 'gaussian':
        transformed_label = np.exp(transformed_label)
    else:
        raise TypeError('Does not know this type [{:}] for label generation'.format(ctype))
    transformed_label[ transformed_label < threshold ] = 0
    transformed_label[ transformed_label >         1 ] = 1
    transformed_label = transformed_label * mask_heatmap[:, :, :num_points]

    background_label  = 1 - np.amax(transformed_label, axis=2)
    background_label[ background_label < 0 ] = 0
    heatmap           = np.concatenate((transformed_label, np.expand_dims(background_label, axis=2)), axis=2).astype('float32')

    return heatmap*mask_heatmap, mask_heatmap

def apply_bound(points, width, height):
    new_points = points.copy()
    oks = np.vstack((points[0, :] >= 0,
                    points[1, :] >=0,
                    points[0, :] <= width,
                    points[1, :] <= height,
                    points[2, :].astype('bool')))
    oks = oks.transpose((1,0))
    new_points[2, :] = np.sum(oks, axis=1) == 5
    return new_points

def _get_paths_from_landmarks(path):
    """get landmark path list from landmark folder"""
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    landmarks = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_landmark_file(fname):
                landmark_path = os.path.join(dirpath, fname)
                landmarks.append(landmark_path)
    assert landmarks, '{:s} has no valid landmark file'.format(path)
    return landmarks


def _get_paths_from_images(path):
    """get image path list from image folder"""
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    num_frames = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        num_frame = len(sorted(fnames))
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
                num_frames.append(num_frame)
    assert images, '{:s} has no valid image file'.format(path)
    return images, num_frames


def _get_paths_from_landmark_lmdb(dataroot):
    """get landmark path list from lmdb meta info"""
    meta_info = pickle.load(open(os.path.join(dataroot, 'meta_info.pkl'), 'rb'))
    paths = meta_info['keys']
    sizes = meta_info['shape']
    if len(sizes) == 1:
        sizes = sizes * len(paths)
    return paths, sizes


def _get_paths_from_lmdb(dataroot):
    """get image path list from lmdb meta info"""
    meta_info = pickle.load(open(os.path.join(dataroot, 'meta_info.pkl'), 'rb'))
    paths = meta_info['keys']
    sizes = meta_info['resolution']
    if len(sizes) == 1:
        sizes = sizes * len(paths)
    return paths, sizes


def get_image_paths(data_type, dataroot):
    """get image path list
    support lmdb or image files"""
    paths, sizes = None, None
    if dataroot is not None:
        if data_type == 'lmdb':
            paths, sizes = _get_paths_from_lmdb(dataroot)
        elif data_type == 'img':
            paths, _ = _get_paths_from_images(dataroot)
            paths = sorted(paths)
        else:
            raise NotImplementedError('data_type [{:s}] is not recognized.'.format(data_type))
    return paths, sizes


def get_landmark_paths(data_type, dataroot):
    """get landmark path list
    support lmdb or landmark files"""
    paths, sizes = None, None
    if dataroot is not None:
        if data_type == 'lmdb':
            paths, sizes = _get_paths_from_landmark_lmdb(dataroot)
        elif data_type == 'landmark':
            paths = sorted(_get_paths_from_landmarks(dataroot))
        else:
            raise NotImplementedError('data_type [{:s}] is not recognized'.format(data_type))
    return paths, sizes


def glob_file_list(root):
    return sorted(glob.glob(os.path.join(root, '*')))


###################### read images ######################
def _read_img_lmdb(env, key, size):
    """read image from lmdb with key (w/ and w/o fixed size)
    size: (C, H, W) tuple"""
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    if buf is None:
        print(key)
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    C, H, W = size
    img = img_flat.reshape(H, W, C)
    return img


def _read_landmark_lmdb(env, key, size):
    """read landmark from lmdb with key (w/ and w/o fixed size)
    size: (3, 68) tuple"""
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    if buf is None:
        print(key)
    landmark_flat = np.frombuffer(buf, np.float32)
    row, col = size
    landmark = landmark_flat.reshape(row, col)
    return landmark


def read_img(env, path, size=None):
    """read image by cv2 or from lmdb
    return: Numpy float32, HWC, BGR, [0,1]"""
    if env is None:  # img
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    else:
        img = _read_img_lmdb(env, path, size)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def read_landmark(env, path, size=None):
    """read landmark by anno_parser or from lmdb
    return: Numpy float32, row col"""
    if env is None:
        landmark, _ = anno_parser(path, 68)
    else:
        landmark = _read_landmark_lmdb(env, path, size)
    return landmark


def read_img_seq(path):
    """Read a sequence of images from a given folder path
    Args:
        path (list/str): list of image paths/image folder path

    Returns:
        imgs (Tensor): size (T, C, H, W), RGB, [0, 1]
    """
    if type(path) is list:
        img_path_l = path
    else:
        img_path_l = sorted(glob.glob(os.path.join(path, '*')))
    img_l = [read_img(None, v) for v in img_path_l]
    # stack to Torch tensor
    imgs = np.stack(img_l, axis=0)
    imgs = imgs[:, :, :, [2, 1, 0]]
    imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 3, 1, 2)))).float()
    return imgs

def read_landmark_seq(path):
    """Read a sequence of landmarks from a given folder path
    Args:
        path (list/str): list of landmark paths/landmark folder path

    Returns:
        landmarks (Tensor): size (T, 68, 3)
    """
    if type(path) is list:
        landmark_path_l = path
    else:
        landmark_path_l = sorted(glob.glob(os.path.join(path, '*')))
    landmark_l = [read_landmark(None, v) for v in landmark_path_l]
    # stack to Torch tensor
    landmarks = np.stack(landmark_l, axis=0)
    landmarks = torch.from_numpy(np.ascontiguousarray(landmarks)).float()
    return landmarks

def index_generation(crt_i, max_n, N, padding='reflection'):
    """Generate an index list for reading N frames from a sequence of images
    Args:
        crt_i (int): current center index
        max_n (int): max number of the sequence of images (calculated from 1)
        N (int): reading N frames
        padding (str): padding mode, one of replicate | reflection | new_info | circle
            Example: crt_i = 0, N = 5
            replicate: [0, 0, 0, 1, 2]
            reflection: [2, 1, 0, 1, 2]
            new_info: [4, 3, 0, 1, 2]
            circle: [3, 4, 0, 1, 2]

    Returns:
        return_l (list [int]): a list of indexes
    """
    max_n = max_n - 1
    n_pad = N // 2
    return_l = []

    for i in range(crt_i - n_pad, crt_i + n_pad + 1):
        if i < 0:
            if padding == 'replicate':
                add_idx = 0
            elif padding == 'reflection':
                add_idx = -i
            elif padding == 'new_info':
                add_idx = (crt_i + n_pad) + (-i)
            elif padding == 'circle':
                add_idx = N + i
            else:
                raise ValueError('Wrong padding mode')
        elif i > max_n:
            if padding == 'replicate':
                add_idx = max_n
            elif padding == 'reflection':
                add_idx = max_n * 2 - i
            elif padding == 'new_info':
                add_idx = (crt_i - n_pad) - (i - max_n)
            elif padding == 'circle':
                add_idx = i - N
            else:
                raise ValueError('Wrong padding mode')
        else:
            add_idx = i
        return_l.append(add_idx)
    return return_l


####################
# image processing
# process on numpy image
####################

def augment_imgs_landmarks(img_size, img_list, pts, hflip=True, rot=True):
    """horizontal flip"""
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment_img(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def pts_hflip(pts):
        num_of_pt = pts.shape[1]
        if num_of_pt == 68:
            flip_index = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                          26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
                          27, 28, 29, 30,
                          35, 34, 33, 32, 31,
                          45, 44, 43, 42, 47, 46,
                          39, 38, 37, 36, 41, 40,
                          54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55,
                          64, 63, 62, 61, 60, 67, 66, 65]
        elif num_of_pt == 194:
            flip_index = [40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                          57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41,
                          70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 59, 58,
                          85, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71,
                          99,  98,  97,  96,  95,  94,  93,  92,  91,  90,  89, 88,  87,  86,
                          113, 112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100,
                          134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153,
                          114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133,
                          174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193,
                          154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173]
        elif num_of_pt == 19:
            flip_index = [5, 4, 3, 2, 1, 0,
                          11, 10, 9, 8, 7, 6,
                          14, 13, 12,
                          17, 16, 15, 18]
        temp = copy.deepcopy(pts)
        new_pts = pts.copy()
        for i in range(0, num_of_pt):
            new_pts[0, flip_index[i]] = img_size - temp[0, i]
            new_pts[1, flip_index[i]] = temp[1, i]
            new_pts[2, flip_index[i]] = temp[2, i]
        return new_pts

    def _augment_pts(pts):
        if hflip:
            pts = pts_hflip(pts)
        if vflip:
            pts = np.array(list(map(lambda a, b, c: [img_size - a, img_size - b, c], pts[0, :], pts[1, :], pts[2, :])))
            pts = np.transpose(pts, (1, 0))
        if rot90:
            pts = np.array(list(map(lambda a, b, c: [b, img_size - a, c], pts[0, :], pts[1, :], pts[2, :])))
            pts = np.transpose(pts, (1, 0))
        return pts

    return [_augment_img(img) for img in img_list], _augment_pts(pts)


def augment(img_list, hflip=True, rot=True):
    """horizontal flip OR rotate (0, 90, 180, 270 degrees)"""
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]

def augment_flow(img_list, flow_list, hflip=True, rot=True):
    """horizontal flip OR rotate (0, 90, 180, 270 degrees) with flows"""
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augment_flow(flow):
        if hflip:
            flow = flow[:, ::-1, :]
            flow[:, :, 0] *= -1
        if vflip:
            flow = flow[::-1, :, :]
            flow[:, :, 1] *= -1
        if rot90:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    rlt_img_list = [_augment(img) for img in img_list]
    rlt_flow_list = [_augment_flow(flow) for flow in flow_list]

    return rlt_img_list, rlt_flow_list


def channel_convert(in_c, tar_type, img_list):
    """conversion among BGR, gray and y"""
    if in_c == 3 and tar_type == 'gray':  # BGR to gray
        gray_list = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in img_list]
        return [np.expand_dims(img, axis=2) for img in gray_list]
    elif in_c == 3 and tar_type == 'y':  # BGR to y
        y_list = [bgr2ycbcr(img, only_y=True) for img in img_list]
        return [np.expand_dims(img, axis=2) for img in y_list]
    elif in_c == 1 and tar_type == 'RGB':  # gray/y to BGR
        return [cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) for img in img_list]
    else:
        return img_list


def rgb2ycbcr(img, only_y=True):
    """same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    """
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def bgr2ycbcr(img, only_y=True):
    """bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    """
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def ycbcr2rgb(img):
    """same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    """
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def modcrop(img_in, scale):
    """img_in: Numpy, HWC or HW"""
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img


####################
# Functions
####################


# matlab 'imresize' function, now only support 'bicubic'
def cubic(x):
    absx = torch.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * (
        (absx <= 1).type_as(absx)) + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * ((
            (absx > 1) * (absx <= 2)).type_as(absx))


def calculate_weights_indices(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    if (scale < 1) and (antialiasing):
        # Use a modified kernel to simultaneously interpolate and antialias- larger kernel width
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5+scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, P) + torch.linspace(0, P - 1, P).view(
        1, P).expand(out_length, P)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, P) - indices
    # apply cubic kernel
    if (scale < 1) and (antialiasing):
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, P)

    # If a column in weights is all zero, get rid of it. only consider the first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, P - 2)
        weights = weights.narrow(1, 1, P - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, P - 2)
        weights = weights.narrow(1, 0, P - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


def imresize(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: CHW RGB [0,1]
    # output: CHW RGB [0,1] w/o round

    in_C, in_H, in_W = img.size()
    _, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_C, in_H + sym_len_Hs + sym_len_He, in_W)
    img_aug.narrow(1, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:, :sym_len_Hs, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[:, -sym_len_He:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(in_C, out_H, in_W)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        out_1[0, i, :] = img_aug[0, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])
        out_1[1, i, :] = img_aug[1, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])
        out_1[2, i, :] = img_aug[2, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(in_C, out_H, in_W + sym_len_Ws + sym_len_We)
    out_1_aug.narrow(2, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :, :sym_len_Ws]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, :, -sym_len_We:]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(in_C, out_H, out_W)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        out_2[0, :, i] = out_1_aug[0, :, idx:idx + kernel_width].mv(weights_W[i])
        out_2[1, :, i] = out_1_aug[1, :, idx:idx + kernel_width].mv(weights_W[i])
        out_2[2, :, i] = out_1_aug[2, :, idx:idx + kernel_width].mv(weights_W[i])

    return out_2


def imresize_np(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: Numpy, HWC BGR [0,1]
    # output: HWC BGR [0,1] w/o round
    img = torch.from_numpy(img)

    in_H, in_W, in_C = img.size()
    _, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_H + sym_len_Hs + sym_len_He, in_W, in_C)
    img_aug.narrow(0, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:sym_len_Hs, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[-sym_len_He:, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(out_H, in_W, in_C)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        out_1[i, :, 0] = img_aug[idx:idx + kernel_width, :, 0].transpose(0, 1).mv(weights_H[i])
        out_1[i, :, 1] = img_aug[idx:idx + kernel_width, :, 1].transpose(0, 1).mv(weights_H[i])
        out_1[i, :, 2] = img_aug[idx:idx + kernel_width, :, 2].transpose(0, 1).mv(weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(out_H, in_W + sym_len_Ws + sym_len_We, in_C)
    out_1_aug.narrow(1, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :sym_len_Ws, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, -sym_len_We:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(out_H, out_W, in_C)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        out_2[:, i, 0] = out_1_aug[:, idx:idx + kernel_width, 0].mv(weights_W[i])
        out_2[:, i, 1] = out_1_aug[:, idx:idx + kernel_width, 1].mv(weights_W[i])
        out_2[:, i, 2] = out_1_aug[:, idx:idx + kernel_width, 2].mv(weights_W[i])

    return out_2.numpy()

def find_tensor_peak_batch(heatmap, downsample, threshold = 0.000001):
    radius = 4
    assert heatmap.dim() == 3, 'The dimension of the heatmap is wrong : {}'.format(heatmap.size())
    num_pts, H, W = heatmap.size(0), heatmap.size(1), heatmap.size(2)
    assert W > 1 and H > 1, 'To avoid the normalization function divide zero'
    # find the approximate location:
    score, index = torch.max(heatmap.view(num_pts, -1), 1)
    index_w = (index % W).float()
    index_h = (index / W).float()

    def normalize(x, L):
        return -1. + 2. * x.data / (L-1)
    boxes = [index_w - radius, index_h - radius, index_w + radius, index_h + radius]
    boxes[0] = normalize(boxes[0], W)
    boxes[1] = normalize(boxes[1], H)
    boxes[2] = normalize(boxes[2], W)
    boxes[3] = normalize(boxes[3], H)
    #affine_parameter = [(boxes[2]-boxes[0])/2, boxes[0]*0, (boxes[2]+boxes[0])/2,
    #                   boxes[0]*0, (boxes[3]-boxes[1])/2, (boxes[3]+boxes[1])/2]
    #theta = torch.stack(affine_parameter, 1).view(num_pts, 2, 3)

    affine_parameter = torch.zeros((num_pts, 2, 3))
    affine_parameter[:,0,0] = (boxes[2]-boxes[0])/2
    affine_parameter[:,0,2] = (boxes[2]+boxes[0])/2
    affine_parameter[:,1,1] = (boxes[3]-boxes[1])/2
    affine_parameter[:,1,2] = (boxes[3]+boxes[1])/2
    # extract the sub-region heatmap
    theta = affine_parameter.to(heatmap.device)
    grid_size = torch.Size([num_pts, 1, radius*2+1, radius*2+1])
    grid = F.affine_grid(theta, grid_size)
    sub_feature = F.grid_sample(heatmap.unsqueeze(1), grid).squeeze(1)
    sub_feature = F.threshold(sub_feature, threshold, np.finfo(float).eps)

    X = torch.arange(-radius, radius+1).to(heatmap).view(1, 1, radius*2+1)
    Y = torch.arange(-radius, radius+1).to(heatmap).view(1, radius*2+1, 1)

    sum_region = torch.sum(sub_feature.view(num_pts,-1),1)
    x = torch.sum((sub_feature*X).view(num_pts,-1),1) / sum_region + index_w
    y = torch.sum((sub_feature*Y).view(num_pts,-1),1) / sum_region + index_h

    x = x * downsample + downsample / 2.0 - 0.5
    y = y * downsample + downsample / 2.0 - 0.5
    return torch.stack([x, y],1), score

def evaluate_normalized_mean_error(predictions, groundtruth, facebb=None):
  ## compute total average normlized mean error
    # if extra_faces is not None: assert len(extra_faces) == len(predictions), 'The length of extra_faces is not right {} vs {}'.format( len(extra_faces), len(predictions) )
    # num_images = len(predictions)
    # for i in range(num_images):
        # c, g = predictions[i], groundtruth[i]
    # error_per_image = np.zeros((num_images,1))
    num_images = 1
    num_points = predictions.shape[1]
    error_per_image = np.zeros((1))

    for i in range(num_images):
        detected_points = predictions
        ground_truth_points = groundtruth
        if num_points == 68:
            interocular_distance = np.linalg.norm(ground_truth_points[:2, 36] - ground_truth_points[:2, 45])
            assert bool(ground_truth_points[2,36]) and bool(ground_truth_points[2,45])
        elif num_points == 51 or num_points == 49:
            interocular_distance = np.linalg.norm(ground_truth_points[:2, 19] - ground_truth_points[:2, 28])
            assert bool(ground_truth_points[2,19]) and bool(ground_truth_points[2,28])
        elif num_points == 19:
            W = facebb[2] - facebb[0]
            H = facebb[3] - facebb[1]
            interocular_distance = np.sqrt(W * H)# common.faceSZ_from_pts(groundtruth) #
        elif num_points == 194:
            interocular_distance = common.faceSZ_from_pts(groundtruth)
        else:
            raise Exception('----> Unknown number of points : {}'.format(num_points))
        dis_sum, pts_sum = 0, 0
        for j in range(num_points):
            if bool(ground_truth_points[2, j]):
                dis_sum = dis_sum + np.linalg.norm(detected_points[:2, j] - ground_truth_points[:2, j])
                pts_sum = pts_sum + 1

        error_per_image = dis_sum / (pts_sum*interocular_distance)

    # normalise_mean_error = error_per_image.mean()
    normalise_mean_error = error_per_image
    # calculate the auc for 0.07
    max_threshold = 0.07
    threshold = np.linspace(0, max_threshold, num=2000)
    accuracys = np.zeros(threshold.shape)
    for i in range(threshold.size):
        accuracys[i] = np.sum(error_per_image < threshold[i]) * 1.0 / error_per_image.size
    area_under_curve07 = auc(threshold, accuracys) / max_threshold
    # calculate the auc for 0.08
    max_threshold = 0.08
    threshold = np.linspace(0, max_threshold, num=2000)
    accuracys = np.zeros(threshold.shape)
    for i in range(threshold.size):
        accuracys[i] = np.sum(error_per_image < threshold[i]) * 1.0 / error_per_image.size
    area_under_curve08 = auc(threshold, accuracys) / max_threshold

    accuracy_under_007 = np.sum(error_per_image<0.07) * 100. / error_per_image.size
    accuracy_under_008 = np.sum(error_per_image<0.08) * 100. / error_per_image.size

    # print('Compute NME and AUC for {:} images with {:} points :: [(nms): mean={:.3f}, std={:.3f}], auc@0.07={:.3f}, auc@0.08-{:.3f}, acc@0.07={:.3f}, acc@0.08={:.3f}'.format(num_images, num_points, normalise_mean_error*100, error_per_image.std()*100, area_under_curve07*100, area_under_curve08*100, accuracy_under_007, accuracy_under_008))

    for_pck_curve = []
    for x in range(0, 3501, 1):
        error_bar = x * 0.0001
        accuracy = np.sum(error_per_image < error_bar) * 1.0 / error_per_image.size
        for_pck_curve.append((error_bar, accuracy))

    return normalise_mean_error, accuracy_under_008, for_pck_curve


def calc_nme(args, pts, batch_heatmaps, mask, hr_np, facebb, filename, sr):
    argmax = 4
    downsample = hr_np.shape[-1]/batch_heatmaps[0].size()[-1] #args.scale[0]
    batch_size = 1
    # The location of the current batch
    batch_locs, batch_scos = [], []
    for ibatch in range(batch_size):
        batch_location, batch_score = find_tensor_peak_batch(batch_heatmaps[-1][ibatch], downsample)
        batch_locs.append( batch_location )
        batch_scos.append( batch_score )
    batch_locs, batch_scos = torch.stack(batch_locs), torch.stack(batch_scos)

    # np_batch_locs: (1, 69, 2)
    np_batch_locs, np_batch_scos = batch_locs.detach().cpu().numpy(), batch_scos.detach().cpu().numpy()

    for i in range(len(np_batch_locs)):
        locations = np_batch_locs[ibatch,:-1,:]
        scores = np.expand_dims(np_batch_scos[ibatch,:-1], -1)

        prediction = np.concatenate((locations, scores), axis=1).transpose(1,0)
        groundtruth = pts[i].numpy()

        facebb = facebb[0].numpy()
        nme, accuracy_under_008, _ = evaluate_normalized_mean_error(prediction, groundtruth, facebb)
    return nme*100


if __name__ == '__main__':
    # test imresize function
    # read images
    img = cv2.imread('test.png')
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    # imresize
    scale = 1 / 4
    import time
    total_time = 0
    for i in range(10):
        start_time = time.time()
        rlt = imresize(img, scale, antialiasing=True)
        use_time = time.time() - start_time
        total_time += use_time
    print('average time: {}'.format(total_time / 10))

    import torchvision.utils
    torchvision.utils.save_image((rlt * 255).round() / 255, 'rlt.png', nrow=1, padding=0,
                                 normalize=False)
