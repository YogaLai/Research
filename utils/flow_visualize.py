# import cvbase
import matplotlib.pyplot as plt
import numpy as np
from math import ceil
import cv2
from skimage.io import imsave


def image_flow_viisualize(flow, img1, img2):
    flow = flow[0] * 20.0
    flow = flow.cpu().data.numpy()
    # scale the flow back to the input size
    divisor=64
    H = img1.shape[1]
    W = img1.shape[2] 
    H_ = int(ceil(H/divisor) * divisor)
    W_ = int(ceil(W/divisor) * divisor)
    flow = np.swapaxes(np.swapaxes(flow, 0, 1), 1, 2) # 
    u_ = cv2.resize(flow[:,:,0],(W,H))
    v_ = cv2.resize(flow[:,:,1],(W,H))
    u_ *= W/ float(W_)
    v_ *= H/ float(H_)
    flow = np.dstack((u_,v_))

    flow = cvbase.read_flow(flow)
    rgb_flow = cvbase.flow2rgb(flow)

    img1 = np.swapaxes(np.swapaxes(img1, 0, 1), 1, 2) # 
    img2 = np.swapaxes(np.swapaxes(img2, 0, 1), 1, 2) # 


    plt.subplot(1,3,1)
    plt.title('image1')
    plt.imshow(img1)

    plt.subplot(1,3,2)
    plt.title('image2')
    plt.imshow(img2)

    plt.subplot(1,3,3)
    plt.title('flow')
    plt.imshow(rgb_flow)

    plt.savefig('testflow.png')

def make_color_wheel(bins=None):
    """Build a color wheel

    Args:
        bins(list or tuple, optional): specify number of bins for each color
            range, corresponding to six ranges: red -> yellow, yellow -> green,
            green -> cyan, cyan -> blue, blue -> magenta, magenta -> red.
            [15, 6, 4, 11, 13, 6] is used for default (see Middlebury).
    
    Returns:
        ndarray: color wheel of shape (total_bins, 3)
    """
    if bins is None:
        bins = [15, 6, 4, 11, 13, 6]
    assert len(bins) == 6

    RY, YG, GC, CB, BM, MR = tuple(bins)

    ry = [1, np.arange(RY) / RY, 0]
    yg = [1 - np.arange(YG) / YG, 1, 0]
    gc = [0, 1, np.arange(GC) / GC]
    cb = [0, 1 - np.arange(CB) / CB, 1]
    bm = [np.arange(BM) / BM, 0, 1]
    mr = [1, 0, 1 - np.arange(MR) / MR]

    num_bins = RY + YG + GC + CB + BM + MR

    color_wheel = np.zeros((3, num_bins), dtype=np.float32)

    col = 0
    for i, color in enumerate([ry, yg, gc, cb, bm, mr]):
        for j in range(3):
            color_wheel[j, col:col + bins[i]] = color[j]
        col += bins[i]

    return color_wheel.T

def flow2rgb(flow, color_wheel=None, unknown_thr=1e6):
    """Convert flow map to RGB image

    Args:
        flow(ndarray): optical flow
        color_wheel(ndarray or None): color wheel used to map flow field to RGB
            colorspace. Default color wheel will be used if not specified
        unknown_thr(str): values above this threshold will be marked as unknown
            and thus ignored
    
    Returns:
        ndarray: an RGB image that can be visualized
    """
    assert flow.ndim == 3 and flow.shape[-1] == 2
    if color_wheel is None:
        color_wheel = make_color_wheel()
    assert color_wheel.ndim == 2 and color_wheel.shape[1] == 3
    num_bins = color_wheel.shape[0]

    dx = flow[:, :, 0].copy()
    dy = flow[:, :, 1].copy()

    ignore_inds = (np.isnan(dx) | np.isnan(dy) | (np.abs(dx) > unknown_thr) |
                   (np.abs(dy) > unknown_thr))
    dx[ignore_inds] = 0
    dy[ignore_inds] = 0

    rad = np.sqrt(dx**2 + dy**2)
    if np.any(rad > np.finfo(float).eps):
        max_rad = np.max(rad)
        dx /= max_rad
        dy /= max_rad

    [h, w] = dx.shape

    rad = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(-dy, -dx) / np.pi

    bin_real = (angle + 1) / 2 * (num_bins - 1)
    bin_left = np.floor(bin_real).astype(int)
    bin_right = (bin_left + 1) % num_bins
    w = (bin_real - bin_left.astype(np.float32))[..., None]
    flow_img = (
        1 - w) * color_wheel[bin_left, :] + w * color_wheel[bin_right, :]
    small_ind = rad <= 1
    flow_img[small_ind] = 1 - rad[small_ind, None] * (1 - flow_img[small_ind])
    flow_img[np.logical_not(small_ind)] *= 0.75

    flow_img[ignore_inds, :] = 0

    return flow_img

def show_flow(flow_path):
    flow = cvbase.optflow.io.read_flow(flow_path)
    flow_img = flow2rgb(flow)
    cvbase.visualize.show_img(cvbase.image.rgb2bgr(flow_img), 'flow', 0)


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img

def save_flow(save_path, flow, normalize=True, info=None, flow_mag_max=None):
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    flow_magnitude, flow_angle = cv2.cartToPolar(flow[..., 0].astype(np.float32), flow[..., 1].astype(np.float32))

    # A couple times, we've gotten NaNs out of the above...
    nans = np.isnan(flow_magnitude)
    if np.any(nans):
        nans = np.where(nans)
        flow_magnitude[nans] = 0.

    # Normalize
    hsv[..., 0] = flow_angle * 180 / np.pi / 2
    if normalize is True:
        if flow_mag_max is None:
            hsv[..., 1] = cv2.normalize(flow_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        else:
            hsv[..., 1] = flow_magnitude * 255 / flow_mag_max
    else:
        hsv[..., 1] = flow_magnitude
    hsv[..., 2] = 255
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # show
    # plt.imsave(save_path, img)
    imsave(save_path, img)

if __name__ == '__main__':
    flow = 'D:/yoga\BSP_lab/temporal_stereo/flow_images/warp_out.flo'
    show_flow(flow)
