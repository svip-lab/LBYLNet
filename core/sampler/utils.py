import cv2
import numpy as np
import random
import math
import pdb
import re


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def normalize_(image, mean, std):
    image -= mean
    image /= std

def lighting_(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3, ))
    image += np.dot(eigvec, eigval * alpha)

def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2

def saturation_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs[:, :, None])

def brightness_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha

def contrast_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs_mean)

def color_jittering_(data_rng, image):
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)

    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.4)

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = center

    height, width = heatmap.shape[0:2]
    
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

def gaussian_radius(det_size, min_overlap):
    height, width = det_size

    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 - sq1) / (2 * a1)

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 - sq2) / (2 * a2)

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / (2 * a3)
    return min(r1, r2, r3)

def _get_border(border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

def valid_affine(bbox, input_size):
    x1, y1 = bbox[0], bbox[1]
    x2, y2 = bbox[2], bbox[3]
    h,  w = input_size
    # at least one point is in the image
    def _is_in(x, y):
        return (x> 0 and x < w-1) and (y > 0 and y < h-1)
    cnd1 = _is_in(x1, y1)
    cnd2 = _is_in(x1, y2)
    cnd3 = _is_in(x2, y1)
    cnd4 = _is_in(x2, y2)
    return  cnd1 or cnd2 or cnd3 or cnd4


def random_crop(image, detections, random_scales, view_size, border=64):
    view_height, view_width   = view_size
    image_height, image_width = image.shape[0:2]

    scale  = np.random.choice(random_scales)
    height = int(view_height * scale)
    width  = int(view_width  * scale)

    cropped_image = np.zeros((height, width, 3), dtype=image.dtype)

    w_border = _get_border(border, image_width)
    h_border = _get_border(border, image_height)

    ctx = np.random.randint(low=w_border, high=image_width - w_border)
    cty = np.random.randint(low=h_border, high=image_height - h_border)

    x0, x1 = max(ctx - width // 2, 0),  min(ctx + width // 2, image_width)
    y0, y1 = max(cty - height // 2, 0), min(cty + height // 2, image_height)

    left_w, right_w = ctx - x0, x1 - ctx
    top_h, bottom_h = cty - y0, y1 - cty

    # crop image
    cropped_ctx, cropped_cty = width // 2, height // 2
    x_slice = slice(cropped_ctx - left_w, cropped_ctx + right_w)
    y_slice = slice(cropped_cty - top_h, cropped_cty + bottom_h)
    cropped_image[y_slice, x_slice, :] = image[y0:y1, x0:x1, :]

    # crop detections
    cropped_detections = detections.copy()
    cropped_detections[:, 0:4:2] -= x0
    cropped_detections[:, 1:4:2] -= y0
    cropped_detections[:, 0:4:2] += cropped_ctx - left_w
    cropped_detections[:, 1:4:2] += cropped_cty - top_h

    return cropped_image, cropped_detections

def random_crop_(image, bbox, random_scales, view_size, border=64):
    view_height, view_width   = view_size
    image_height, image_width = image.shape[0:2]

    scale  = np.random.choice(random_scales)
    height = int(view_height * scale)
    width  = int(view_width  * scale)

    cropped_image = np.zeros((height, width, 3), dtype=image.dtype)

    w_border = _get_border(border, image_width)
    h_border = _get_border(border, image_height)

    ctx = np.random.randint(low=w_border, high=image_width - w_border)
    cty = np.random.randint(low=h_border, high=image_height - h_border)

    x0, x1 = max(ctx - width // 2, 0),  min(ctx + width // 2, image_width)
    y0, y1 = max(cty - height // 2, 0), min(cty + height // 2, image_height)

    left_w, right_w = ctx - x0, x1 - ctx
    top_h, bottom_h = cty - y0, y1 - cty

    # crop image
    cropped_ctx, cropped_cty = width // 2, height // 2
    x_slice = slice(cropped_ctx - left_w, cropped_ctx + right_w)
    y_slice = slice(cropped_cty - top_h, cropped_cty + bottom_h)
    cropped_image[y_slice, x_slice, :] = image[y0:y1, x0:x1, :]

    # crop detections
    cropped_bbox = bbox.copy()
    cropped_bbox[0:4:2] -= x0
    cropped_bbox[1:4:2] -= y0
    cropped_bbox[0:4:2] += cropped_ctx - left_w
    cropped_bbox[1:4:2] += cropped_cty - top_h

    return cropped_image, cropped_bbox

def crop_image(image, center, size, output_size=None):
    if output_size == None:
        output_size = size

    cty, ctx            = center
    height, width       = size
    o_height, o_width   = output_size
    im_height, im_width = image.shape[0:2]
    cropped_image       = np.zeros((o_height, o_width, 3), dtype=image.dtype)

    x0, x1 = max(0, ctx - width // 2), min(ctx + width // 2, im_width)
    y0, y1 = max(0, cty - height // 2), min(cty + height // 2, im_height)

    left, right = ctx - x0, x1 - ctx
    top, bottom = cty - y0, y1 - cty

    cropped_cty, cropped_ctx = o_height // 2, o_width // 2
    y_slice = slice(cropped_cty - top, cropped_cty + bottom)
    x_slice = slice(cropped_ctx - left, cropped_ctx + right)
    cropped_image[y_slice, x_slice, :] = image[y0:y1, x0:x1, :]

    border = np.array([
       cropped_cty - top,
       cropped_cty + bottom,
       cropped_ctx - left,
       cropped_ctx + right
    ], dtype=np.float32)

    offset = np.array([
        cty - o_height // 2,
        ctx - o_width  // 2
    ])

    return cropped_image, border, offset

def letterbox(img, mask, height, color=None):  # resize a rectangular image to a padded square
    if color is None:
        color = (123.7, 116.3, 103.5)
    shape = img.shape[:2]  # shape = [height, width]
    ratio = float(height) / max(shape)  # ratio  = old / new
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))
    dw = (height - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded square
    if mask is not None:
        mask = cv2.resize(mask, new_shape, interpolation=cv2.INTER_NEAREST)  # resized, no border
        mask = cv2.copyMakeBorder(mask, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)  # padded square
    return img, mask, ratio, dw, dh


def random_affine(img, mask, targets, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
                  borderValue=(123.7, 116.3, 103.5), all_bbox=None):
    

    border = 0  # width of added border (optional)
    height = max(img.shape[0], img.shape[1]) + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * img.shape[1] + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpPerspective(img, M, dsize=(height, height), flags=cv2.INTER_LINEAR,
                              borderValue=borderValue)  # BGR order borderValue
    if mask is not None:
        maskw = cv2.warpPerspective(mask, M, dsize=(height, height), flags=cv2.INTER_NEAREST,
                                  borderValue=255)  # BGR order borderValue
    else:
        maskw = None

    # Return warped points also
    if type(targets)==type([1]):
        targetlist=[]
        for bbox in targets:
            targetlist.append(wrap_points(bbox, M, height, a))
        return imw, maskw, targetlist, M
    elif all_bbox is not None:
        targets = wrap_points(targets, M, height, a)
        for ii in range(all_bbox.shape[0]):
            all_bbox[ii,:] = wrap_points(all_bbox[ii,:], M, height, a)
        return imw, maskw, targets, all_bbox, M
    elif targets is not None:   ## previous main
        targets = wrap_points(targets, M, height, a)
        return imw, maskw, targets, M
    else:
        return imw




def wrap_points(targets, M, height, a):
    # n = targets.shape[0]
    # points = targets[:, 1:5].copy()
    points = targets.copy()
    # area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])
    area0 = (points[2] - points[0]) * (points[3] - points[1])

    # warp points
    xy = np.ones((4, 3))
    xy[:, :2] = points[[0, 1, 2, 3, 0, 3, 2, 1]].reshape(4, 2)  # x1y1, x2y2, x1y2, x2y1
    xy = (xy @ M.T)[:, :2].reshape(1, 8)

    # create new boxes
    x = xy[:, [0, 2, 4, 6]]
    y = xy[:, [1, 3, 5, 7]]
    xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, 1).T

    # apply angle-based reduction
    radians = a * math.pi / 180
    reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
    x = (xy[:, 2] + xy[:, 0]) / 2
    y = (xy[:, 3] + xy[:, 1]) / 2
    w = (xy[:, 2] - xy[:, 0]) * reduction
    h = (xy[:, 3] - xy[:, 1]) * reduction
    xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, 1).T

    # reject warped points outside of image
    np.clip(xy, 0, height, out=xy)
    w = xy[:, 2] - xy[:, 0]
    h = xy[:, 3] - xy[:, 1]
    area = w * h
    ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
    i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

    ## print(targets, xy)
    ## [ 56  36 108 210] [[ 47.80464857  15.6096533  106.30993434 196.71267693]]
    # targets = targets[i]
    # targets[:, 1:5] = xy[i]
    targets = xy[0]
    return targets   


def clip_bbox_(bbox, input_size):
    height, width = input_size
    bbox[0:4:2] = np.clip(bbox[0:4:2], 0, width - 1)
    bbox[1:4:2] = np.clip(bbox[1:4:2], 0, height - 1)
    return bbox

def random_affine_(image, bbox):
    affine_image, _, affine_bbox, M = random_affine(image, None, bbox, \
                    degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.90, 1.10))
    return affine_image, affine_bbox.astype(int)

def random_flip_(image, phrase, bbox):
    _, w = image.shape[0], image.shape[1]
    image = cv2.flip(image, 1)
    bbox[0], bbox[2] = w-bbox[2]-1, w-bbox[0]-1
    phrase = phrase.replace('right','*&^special^&*').replace('left','right').replace('*&^special^&*','left')
    return image, phrase, bbox

def resize_image_(image, bbox, input_size, padding_color=None):
    image, _, ratio, dw, dh = letterbox(image, None, input_size[0], color=padding_color)
    bbox[0], bbox[2] = bbox[0]*ratio+dw, bbox[2]*ratio+dw
    bbox[1], bbox[3] = bbox[1]*ratio+dh, bbox[3]*ratio+dh
    return image, bbox


def show_example(image, bbox, phrase, font_size=0.5, color=None, name=None):
    image = image.copy()
    # pdb.set_trace()
    if image.dtype == np.float32:
        image = (image * 255).astype(np.uint8)
    bbox = bbox.copy().astype(int)

    tl = bbox[0], bbox[1]
    br = bbox[2], bbox[3]
    if color is None: color = (0.0, 255.0, 0.0)
    phrase_size  = cv2.getTextSize(phrase, cv2.FONT_HERSHEY_SIMPLEX, font_size, 2)[0]
    #bbox = bbox[0:4].astype(np.int32)
    if bbox[1] - phrase_size[1] - 2 < 0:
        cv2.rectangle(image,
            (bbox[0], bbox[1] + 2),
            (bbox[0] + phrase_size[0], bbox[1] + phrase_size[1] + 2),
            color, -1
        )
        cv2.putText(image, phrase,
            (bbox[0], bbox[1] + phrase_size[1] + 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), thickness=1
        )
    else:
        cv2.rectangle(image,
            (bbox[0], bbox[1] - phrase_size[1] - 2),
            (bbox[0] + phrase_size[0], bbox[1] - 2),
            color, -1
        )
        cv2.putText(image, phrase,
            (bbox[0], bbox[1] - 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), thickness=1
        )
    cv2.rectangle(image,
        (bbox[0], bbox[1]),
        (bbox[2], bbox[3]),
        color, 2
    )
    cv2.imwrite("trash/{}.jpg".format(name), image)


# lanuage utils
# from pytorch_pretrained_bert.tokenization import BertTokenizer
# from pytorch_pretrained_bert.modeling import BertModel
def read_examples(input_line, unique_id):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    # unique_id = 0
    line = input_line #reader.readline()
    # if not line:
    #     break
    line = line.strip()
    text_a = None
    text_b = None
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        text_a = line
    else:
        text_a = m.group(1)
        text_b = m.group(2)
    examples.append(
        InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    # unique_id += 1
    return examples

## Bert text encoding
class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids

def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features