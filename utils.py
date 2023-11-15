import tensorflow as tf
import cv2
import numpy as np
from tqdm import tqdm
import six
import pandas as pd
import json
import time


class FashionTagging():

  def __init__(self, model_path  = 'mask_rcnn_model'):
      print('Loading Model ...')
      load_start = time.time()
      model = tf.saved_model.load(model_path)
      load_end = time.time()
      print(f'Loading model sucessfully in {int(load_end - load_start)}s')

      self.model = model.signatures['serving_default']
      self.category_map, self.attribute_map, self.category_to_attribute_map, self.imposible_pair, \
      self.merging_map, self.type_of_cloth = self.load_map()

      # first run
      self.process_one_image(image = cv2.cvtColor(cv2.imread('assets/polo (shirt)_2.jpeg'), cv2.COLOR_BGR2RGB),
                             image_name = 'sample_image.jpg')


  def process_one_image(self, image, image_name, id = 1):
      img = tf.constant(image, dtype = tf.uint8)
      img = tf.expand_dims(img, axis = 0)

      prediction = self.model(input = img)
      prediction['source_id'] = np.array([id])

      image_scale = self.get_image_scale(image.shape[:2], 1024)
      prediction = self.postprocess(prediction, image_scale = image_scale)


      return {image_name: prediction}
  
  def postprocess(self, predictions, image_scale, score_threshold = 0.7):

    predictions = self.convert_to_numpy(predictions)
    predictions = self.process_predictions(predictions, image_scale)

    processed_predictions = {}
    for k, v in six.iteritems(predictions):
      if k not in processed_predictions:
        processed_predictions[k] = [v]
      else:
        processed_predictions[k].append(v)
    
    coco_result = self.convert_predictions_to_coco_annotations(processed_predictions,
                                                            output_image_size=1024,
                                                            encode_mask_fn=self.encode_mask,
                                                            score_threshold=score_threshold)
    

    

    final_result = self.process_output(coco_result, self.category_map, self.attribute_map, 
                                       self.category_to_attribute_map, self.imposible_pair)

    final_result = self.filter_output(final_result)
    final_result = self.merging_attribute(final_result, self.merging_map)
    final_result = self.merging_cloth_part(final_result)
    final_result = self.merging_category_attribute_convert_attribute_and_sub_cloth_part(final_result, self.type_of_cloth)
    
    
    return final_result

  def convert_to_numpy(self, predictions):
    new_predictions = {}
    for key, value in predictions.items():
        if key != 'source_id':
            new_predictions[key] = value.numpy()
        else:
            new_predictions[key] = value

    return new_predictions

  def process_predictions(self, predictions, image_scale):
    predictions['detection_boxes'] = (
        predictions['detection_boxes'].astype(np.float32))
    predictions['detection_boxes'] /= image_scale
    if 'detection_outer_boxes' in predictions:
        predictions['detection_outer_boxes'] = (
            predictions['detection_outer_boxes'].astype(np.float32))
        predictions['detection_outer_boxes'] /= image_scale
        
    return predictions

  def yxyx_to_xyxy(self, boxes):
    """Converts boxes from ymin, xmin, ymax, xmax to xmin, ymin, width, height.

    Args:
      boxes: a numpy array whose last dimension is 4 representing the coordinates
        of boxes in ymin, xmin, ymax, xmax order.

    Returns:
      boxes: a numpy array whose shape is the same as `boxes` in new format.

    Raises:
      ValueError: If the last dimension of boxes is not 4.
    """
    if boxes.shape[-1] != 4:
      raise ValueError(
          'boxes.shape[-1] is {:d}, but must be 4.'.format(boxes.shape[-1]))

    boxes_ymin = boxes[..., 0]
    boxes_xmin = boxes[..., 1]
    boxes_ymax = boxes[..., 2]
    boxes_xmax = boxes[..., 3]
    new_boxes = np.stack(
        [boxes_xmin, boxes_ymin, boxes_xmax, boxes_ymax], axis=-1)

    return new_boxes

  def get_new_image_size(self, image_size, output_size: int):
    image_height, image_width = image_size

    if image_width > image_height:
        scale = image_width / output_size
        new_width = output_size
        new_height = int(image_height / scale)
    else:
        scale = image_height / output_size
        new_height = output_size
        new_width = int(image_width / scale)

    return new_height, new_width

  def paste_instance_masks(self,
                          masks,
                          detected_boxes,
                          image_height,
                          image_width):
    """Paste instance masks to generate the image segmentation results.

    Args:
      masks: a numpy array of shape [N, mask_height, mask_width] representing the
        instance masks w.r.t. the `detected_boxes`.
      detected_boxes: a numpy array of shape [N, 4] representing the reference
        bounding boxes.
      image_height: an integer representing the height of the image.
      image_width: an integer representing the width of the image.

    Returns:
      segms: a numpy array of shape [N, image_height, image_width] representing
        the instance masks *pasted* on the image canvas.
    """

    def expand_boxes(boxes, scale):
      """Expands an array of boxes by a given scale."""
      # Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/boxes.py#L227  # pylint: disable=line-too-long
      # The `boxes` in the reference implementation is in [x1, y1, x2, y2] form,
      # whereas `boxes` here is in [x1, y1, w, h] form
      w_half = boxes[:, 2] * .5
      h_half = boxes[:, 3] * .5
      x_c = boxes[:, 0] + w_half
      y_c = boxes[:, 1] + h_half

      w_half *= scale
      h_half *= scale

      boxes_exp = np.zeros(boxes.shape)
      boxes_exp[:, 0] = x_c - w_half
      boxes_exp[:, 2] = x_c + w_half
      boxes_exp[:, 1] = y_c - h_half
      boxes_exp[:, 3] = y_c + h_half

      return boxes_exp

    # Reference: https://github.com/facebookresearch/Detectron/blob/master/detectron/core/test.py#L812  # pylint: disable=line-too-long
    # To work around an issue with cv2.resize (it seems to automatically pad
    # with repeated border values), we manually zero-pad the masks by 1 pixel
    # prior to resizing back to the original image resolution. This prevents
    # "top hat" artifacts. We therefore need to expand the reference boxes by an
    # appropriate factor.
    _, mask_height, mask_width = masks.shape
    scale = max((mask_width + 2.0) / mask_width,
                (mask_height + 2.0) / mask_height)

    ref_boxes = expand_boxes(detected_boxes, scale)
    ref_boxes = ref_boxes.astype(np.int32)
    padded_mask = np.zeros((mask_height + 2, mask_width + 2), dtype=np.float32)
    segms = []
    for mask_ind, mask in enumerate(masks):
      im_mask = np.zeros((image_height, image_width), dtype=np.uint8)
      # Process mask inside bounding boxes.
      padded_mask[1:-1, 1:-1] = mask[:, :]

      ref_box = ref_boxes[mask_ind, :]
      w = ref_box[2] - ref_box[0] + 1
      h = ref_box[3] - ref_box[1] + 1
      w = np.maximum(w, 1)
      h = np.maximum(h, 1)

      mask = cv2.resize(padded_mask, (w, h))
      mask = np.array(mask > 0.5, dtype=np.uint8)

      x_0 = min(max(ref_box[0], 0), image_width)
      x_1 = min(max(ref_box[2] + 1, 0), image_width)
      y_0 = min(max(ref_box[1], 0), image_height)
      y_1 = min(max(ref_box[3] + 1, 0), image_height)

      im_mask[y_0:y_1, x_0:x_1] = mask[
          (y_0 - ref_box[1]):(y_1 - ref_box[1]),
          (x_0 - ref_box[0]):(x_1 - ref_box[0])
      ]
      segms.append(im_mask)

    segms = np.array(segms)
    assert masks.shape[0] == segms.shape[0]
    return segms

  def encode_mask(self, mask: np.ndarray) -> str:
      pixels = mask.T.flatten()

      # We need to allow for cases where there is a '1' at either end of the sequence.
      # We do this by padding with a zero at each end when needed.
      use_padding = False
      if pixels[0] or pixels[-1]:
          use_padding = True
          pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
          pixel_padded[1:-1] = pixels
          pixels = pixel_padded

      rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
      if use_padding:
          rle = rle - 1

      rle[1::2] = rle[1::2] - rle[:-1:2]

      return ' '.join(str(x) for x in rle)

  def convert_predictions_to_coco_annotations(self, 
                                              predictions, 
                                              eval_image_sizes: dict = None, 
                                              output_image_size: int = None,
                                              encode_mask_fn=None, 
                                              score_threshold=0.05):
    """Converts a batch of predictions to annotations in COCO format.

    Args:
      predictions: a dictionary of lists of numpy arrays including the following
        fields. K below denotes the maximum number of instances per image.
        Required fields:
          - source_id: a list of numpy arrays of int or string of shape
              [batch_size].
          - num_detections: a list of numpy arrays of int of shape [batch_size].
          - detection_boxes: a list of numpy arrays of float of shape
              [batch_size, K, 4], where coordinates are in the original image
              space (not the scaled image space).
          - detection_classes: a list of numpy arrays of int of shape
              [batch_size, K].
          - detection_scores: a list of numpy arrays of float of shape
              [batch_size, K].
        Optional fields:
          - detection_masks: a list of numpy arrays of float of shape
              [batch_size, K, mask_height, mask_width].

    Returns:
      coco_predictions: prediction in COCO annotation format.
    """
    coco_predictions = []
    num_batches = len(predictions['source_id'])
    use_outer_box = 'detection_outer_boxes' in predictions

    if encode_mask_fn is None:
      raise Exception

    for i in tqdm(range(num_batches), total=num_batches):
      predictions['detection_boxes'][i] = self.yxyx_to_xyxy(
          predictions['detection_boxes'][i])

      if use_outer_box:
        predictions['detection_outer_boxes'][i] = self.yxyx_to_xyxy(
            predictions['detection_outer_boxes'][i])
        mask_boxes = predictions['detection_outer_boxes']
      else:
        mask_boxes = predictions['detection_boxes']

      batch_size = predictions['source_id'][i].shape[0]
      for j in range(batch_size):
        image_id = predictions['source_id'][i][j]
        orig_image_size = predictions['image_info'][i][j, 0]

        if eval_image_sizes:
          eval_image_size = eval_image_sizes[image_id] if eval_image_sizes else orig_image_size
        elif output_image_size:
          eval_image_size = self.get_new_image_size(orig_image_size, output_image_size)
        else:
          eval_image_size = orig_image_size

        eval_scale = orig_image_size[0] / eval_image_size[0]

        bbox_indices = np.argwhere(predictions['detection_scores'][i][j] >= score_threshold).flatten()

        if 'detection_masks' in predictions:
          predicted_masks = predictions['detection_masks'][i][j, bbox_indices]
          image_masks = self.paste_instance_masks(
              predicted_masks,
              mask_boxes[i][j, bbox_indices].astype(np.float32)/ eval_scale,
              int(eval_image_size[0]),
              int(eval_image_size[1]))
          binary_masks = (image_masks > 0.0).astype(np.uint8)
          encoded_masks = [encode_mask_fn(binary_mask) for binary_mask in list(binary_masks)]

          mask_masks = (predicted_masks > 0.5).astype(np.float32)
          mask_areas = mask_masks.sum(axis=-1).sum(axis=-1)
          mask_area_fractions = (mask_areas / np.prod(predicted_masks.shape[1:])).tolist()
          mask_mean_scores = ((predicted_masks * mask_masks).sum(axis=-1).sum(axis=-1) / mask_areas).tolist()

        for m, k in enumerate(bbox_indices):
          ann = {
            'image_id': int(image_id),
            'category_id': int(predictions['detection_classes'][i][j, k]),
            # 'bbox': (predictions['detection_boxes'][i][j, k].astype(np.float32) / eval_scale).tolist(),
            'bbox': (predictions['detection_boxes'][i][j, k].astype(int)).tolist(),
            'score': float(predictions['detection_scores'][i][j, k]),
          }

          if 'detection_masks' in predictions:
            ann['segmentation'] = encoded_masks[m]
            ann['mask_mean_score'] = mask_mean_scores[m]
            ann['mask_area_fraction'] = mask_area_fractions[m]

          if 'detection_attributes' in predictions:
            ann['attribute_probabilities'] = predictions['detection_attributes'][i][j, k].tolist()

          coco_predictions.append(ann)

    for i, ann in enumerate(coco_predictions):
      ann['id'] = i + 1

    return coco_predictions

  def load_map(self):
      category_map = pd.read_csv('assets/category_map.csv', index_col = 0)
      attribute_map = pd.read_csv('assets/attribute_map.csv', index_col= 0)

      support_att = np.array(json.load(open('assets/support_attributes.json'))["support_att"])

      map_new_id_att = {key: value for value, key in enumerate(list(attribute_map.index))}
      category_to_attribute_content = json.load(open('assets/category-attributes-2.json', 'r'))
      category_to_attribute_map = {}

      for ix in range(46):
          value = np.zeros(294)
          if str(ix) in category_to_attribute_content.keys():
              for ele in category_to_attribute_content[str(ix)]:
                  value[map_new_id_att[ele]] = 1.
          category_to_attribute_map[ix + 1] = value  * support_att

      # print(category_to_attribute_map)

      
      
      
      imposible_pair = json.load(open('assets/imposible_pair.json'))['imposible_pair']

      merging_map = json.load(open('assets/merging_map.json', 'r'))
      type_of_cloth = json.load(open('assets/type_of_cloth.json', 'r'))["type"]
      
      return category_map, attribute_map, category_to_attribute_map, imposible_pair, merging_map, type_of_cloth

  def process_output(self, predictions, category_map, attribute_map, category_to_attribute, imposible_pair):
    

    for i in range(len(predictions)):
        predictions[i]['category_info'] = category_map[category_map['id'] == predictions[i]['category_id']].iloc[0].to_dict()

        # rule checking
        attributes_ele = np.array(predictions[i]['attribute_probabilities']) * category_to_attribute[predictions[i]['category_id']]
        
        attribute_map['gt'] = attributes_ele
        predictions[i]['attribute_info'] = attribute_map[attribute_map['gt'] > attribute_map['threshold']].T.to_dict()

        ls_att = sorted([int(x) for x in predictions[i]['attribute_info'].keys()])
        att_error = []
        for j in range(len(ls_att)):
           for k in range(j + 1, len(ls_att)):
              if [ls_att[j], ls_att[k]] in imposible_pair:
                    if attribute_map['gt'][ls_att[j]] < attribute_map['gt'][ls_att[k]]:
                        att_error.append(ls_att[j])
                    if attribute_map['gt'][ls_att[j]] > attribute_map['gt'][ls_att[k]]:
                        att_error.append(ls_att[k])

        att_error = list(set(att_error))
        
        predictions[i]['attribute_error'] = {}

        for ele in att_error:
          predictions[i]['attribute_error'][ele] = predictions[i]['attribute_info'].pop(ele)
           
                
    return predictions

  def filter_output(self, predictions):
    new_predictions = []
       
    for ele in predictions:
      new_predictions.append({'image_id': ele['image_id'],
                              'category_id': ele['category_id'],
                              'bbox': ele['bbox'],
                              'score': ele['score'],
                              'id': ele['id'],
                              'category_info': ele['category_info'],
                              'attribute_info': ele['attribute_info'],
                              })
    
    return new_predictions
  
  def get_image_scale(self, original_size, output_image_size = 1024):
    
      x_ratio = output_image_size / original_size[1]
      y_ratio = output_image_size / original_size[0]

      return np.array([[[y_ratio, x_ratio, y_ratio, x_ratio]]])
  
  def merging_attribute(self, prediction, merging_map):
    new_prediction = []
    for prediction_element in prediction:
        new_prediction_element = {}
        new_prediction_element['bbox'] = prediction_element['bbox']
        new_prediction_element['category_info'] = prediction_element['category_info']
        new_prediction_element['attribute_info'] = []
        for _, attribute_content in prediction_element['attribute_info'].items():
          new_element_of_attribute_info = {}
          new_element_of_attribute_info["supercategory"] = attribute_content["supercategory"]
          # new_element_of_attribute_info["id"] = attribute_index
          if attribute_content['name'] in merging_map.keys():
              new_element_of_attribute_info['name'] = merging_map[attribute_content['name']]
          else:
              new_element_of_attribute_info['name'] = attribute_content['name']
          if new_element_of_attribute_info not in new_prediction_element['attribute_info']:
            new_prediction_element['attribute_info'].append(new_element_of_attribute_info)
        
        new_prediction.append(new_prediction_element)
    return new_prediction

  def bb_intersection_over_boxB(self, boxA, boxB):
	
      xA = max(boxA[0], boxB[0])
      yA = max(boxA[1], boxB[1])
      xB = min(boxA[2], boxB[2])
      yB = min(boxA[3], boxB[3])
      
      interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
      
      boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
      
      ioB = interArea / boxBArea
      
      return ioB

  def merging_cloth_part(self, prediction):
      main_cloth_part =  []
      sub_cloth_part = []
      other_cloth_part = []

      for ele in prediction:
          if ele['category_info']['id'] <= 12:
            main_cloth_part.append(ele)
          elif ele['category_info']['id'] <= 26:
            other_cloth_part.append(ele)
          elif ele['category_info']['id'] <= 35:
            sub_cloth_part.append(ele)
      
      sub_used_list = []
      for i in range(len(main_cloth_part)):
        main_cloth_part[i]['sub_cloth_part'] = []
        boxA = main_cloth_part[i]['bbox']
        for j in range(len(sub_cloth_part)):
            if j not in sub_used_list:
              boxB = sub_cloth_part[j]['bbox']
              if self.bb_intersection_over_boxB(boxA, boxB) > 0.9:
                main_cloth_part[i]['sub_cloth_part'].append(sub_cloth_part[j])
                sub_used_list.append(j)
      
      for i in range(len(main_cloth_part)):
          if len(main_cloth_part[i]['sub_cloth_part']) == 0:
            continue
          else:
            exist_sub_cloth_part = []
            exist_id_sub_cloth_part = []

            for ele in main_cloth_part[i]['sub_cloth_part']:
              if ele['category_info']['id'] not in exist_id_sub_cloth_part:
                exist_id_sub_cloth_part.append(ele['category_info']['id'])
                exist_sub_cloth_part.append(ele)
              else:
                index_duplicate = exist_id_sub_cloth_part.index(ele['category_info']['id'])
                if len(exist_sub_cloth_part[index_duplicate]['attribute_info']) < len(ele['attribute_info']) :
                  exist_id_sub_cloth_part.pop(index_duplicate)
                  exist_sub_cloth_part.pop(index_duplicate)
                  exist_id_sub_cloth_part.append(ele['category_info']['id'])
                  exist_sub_cloth_part.append(ele)

            main_cloth_part[i]['sub_cloth_part'] = exist_sub_cloth_part

      output = main_cloth_part + other_cloth_part
      return output

  def convert_attribute(self, attributes):
    result = {}
    for attribute in attributes:
      if attribute["supercategory"] not in result.keys():
          result[attribute["supercategory"]] = []
      result[attribute["supercategory"]].append(attribute["name"])

    return result
  def convert_sub_cloth_part(self, sub_cloth_part):
    result = {}
    for ele in sub_cloth_part:
      if ele["category_info"]["name"] not in result.keys():
        result[ele["category_info"]["name"]] = []
      for sub_ele in ele["attribute_info"]:
         result[ele["category_info"]["name"]].append(sub_ele["name"])
      

    return result

  def merging_category_attribute_convert_attribute_and_sub_cloth_part(self, prediction, type_of_cloth):
      new_prediction = []
      for ele in prediction:
        new_ele = {"bbox": ele["bbox"]}
        
      
        if ele["category_info"]["id"] == 1:
          new_ele["name"] = "shirt"
          new_ele["supercategory"] = "upperbody"
          new_ele["attribute_info"] = self.convert_attribute(ele["attribute_info"])
        elif ele["category_info"]["id"] <= 13:
          other_attribute = []
          have_new_name = False
          for att in ele["attribute_info"]:
            if att["name"] in type_of_cloth:
                have_new_name = True
                new_ele["name"] = att["name"]
                new_ele["supercategory"] = ele["category_info"]["supercategory"]
            else:
                other_attribute.append(att)

          if not have_new_name:
            new_ele["name"] = ele["category_info"]["name"]
            new_ele["supercategory"] = ele["category_info"]["supercategory"]
          
          new_ele["attribute_info"] = self.convert_attribute(other_attribute)
        else:
          new_ele["name"] = ele["category_info"]["name"]
          new_ele["supercategory"] = ele["category_info"]["supercategory"]
          new_ele["attribute_info"] = self.convert_attribute(ele["attribute_info"])
        
        if "sub_cloth_part" in ele.keys():
          #  new_ele["sub_cloth_part"] = ele["sub_cloth_part"]
          new_ele["attribute_info"] = new_ele["attribute_info"] | self.convert_sub_cloth_part(ele["sub_cloth_part"])


        new_prediction.append(new_ele)  
            
     
      return new_prediction
  
  

class ErrorCode(Exception):
    def __init__(self, message, code=400):
        self.message = message
        self.code = code
    
  