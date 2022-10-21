import numpy as np

def calc_iou(gt, anchor_boxes):        
        """
        Calculate IOU of ground truth and anchor boxes
        
        Input:
            gt: ground truth image, shape: (#object per image, 4)
            anchor_boxes: anchor boxes, shape: (sum of grid size of all classifier * num_boxes, 4)
        Output:
            Matrix of iou. Row indicates each ground truth box and column indicates each anchor box.
            shape: (#object per image, sum of grid size of all classifier)        
        """      
        
        m = gt.shape[0] # Object per image
        n = anchor_boxes.shape[0] # Number of all boxes
        
        #Calculate min_xy
        min_xy = np.maximum(np.tile(np.expand_dims(gt[:,0:2], axis = 1), reps = (1,n,1)),
                            np.tile(np.expand_dims(anchor_boxes[:, 0:2], axis = 0), reps = (m,1,1)))
        
        #Calculate max_xy
        max_xy = np.minimum(np.tile(np.expand_dims(gt[:,2:4], axis = 1), reps = (1,n,1)),
                            np.tile(np.expand_dims(anchor_boxes[:, 2:4], axis = 0), reps = (m,1,1)))
        
        #calculate intersection
        intersection = np.maximum((max_xy - min_xy)[:,:,0],0) * np.maximum((max_xy - min_xy)[:,:,1],0)
        
        #calculate union
        edge_gt = np.tile(np.expand_dims(gt[:,2:4] - gt[:,0:2], axis = 1), reps = (1,n,1))         
        area_gt = edge_gt[:,:,0] * edge_gt[:,:,1]
        
        edge_anchor_boxes = np.tile(np.expand_dims(anchor_boxes[:,2:4] - anchor_boxes[:,0:2], axis = 0), reps = (m,1,1))         
        area_anchor_boxes = edge_anchor_boxes[:,:,0] * edge_anchor_boxes[:,:,1]

        union = area_gt + area_anchor_boxes - intersection
        
        return intersection / union
    
def match_bipartite_greedy(weight_matrix):
    """
    Calculate the highest matching anchor box per each ground truth
    Input: iou between each ground truth and anchor boxes, shape: (#gt, #anchor boxes)
    Output: List of matched anchor per each ground truth
    """
    m = weight_matrix.shape[0]
    n = weight_matrix.shape[1]
    
    matches = np.zeros(m, dtype = np.int)
    weight_cp = np.copy(weight_matrix)
    
    #Find the largest iou per each ground truth box in descending order
    for _ in range(m):
        largest_indices = np.argmax(weight_cp, axis = 1)
        iou_largest = weight_cp[list(range(m)), largest_indices]
        match_gt = np.argmax(iou_largest, axis = 0)
        match_anchor = largest_indices[match_gt]
        matches[match_gt] = match_anchor
        
        #Set the selected ground truth to 0, matched anchor box to 0 as well.
        weight_cp[match_gt, :] = 0
        weight_cp[:, match_anchor] = 0
        
    return matches    

def match_multi(weight_matrix, threshold):
    """
    Multiple object match
    From remaining anchor boxes, find the most similar ground truth 
    whose iou is greater than pos_threshold
    """
    m = weight_matrix.shape[0]
    n = weight_matrix.shape[1]

    #Find the largest iou per each anchor box
    largest_indices = np.argmax(weight_matrix, axis = 0)
    iou_largest = weight_matrix[largest_indices, list(range(n))]
    
    #Filter iou is greater than the threshold
    matches_anchor = np.nonzero(iou_largest >= threshold)[0].astype(np.int)
    matches_gt = iou_largest[matches_anchor].astype(np.int)
    
    return matches_anchor, matches_gt


def convert_coord(boxes, type='centroid2corner'):
    """
        Input: Input labels 
        type: how to convert
            centroid2corner: (cx, cy, w, h) -> (xmin, ymin, xmax, ymax)
            corner2centroid: (xmin, ymin, xmax, ymax) -> (cx, cy, w, h)    
    """
    
    if type=='centroid2corner':
        cx = boxes[..., -4]
        cy = boxes[..., -3]
        w = boxes[..., -2]
        h = boxes[..., -1]
        
        converted_boxes = np.copy(boxes)
        converted_boxes[..., -4] = cx - w / 2 #xmin
        converted_boxes[..., -3] = cy - h / 2 #ymin
        converted_boxes[..., -2] = cx + w / 2 #xmax
        converted_boxes[..., -1] = cy + h / 2 #ymax
    elif type=='corner2centroid':
        xmin = boxes[..., -4]
        ymin = boxes[..., -3]
        xmax = boxes[..., -2]
        ymax = boxes[..., -1]
        
        converted_boxes = np.copy(boxes)
        
        converted_boxes[..., -4] = (xmin + xmax) / 2 #cx
        converted_boxes[..., -3] = (ymin + ymax) / 2 #cy
        converted_boxes[..., -2] = xmax - xmin #w
        converted_boxes[..., -1] = ymax - ymin #h
        
    return converted_boxes


def _greedy_nms(predictions, iou_threshold=0.45):
    '''
    Non-maximum suppression.
    '''
    boxes_left = np.copy(predictions)
    # This is where we store the boxes that make it through the non-maximum suppression
    maxima = [] 
    # While there are still boxes left to compare...
    while boxes_left.shape[0] > 0: 
         # ...get the index of the next box with the himghest confidence...
        maximum_index = np.argmax(boxes_left[:,0])
        # ...copy that box and...
        maximum_box = np.copy(boxes_left[maximum_index]) 
        # ...append it to `maxima` because we'll definitely keep it
        maxima.append(maximum_box) 
        # Now remove the maximum box from `boxes_left`
        boxes_left = np.delete(boxes_left, maximum_index, axis=0) 
        # If there are no boxes left after this step, break. Otherwise...
        if boxes_left.shape[0] == 0: 
            break 
        # ...compare (IoU) the other left over boxes to the maximum box...
        similarities = calc_iou(boxes_left[:,1:],np.expand_dims(maximum_box[1:],axis=0))
        # ...so that we can remove the ones that overlap too much with the maximum box
        boxes_left = boxes_left[(similarities <= iou_threshold)[:,0]] 
    return np.array(maxima)

def decode_detections(y_pred, n_classes, confidence_thresh=0.01, iou_threshold=0.45, top_k=200, img_height=None, img_width=None, background_id=10):
    # 1: Convert the box coordinates from the predicted anchor box offsets to predicted absolute coordinates

    y_pred_decoded_raw = np.copy(y_pred[:,:,:-4]) # Slice out the classes and the four offsets, throw away the anchor coordinates and variances, resulting in a tensor of shape `[batch, n_boxes, n_classes + 4 coordinates]`

    # exp(ln(w(pred)/w(anchor)) / w_variance * w_variance) == w(pred) / w(anchor), exp(ln(h(pred)/h(anchor)) / h_variance * h_variance) == h(pred) / h(anchor)
    y_pred_decoded_raw[:,:,[-2,-1]] = np.exp(y_pred_decoded_raw[:,:,[-2,-1]]) 
    
    # (w(pred) / w(anchor)) * w(anchor) == w(pred), (h(pred) / h(anchor)) * h(anchor) == h(pred)
    y_pred_decoded_raw[:,:,[-2,-1]] *= y_pred[:,:,[-2,-1]] 
    
    # (delta_cx(pred) / w(anchor) / cx_variance) * cx_variance * w(anchor) == delta_cx(pred), (delta_cy(pred) / h(anchor) / cy_variance) * cy_variance * h(anchor) == delta_cy(pred)
    y_pred_decoded_raw[:,:,[-4,-3]] *= y_pred[:,:,[-2,-1]] 
    
    # delta_cx(pred) + cx(anchor) == cx(pred), delta_cy(pred) + cy(anchor) == cy(pred)
    y_pred_decoded_raw[:,:,[-4,-3]] += y_pred[:,:,[-4,-3]]     
    y_pred_decoded_raw = convert_coord(y_pred_decoded_raw, type='centroid2corner')    
    
    # 2: If the model predicts normalized box coordinates and they are supposed to be converted back to absolute coordinates, do that   
    y_pred_decoded_raw[:,:,[-4,-2]] *= img_width # Convert xmin, xmax back to absolute coordinates
    y_pred_decoded_raw[:,:,[-3,-1]] *= img_height # Convert ymin, ymax back to absolute coordinates

    # 3: Apply confidence thresholding and non-maximum suppression per class

    #n_classes = y_pred_decoded_raw.shape[-1] - 4 # The number of classes is the length of the last axis minus the four box coordinates

    y_pred_decoded = [] # Store the final predictions in this list
    for batch_item in y_pred_decoded_raw: # `batch_item` has shape `[n_boxes, n_classes + 4 coords]`
        pred = [] # Store the final predictions for this batch item here        
        for class_id in range(n_classes): # For each class except the background class 
            if class_id == background_id: continue
            single_class = batch_item[:,[class_id, -4, -3, -2, -1]] # ...keep only the confidences for that class, making this an array of shape `[n_boxes, 5]` and...
            which_box = np.argwhere(single_class[:,0] > confidence_thresh)
            threshold_met = single_class[single_class[:,0] > confidence_thresh] # ...keep only those boxes with a confidence above the set threshold.            
            
            if threshold_met.shape[0] > 0: # If any boxes made the threshold...
                maxima = _greedy_nms(threshold_met, iou_threshold=iou_threshold) # ...perform NMS on them.
                maxima_output = np.zeros((maxima.shape[0], maxima.shape[1] + 1)) # Expand the last dimension by one element to have room for the class ID. This is now an arrray of shape `[n_boxes, 6]`
                maxima_output[:,0] = class_id # Write the class ID to the first column...
                maxima_output[:,1:] = maxima # ...and write the maxima to the other columns...
                pred.append(maxima_output) # ...and append the maxima for this class to the list of maxima for this batch item.
        # Once we're through with all classes, keep only the `top_k` maxima with the highest scores
        if pred: # If there are any predictions left after confidence-thresholding...
            pred = np.concatenate(pred, axis=0)
            if top_k != 'all' and pred.shape[0] > top_k: # If we have more than `top_k` results left at this point, otherwise there is nothing to filter,...
                top_k_indices = np.argpartition(pred[:,1], kth=pred.shape[0]-top_k, axis=0)[pred.shape[0]-top_k:] # ...get the indices of the `top_k` highest-score maxima...
                pred = pred[top_k_indices] # ...and keep only those entries of `pred`...
        else:
            pred = np.array(pred) # Even if empty, `pred` must become a Numpy array.
        y_pred_decoded.append(pred) # ...and now that we're done, append the array of final predictions for this batch item to the output list

    return y_pred_decoded