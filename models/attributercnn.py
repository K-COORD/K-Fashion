import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import boxes as box_ops
from torch.hub import load_state_dict_from_url
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

params = {
    'rpn_anchor_generator': AnchorGenerator((32, 64, 128, 256, 512), (0.5, 1.0, 2.0)),
    'rpn_pre_nms_top_n_train': 2000,
    'rpn_pre_nms_top_n_test': 1000,
    'rpn_post_nms_top_n_test': 4000,
    'rpn_post_nms_top_n_train': 8000,

    'box_roi_pool': MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=7,
        sampling_ratio=2),
    'mask_roi_pool': MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=14,
        sampling_ratio=2),
}


model_urls = {
    'maskrcnn_resnet50_fpn_coco':
        'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
}



def fastrcnn_loss_with_attr(class_logits, attributes_logits, box_regression, labels, attributes, regression_targets):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.
    Args:
        class_logits (Tensor)
        attributes_logits (list(Tensor))
        box_regression (Tensor)

        labels (list[BoxList])
        attributes (dict(Tensor))
        regression_targets (Tensor)
    Returns:
        classification_loss (Tensor)
        attributes_loss (Tensor)
        box_loss (Tensor)
    """
    labels = torch.cat(labels, dim=0)

    attribute_keys = list(attributes[0].keys())
    attributes_dict = {}    
    for attr_key in attribute_keys:
        attributes_dict[attr_key] = torch.cat([
            one[attr_key] for one in attributes
        ], dim=0)



    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels)

    attribute_loss_collect = []
    for key in attributes_dict.keys():
        gt_attr = attributes_dict[key]
        # print("gt_attr", gt_attr.size())
        if len(gt_attr.size()) == 2:
            sig = F.sigmoid(attributes_logits[key])
            one_attr_loss = F.binary_cross_entropy(sig, gt_attr)
            attribute_loss_collect.append(one_attr_loss)
        elif len(gt_attr.size()) == 1:
            one_attr_loss = F.cross_entropy(attributes_logits[key], gt_attr)
            attribute_loss_collect.append(one_attr_loss)

    attribute_loss = torch.mean(torch.stack(attribute_loss_collect))
    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, attribute_loss, box_loss

class FastRCNNAttributePredictor(torch.nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
        attribute_dict (dict): {
            attribute_name: num_classes,
            ...
        }
    """

    def __init__(self, in_channels, num_classes, attribute_dict):
        super(FastRCNNAttributePredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.attribute_dict = attribute_dict
        
        for attribute_name, (mult_or_single, attribute_classes) in attribute_dict.items():
            setattr(self, f"{attribute_name}_score", nn.Linear(in_channels, attribute_classes))

        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)

        attribute_scores = {}
        for attribute_name, (mult_or_single, attribute_classes) in self.attribute_dict.items():
            attr_score = getattr(self, f"{attribute_name}_score")
            # print("nnattr_score", attr_score)
            one_attr_score = attr_score(x)
            # print("one_attr_score", one_attr_score.size())
            attribute_scores[attribute_name] = one_attr_score

        
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return scores, attribute_scores, bbox_deltas

class NewRoIHeadsAttributes(torch.nn.Module):
    def __init__(self, orh, num_classes, attribute_dict):
        # orh: old_roi_heads
        super(NewRoIHeadsAttributes, self).__init__()

        self.box_roi_pool = orh.box_roi_pool
        self.box_head = orh.box_head

        inchannels = orh.box_predictor.cls_score.in_features # 1024
        # self.box_predictor = orh.box_predictor
        self.box_predictor = FastRCNNAttributePredictor(inchannels, num_classes, attribute_dict)

        self.score_thresh = orh.score_thresh
        self.nms_thresh = orh.nms_thresh
        self.detections_per_img = orh.detections_per_img

        self.proposal_matcher = orh.proposal_matcher
        self.fg_bg_sampler = orh.fg_bg_sampler
        self.box_coder = orh.box_coder
        self.box_similarity = box_ops.box_iou

    @property
    def has_mask(self):
        if self.mask_roi_pool is None:
            return False
        if self.mask_head is None:
            return False
        if self.mask_predictor is None:
            return False
        return True

    @property
    def has_keypoint(self):
        if self.keypoint_roi_pool is None:
            return False
        if self.keypoint_head is None:
            return False
        if self.keypoint_predictor is None:
            return False
        return True

    @property
    def has_match(self):
        if self.match_predictor is None:
            return False
        if self.match_loss is None:
            return False
        return True

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels, gt_attributes):
        matched_idxs = []
        labels = []
        attributes = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image, gt_attributes_in_image in zip(proposals, gt_boxes, gt_labels, gt_attributes):
            match_quality_matrix = self.box_similarity(gt_boxes_in_image, proposals_in_image)
            matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

            clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

            labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]

            bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
            labels_in_image[bg_inds] = 0

            ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
            labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)

            attributes_in_image = {}
            for key, value in gt_attributes_in_image.items():
                attributes_in_image[key] = value[clamped_matched_idxs_in_image]

            attributes.append(attributes_in_image)
        return matched_idxs, labels, attributes

    def subsample(self, labels):
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_inds = []
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
                zip(sampled_pos_inds, sampled_neg_inds)
        ):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            sampled_inds.append(img_sampled_inds)
        return sampled_inds

    def add_gt_proposals(self, proposals, gt_boxes):
        proposals = [
            torch.cat((proposal, gt_box))
            for proposal, gt_box in zip(proposals, gt_boxes)
        ]

        return proposals

    def select_training_samples(self, proposals, targets):
        self.check_targets(targets)
        gt_boxes = [t["boxes"] for t in targets]
        gt_labels = [t["labels"] for t in targets]
        gt_attributes = [t["attributes"] for t in targets]

        # append ground-truth bboxes to propos
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels, attributes = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels, gt_attributes)
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]

            for key, value in attributes[img_id].items():
                attributes[img_id][key] = value[img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]
            matched_gt_boxes.append(gt_boxes[img_id][matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, attributes, regression_targets


    def postprocess_detections(self, class_logits, box_regression, proposals, image_shapes):
        device = class_logits.device
        num_classes = class_logits.shape[-1]

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)

        # split boxes and scores per image
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_scores = pred_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        for boxes, scores, image_shape in zip(pred_boxes, pred_scores, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.flatten()
            labels = labels.flatten()

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[:self.detections_per_img]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

        return all_boxes, all_scores, all_labels

    def forward(self, features, proposals, image_shapes, targets=None):
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                assert t["boxes"].dtype.is_floating_point, 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
               
        if self.training:
            proposals, matched_idxs, labels, attributes, regression_targets = self.select_training_samples(proposals, targets)

        box_features_roi = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features_roi)
        class_logits, attribute_logits, box_regression = self.box_predictor(box_features)

        # print("attribute_logits", attribute_logits)
        result, losses = [], {}
        if self.training:
            loss_classifier, loss_attributes, loss_box_reg = fastrcnn_loss_with_attr(
                class_logits, attribute_logits, box_regression, labels, attributes, regression_targets)
            losses = dict(loss_classifier=loss_classifier, loss_box_reg=loss_box_reg, loss_attributes=loss_attributes)
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                if boxes[i].numel() > 0:
                    result.append(
                        dict(
                            boxes=boxes[i],
                            labels=labels[i],
                            scores=scores[i],
                        )
                    )
                else:
                    result.append(
                        dict(
                            boxes=torch.tensor([0.0, 0.0, image_shapes[i][1], image_shapes[i][0]]).to(
                                boxes[i].device).unsqueeze(0),
                            labels=torch.tensor([0]).to(boxes[i].device),
                            scores=torch.tensor([1.0]).to(boxes[i].device),
                        )
                    )

        return result, losses


class MatchRCNN(MaskRCNN):
    def __init__(self, backbone, num_classes, **kwargs):
        super(MatchRCNN, self).__init__(backbone, num_classes, **kwargs)
        self.orh = self.roi_heads
        self.roi_heads = NewRoIHeads(self.orh)

    def load_saved_matchrcnn(self, sd, new_num_classes, attribute_dict):
        # load the deepfashion2 checkpoint first
        self.load_state_dict(sd, strict=False)

        # then create new heads
        self.roi_heads = NewRoIHeadsAttributes(self.orh, new_num_classes, attribute_dict)



def matchrcnn_resnet50_fpn(pretrained=False, progress=True,
                           num_classes=14, pretrained_backbone=True, **kwargs):
    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False
    backbone = resnet_fpn_backbone('resnet50', pretrained_backbone)
    model = MatchRCNN(backbone, num_classes, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['maskrcnn_resnet50_fpn_coco'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
