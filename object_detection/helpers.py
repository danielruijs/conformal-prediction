"""Helper utilities for Pascal VOC label parsing, bounding box geometry, and detection matching."""

from PIL import Image
from pydantic import BaseModel
from torchvision.datasets import VOCDetection
from ultralytics import YOLO

# Mapping from COCO-80 class names (used by pretrained YOLO) to VOC-20 class names
COCO_TO_VOC_NAME: dict[str, str] = {
    "airplane": "aeroplane",
    "bicycle": "bicycle",
    "bird": "bird",
    "boat": "boat",
    "bottle": "bottle",
    "bus": "bus",
    "car": "car",
    "cat": "cat",
    "chair": "chair",
    "cow": "cow",
    "dining table": "diningtable",
    "dog": "dog",
    "horse": "horse",
    "motorcycle": "motorbike",
    "person": "person",
    "potted plant": "pottedplant",
    "sheep": "sheep",
    "couch": "sofa",
    "train": "train",
    "tv": "tvmonitor",
}

# Derived canonical list of 20 Pascal VOC classes and ID mapping
VOC_CLASSES: list[str] = list(COCO_TO_VOC_NAME.values())
VOC_NAME_TO_ID: dict[str, int] = {name: i for i, name in enumerate(VOC_CLASSES)}


class BoundingBox(BaseModel):
    """Bounding box coordinates in pixel units [x1, y1, x2, y2]."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        """Width of the bounding box."""
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        """Height of the bounding box."""
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        """Area of the bounding box in square pixels."""
        return self.width * self.height


class GroundTruthBox(BaseModel):
    """Ground truth annotation with class ID, class name, and bounding box."""

    class_id: int
    class_name: str
    box: BoundingBox


class Detection(BaseModel):
    """Candidate detection prediction from an object detection model."""

    box: BoundingBox
    cls: int
    class_name: str
    conf: float
    probs: list[float]  # Probability distribution across the 20 VOC classes
    img_w: float
    img_h: float
    img_id: str


class MatchedPair(BaseModel):
    """Matched detection candidate and ground-truth annotation pair."""

    pred: Detection
    gt: GroundTruthBox
    raw_iou: float


def parse_voc_annotation(target: dict) -> list[GroundTruthBox]:
    """Parse a Pascal VOC annotation dict into GroundTruthBox instances.

    Args:
        target: Dictionary returned by VOCDetection[i][1].

    Returns:
        List of GroundTruthBox instances with class ID and pixel bounding box.
    """
    boxes = []
    annotation = target.get("annotation", {})
    objs = annotation.get("object", [])
    if not isinstance(objs, list):
        objs = [objs]

    for obj in objs:
        name = obj.get("name", "").lower()
        if name not in VOC_NAME_TO_ID:
            continue
        cls_id = VOC_NAME_TO_ID[name]
        bnd = obj.get("bndbox", {})
        x1 = float(bnd.get("xmin", 0.0))
        y1 = float(bnd.get("ymin", 0.0))
        x2 = float(bnd.get("xmax", 0.0))
        y2 = float(bnd.get("ymax", 0.0))
        boxes.append(
            GroundTruthBox(
                class_id=cls_id,
                class_name=name,
                box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
            )
        )
    return boxes


def box_iou(box1: BoundingBox, box2: BoundingBox) -> float:
    """Calculate the Intersection-over-Union (IoU) between two 2D bounding boxes.

    Args:
        box1: First bounding box.
        box2: Second bounding box.

    Returns:
        IoU value between 0.0 and 1.0.
    """
    xA = max(box1.x1, box2.x1)
    yA = max(box1.y1, box2.y1)
    xB = min(box1.x2, box2.x2)
    yB = min(box1.y2, box2.y2)

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter = inter_w * inter_h

    area1 = box1.area
    area2 = box2.area
    union = area1 + area2 - inter

    return inter / union if union > 0.0 else 0.0


def box_coverage(pred_box: BoundingBox, gt_box: BoundingBox) -> float:
    """Calculate the coverage (recall) of the ground-truth box by the predicted box.

    Returns the fraction of the ground-truth box area covered by the prediction:
        Area(pred ∩ gt) / Area(gt) in [0.0, 1.0].
    """
    xA = max(pred_box.x1, gt_box.x1)
    yA = max(pred_box.y1, gt_box.y1)
    xB = min(pred_box.x2, gt_box.x2)
    yB = min(pred_box.y2, gt_box.y2)

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter = inter_w * inter_h

    gt_area = gt_box.area
    return inter / gt_area if gt_area > 0.0 else 1.0


def expand_box(
    box: BoundingBox,
    lam: float,
    img_w: float,
    img_h: float,
) -> BoundingBox:
    """Expand a bounding box symmetrically by slack factor lambda.

    Args:
        box: Initial bounding box.
        lam: Relative expansion slack (fraction of box width and height).
        img_w: Image width for clipping.
        img_h: Image height for clipping.

    Returns:
        Expanded BoundingBox clamped to image dimensions.
    """
    w = box.width
    h = box.height
    return BoundingBox(
        x1=max(0.0, box.x1 - lam * w),
        y1=max(0.0, box.y1 - lam * h),
        x2=min(img_w, box.x2 + lam * w),
        y2=min(img_h, box.y2 + lam * h),
    )


def run_voc_inference(
    model: YOLO,
    img: Image.Image,
    img_id: str,
    conf_thresh: float,
) -> list[Detection]:
    """Run YOLO inference on an image and extract structured detections mapped to VOC classes.

    Args:
        model: Loaded YOLO model.
        img: Input PIL Image.
        img_id: Identifier or filename for the image.
        conf_thresh: Minimum confidence threshold.

    Returns:
        List of Detection instances mapped to the 20 VOC classes.
    """
    w, h = img.size
    results = model.predict(img, conf=conf_thresh, verbose=False)[0]

    preds = []
    num_voc = len(VOC_CLASSES)

    for b in results.boxes:
        coco_cls_id = int(b.cls[0].item())
        conf = float(b.conf[0].item())
        coco_name = model.names.get(coco_cls_id, "")
        voc_name = COCO_TO_VOC_NAME.get(coco_name)

        if not voc_name or voc_name not in VOC_NAME_TO_ID:
            continue

        voc_cls_id = VOC_NAME_TO_ID[voc_name]
        xyxy = b.xyxy[0].cpu().numpy().tolist()

        # Construct probability distribution across 20 VOC classes
        # Predicted class receives conf, remaining probability mass is distributed evenly
        rem = max(0.0, (1.0 - conf) / (num_voc - 1))
        probs = [rem] * num_voc
        probs[voc_cls_id] = conf

        preds.append(
            Detection(
                box=BoundingBox(x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3]),
                cls=voc_cls_id,
                class_name=voc_name,
                conf=conf,
                probs=probs,
                img_w=w,
                img_h=h,
                img_id=img_id,
            )
        )
    return preds


def match_detections_to_ground_truth(
    preds: list[Detection],
    gt_items: list[GroundTruthBox],
    min_iou: float,
) -> list[MatchedPair]:
    """Greedily match candidate detections to unassigned ground-truth boxes.

    Args:
        preds: Candidate detections.
        gt_items: Ground truth list of GroundTruthBox instances.
        min_iou: Minimum IoU overlap required for a valid match.

    Returns:
        List of MatchedPair instances.
    """
    matched_pairs = []
    matched_gt_indices = set()

    for p in preds:
        best_iou, best_idx = 0.0, -1
        for g_idx, gt in enumerate(gt_items):
            if g_idx in matched_gt_indices:
                continue
            iou = box_iou(p.box, gt.box)
            if iou > best_iou:
                best_iou, best_idx = iou, g_idx

        if best_idx >= 0 and best_iou >= min_iou:
            matched_gt_indices.add(best_idx)
            gt = gt_items[best_idx]
            matched_pairs.append(
                MatchedPair(
                    pred=p,
                    gt=gt,
                    raw_iou=best_iou,
                )
            )

    return matched_pairs


def extract_voc_matches(
    model: YOLO,
    voc_dataset: VOCDetection,
    indices: list[int],
    conf_thresh: float,
    min_iou: float,
) -> list[MatchedPair]:
    """Extract model detections and match against VOC ground truth for a subset of indices.

    Args:
        model: Loaded YOLO model.
        voc_dataset: Loaded VOCDetection dataset.
        indices: Subset of dataset indices to evaluate.
        conf_thresh: Confidence threshold for predictions.
        min_iou: Minimum IoU overlap for valid match.

    Returns:
        List of MatchedPair records.
    """
    matched_pairs = []

    for idx in indices:
        img, target = voc_dataset[idx]
        img_id = target.get("annotation", {}).get("filename", str(idx))
        gt_items = parse_voc_annotation(target)
        if not gt_items:
            continue

        preds = run_voc_inference(model, img, img_id, conf_thresh)
        if not preds:
            continue

        matches = match_detections_to_ground_truth(preds, gt_items, min_iou)
        matched_pairs.extend(matches)

    return matched_pairs
