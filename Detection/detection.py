# vim: expandtab:ts=4:sw=4
import torch
import numpy as np
from pathlib import Path
import cv2
import os

# import sys
# sys.path.insert(0, './yolov5')
from .yolov5.models.common import DetectMultiBackend
from .yolov5.utils.datasets import LoadImages, LoadStreams
from .yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
                                  check_imshow, xyxy2xywh, increment_path)
from .yolov5.utils.torch_utils import select_device, time_sync
from .yolov5.utils.plots import Annotator, colors
from .deep.feature_extractor import Extractor
from TiProcess.proj_radar2cam import cam_to_radar, cam_to_radar2
from Tools.GetImgDepth import getImgDepth

# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]  # yolov5 deepsort root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh), shape   # shape [height, width]

def img_preprocess(raw_img, img_size=[640], stride=32, auto=False):

    # rgb_img0 = cv2.resize(rgb_img, (1280, 720))
    img, ratio, pad, raw_shape = letterbox(raw_img, img_size, stride=stride, auto=auto)
    img_size *= 2 if len(img_size) == 1 else 1  # expand
    # Stack
    img = np.stack(img, 0)
    # Convert
    img = img.transpose((2, 0, 1))  # BHWC to BCHW
    # img = img[..., ::-1].transpose((2, 0, 1))  # BGR to RGB, BHWC to BCHW
    img = np.ascontiguousarray(img)

    return raw_img, img, ratio, pad, raw_shape

def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


class Detection(object):
    """
    fusion_state:
        0: no target
        1: only_camera
        2: only_radar
        3: fusion_sensors
    """
    def __init__(self, center=None, fusion_state=0, tlwh=np.zeros(shape=(4,)), proj_xy=np.zeros(shape=(2,)), classes=None, confidence=0,
                 feature=np.zeros(shape=(1, 512)), r_center=None, r_feature=np.zeros(shape=(10,)),
                 footpoint=np.zeros(shape=(2,))):
        """
        :param box:
        if image, box is target size in image, tlwh
        :param classes:
        if contain image information, it is the class of the target
        :param feature:
        if contain image information, it is the image vector of the target
        :param confidence:
        if contain image information, it is the confidence of detection result
        :param center:
        radar coordinate: the center of point cloud
        :param density:
        if contain point cloud, it's cluster density of point cloud. it's calculated by number of points in cluster
        :param radius:
        size of point cloud, [x, y]
        :param velocity:
        if point cluster, the mean velocity of point cloud, and it's radial
        :param state:
        the assign state, if only cam, state = 1; if only radar, state = 2; if cam + radar, state = 3
        :param intensity:
        if radar, the intensity of radar point (SNR+NOISE)
        """
        self.center = center
        self.fusion_state = fusion_state
        self.tlwh = tlwh
        self.proj_xy = proj_xy
        self.classes = classes
        self.feature = feature
        self.confidence = confidence
        self.r_center = r_center
        self.r_feature = r_feature
        self.footpoint = footpoint

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret


class YOLO_reid_model(object):
    def __init__(self, yolo_model=None, Reid_model=None, device=0, imgsz=[640], classes=2, conf_thres=0.3,
                 iou_thres=0.5, max_det=1000, half=True, augment=True, agnostic_nms=True, dnn=True):

        self.detection_model = yolo_model
        self.reid_model = Reid_model
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.max_det = max_det
        self.half = half

        self.extractor = Extractor(self.reid_model, use_cuda=True)


        # Initialize
        self.device = select_device(device)
        self.half &= self.device.type != 'cpu'  # half precision only supported on CUDA
        self.augment = augment

        # Load model
        self.model = DetectMultiBackend(self.detection_model, device=self.device, dnn=dnn)
        stride, names, pt, jit, _ = self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx
        imgsz *= 2 if len(imgsz) == 1 else 1  # expand
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Half
        self.half &= pt and self.device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
        if pt:
            self.model.model.half() if self.half else self.model.model.float()

        # Get names and colors
        names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        # extract what is in between the last '/' and last '.'
        # txt_file_name = source.split('/')[-1].split('.')[0]
        # txt_path = str(Path(save_dir)) + '/' + txt_file_name + '.txt'

        if pt and self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, *imgsz).to(self.device).type_as(next(self.model.model.parameters())))  # warmup
        self.dt, self.seen = [0.0, 0.0, 0.0, 0.0], 0

    def get_detections(self, rgb_img0, img, rgb_json, OCU_json, OCU_data):

        confidence_threshold = 0.3
        # rgb_img0, img, ratio, pad, raw_shape = img_preprocess(rgb_img0)

        t1 = time_sync()
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        self.dt[0] += t2 - t1

        # Inference
        # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
        # img:[1, 3, 384, 640]
        pred = self.model(img, augment=self.augment, visualize=False)  # torch.Size([1, 15120, 85])
        t3 = time_sync()
        self.dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                   max_det=self.max_det)
        # pred = pred[pred[:, 4] > confidence_threshold]
        self.dt[2] += time_sync() - t3
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            self.seen += 1

            im0 = rgb_img0.copy()
            annotator = Annotator(im0, line_width=2, pil=not ascii)
            # print("img:", img.shape, "        im0:", im0.shape)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()
                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]
                confs_cpu = confs.cpu()
                clss_cpu = clss.cpu()
                # outputs, detections = FusionSort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                self.height, self.width = im0.shape[:2]


                # generate detections
                features = self._get_features(xywhs.cpu(), im0)
                # print('features:', features)
                bbox_tlwh = self._xywh_to_tlwh(xywhs.cpu()) # (N, 4)
                # print('bbox_tlwh', bbox_tlwh)
                IntrinsicMatrix = np.array(rgb_json['intrinsic'])
                OCU2img_Matrix = np.array(OCU_json['OCULiiRadar_to_LeopardCamera0_extrinsic'])

                proj_xy = cam_to_radar2(bbox_tlwh, OCU2img_Matrix, IntrinsicMatrix)
                # proj_xy = getImgDepth(xywhs.cpu(), rgb_json, OCU_json, OCU_data)
                # print('proj_xy', proj_xy, '\n')

                # proj_xy = cam_to_radar2(bbox_tlwh, TI2img_Matrix, IntrinsicMatrix)
                # outputs: xyxy,cls. detections: confidence, feature, tlwh
                outputs = self.get_output(bbox_tlwh, clss_cpu, confs_cpu)
                detections = []

                for i in range(len(confs_cpu)):
                    if confs_cpu[i] < confidence_threshold:
                        continue
                    # if proj_xy[i][0] >= 20 or proj_xy[i][0] <= -10 or proj_xy[i][1] >= 40 or proj_xy[i][1] <= 0:
                    #     continue
                    if proj_xy[i][0] >= 10 or proj_xy[i][0] <= -10 or proj_xy[i][1] > 50 or proj_xy[i][1] < 0:
                        continue
                    # print('********features[i].shape*********', features[i].shape)
                    detections.append(Detection(center=proj_xy[i], fusion_state=1, tlwh=bbox_tlwh[i],
                                                classes=clss_cpu[i], proj_xy=proj_xy[i], confidence=confs_cpu[i],
                                                feature=features[i].reshape(1, 512)))

                # detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(
                #     confs.cpu())]
                # print('outputs:\n', outputs)

                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)):
                        bboxes = output[0:4]
                        cls = output[4]
                        c = int(cls)  # integer class
                        # label = f'{names[c]} {conf:.2f}'
                        label = ''
                        # annotator.box_label(bboxes, label, color=colors(c, True))
                        annotator.box_label(bboxes, label, color=(255, 0, 0))
                        im0 = annotator.result()
                        # Stream results
                        # cv2.imshow('test', im0)
                        # # cv2.imshow('test', cv2.resize(im0, (1080, 720)))
                        # cv2.waitKey(0)
                if len(detections) > 0:
                        return outputs, detections, im0
                else:
                    # LOGGER.info('No detections')
                    return [], [], im0
            else:
                # LOGGER.info('No detections')
                return [], [], im0

        print('')

    # bbox_xyxy:(n,4)
    def get_output(self, tlwhs, classes, confs_cpu, confidence_threshold=0.5):
        # generate detections

        # output bbox identities
        outputs = []
        for i in range(tlwhs.shape[0]):
            if confs_cpu[i] < confidence_threshold:
                continue
            x1, y1, x2, y2 = self._tlwh_to_xyxy(tlwhs[i])
            # outputs.append(np.array([x1, y1, x2, y2, classes[i]],
            #                         dtype=np.int64))
            # outputs.append(np.array([int(x1), int(y1), int(x2), int(y2), int(classes[i]), confs_cpu[i]]))
            # tmp = np.array([x1, y1, x2, y2, classes[i], confs_cpu[i]])
            tmp = np.array([x1, y1, x2, y2, classes[i], confs_cpu[i]])
            tmp[0: 5].astype(np.int64)
            outputs.append(tmp)

        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs


    def get_GT_detections(self, rgb_img0, rgb_json, TI_json):

        rgb_annotations = rgb_json['annotation']
        detections = []
        # x, y, w, h, cls = [], [], [], [], []
        bbox_tlwh = []
        im0 = rgb_img0.copy()
        self.height, self.width = im0.shape[:2]

        for idx in range(len(rgb_annotations)):
            if rgb_annotations[idx]['class'] != 'car':
                continue
            if rgb_annotations[idx]['x'] < 0 or rgb_annotations[idx]['x'] > 960 \
                    or rgb_annotations[idx]['y'] < 0 or rgb_annotations[idx]['y'] > 510\
                    or rgb_annotations[idx]['w'] < 25 or rgb_annotations[idx]['h'] < 25:
                continue
            x = rgb_annotations[idx]['x']
            y = rgb_annotations[idx]['y']
            w = rgb_annotations[idx]['w']
            h = rgb_annotations[idx]['h']
            bbox_tlwh.append([x, y, w, h])

        bbox_tlwh = np.array(bbox_tlwh)
        # print('bbox_tlwh', bbox_tlwh)
        IntrinsicMatrix = np.array(rgb_json['intrinsic'])
        TI2img_Matrix = np.array(TI_json['OCULiiRadar_to_LeopardCamera0_extrinsic'])
        if len(bbox_tlwh) > 0:
            bbox_xywh = self._tlwh_to_xywh(bbox_tlwh)
            # bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
            features = self._get_features(bbox_xywh, im0)

            proj_xy = cam_to_radar(bbox_tlwh, TI2img_Matrix, IntrinsicMatrix)
            # print('proj_xy', proj_xy)

            for i in range(bbox_tlwh.shape[0]):
                if proj_xy[i][0] >= 20 or proj_xy[i][0] <= -10 or proj_xy[i][1] >= 40 or proj_xy[i][1] <= 0:
                    continue
                # print('********features[i].shape*********', features[i].shape)
                detections.append(Detection(center=proj_xy[i], fusion_state=1, tlwh=bbox_tlwh[i],
                                            classes=2, proj_xy=proj_xy[i], confidence=0.99,
                                            feature=features[i].reshape(1, 512)))

        return detections

    """
        TODO:
            Convert bbox from xc_yc_w_h to xtl_ytl_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _tlwh_to_xywh(self, bbox_tlwh):
        if isinstance(bbox_tlwh, np.ndarray):
            bbox_xywh = bbox_tlwh.copy()
        elif isinstance(bbox_tlwh, torch.Tensor):
            bbox_xywh = bbox_tlwh.clone()
        bbox_xywh[:, 0] = bbox_xywh[:, 0] + bbox_xywh[:, 2] / 2.
        bbox_xywh[:, 1] = bbox_xywh[:, 1] + bbox_xywh[:, 3] / 2.
        return bbox_xywh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x + w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y + h), self.height - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features


def run():

    print()

if __name__ == '__main__':
    run()
