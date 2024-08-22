# limit the number of cpus used by high performance libraries
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')

import argparse
import os
import platform
import shutil
import numpy as np
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

from FusionSORT.FusionSORT import FusionSORT
from dataset.self_dataset import self_dataset

from Detection.yolov5.models.experimental import attempt_load
from Detection.yolov5.utils.downloads import attempt_download
from Detection.yolov5.models.common import DetectMultiBackend
from Detection.yolov5.utils.datasets import LoadImages, LoadStreams
from Detection.yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords,
                                  check_imshow, xyxy2xywh, increment_path)
from Detection.yolov5.utils.torch_utils import select_device, time_sync
from Detection.yolov5.utils.plots import Annotator, colors

# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]  # yolov5 deepsort root directory
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def run(opt):
    out, source, yolo_model, reid_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, project, name, exist_ok= \
        opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.name, opt.exist_ok

    # initialize deepsort
    # cfg = get_config()
    # cfg.merge_from_file(opt.config_deepsort)
    FusionSort = FusionSORT(reid_model, use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir
    print('save_dir', save_dir)

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()


    # Dataloader
    dataset = self_dataset(source, img_size=imgsz, stride=stride, auto=pt and not jit)
    bs = 1  # batch_size
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(save_dir)) + '/' + txt_file_name + '.txt'

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    fusion_num = 0
    for frame_idx, (rgb_img0, img, TI_data, lidar_annotation) in enumerate(dataset):
        # print('\nframe_idx:', frame_idx)
        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
        # img:[1, 3, 384, 640]
        pred = model(img, augment=opt.augment, visualize=False)  # torch.Size([1, 15120, 85])
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                   max_det=opt.max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1

            im0 = rgb_img0.copy()

            annotator = Annotator(im0, line_width=2, pil=not ascii)
            # print("img:", img.shape, "        im0:", im0.shape)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()
                # # Print results
                # for c in det[:, -1].unique():
                #     n = (det[:, -1] == c).sum()  # detections per class
                #     # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                #     # print("n:", n, "        s:", s, '\n')
                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # outputs: xyxy,cls. detections: confidence, feature, tlwh
                outputs, detections = FusionSort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                print('outputs:', outputs)

                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)):

                        bboxes = output[0:4]
                        cls = output[4]
                        c = int(cls)  # integer class
                        # label = f'{names[c]} {conf:.2f}'
                        label = ''
                        # annotator.box_label(bboxes, label, color=colors(c, True))
                        annotator.box_label(bboxes, label, color=(255, 0, 0))
                        # save_txt = True
                        # if save_txt:
                        #     # to MOT format
                        #     bbox_left = output[0]
                        #     bbox_top = output[1]
                        #     bbox_w = output[2] - output[0]
                        #     bbox_h = output[3] - output[1]
                        #     # Write MOT compliant results to file
                        #     with open(txt_path, 'a') as f:
                        #         f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                        #                                        bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))
            else:
                LOGGER.info('No detections')

            # Stream results
            im0 = annotator.result()
            # if cv2.waitKey(0) & 0xFF == 27:
            #     break
            cv2.imshow('test', im0)
            cv2.waitKey(0)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x1_0')   # osnet_x0_25
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', default=True, action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default='runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    # opt.source = 'video/twopeople.mp4'
    # opt.source = 'video/206.mp4'
    opt.source = '/media/ourDataset/v1.0_label/20211025_1_group0012_185frames_37labeled'
    # opt.source = '/media/ourDataset/v1.0_label/20211027_1_group0010_148frames_30labeled'
    # opt.source = '/media/personal_data/zhangq/DeepSORT/Yolov5_DeepSort_Pytorch/val_utils/data/MOT17/train/MOT17-04-FRCNN/MOT17-04-FRCNN'
    opt.yolo_model = 'yolov5/weights/yolov5m.pt'
    opt.deep_sort_model = 'osnet_ain_x1_0'
    opt.classes = 2
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    with torch.no_grad():
        run(opt)