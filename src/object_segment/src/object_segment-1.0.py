# coding=utf-8
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 segmentation inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python object_segment.py --weights yolov5s-seg.pt --source 0                               # webcam
                                                                  img.jpg                         # image
                                                                  vid.mp4                         # video
                                                                  screen                          # screenshot
                                                                  path/                           # directory
                                                                  list.txt                        # list of images
                                                                  list.streams                    # list of streams
                                                                  'path/*.jpg'                    # glob
                                                                  'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                                  'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

"""

import argparse
import os
import platform
import sys
from pathlib import Path
import numpy

import torch
import rospy 
import queue
from cv_bridge import CvBridge
import os.path as osp

import sys
if sys.version_info < (3, 0):
    import Queue
    import thread
else:
    import queue as Queue
    import _thread as thread

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import masks2segments, process_mask, process_mask_native
from utils.torch_utils import select_device, smart_inference_mode

import sys
#sys.path.append("/home/yd/duchangchun/tunnel_monitor/src/")
from object_location_msgs.msg import ObjectDetect
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from geometry_msgs.msg import Polygon
from geometry_msgs.msg import Point32
import time
import numpy as np
import traceback

q = Queue.Queue(maxsize = 1)   ## dui lie da yu 2 de, zhi jie bu guan
q_pose = Queue.Queue()
q_msg = Queue.Queue()

def imgmsg_to_cv2(img_msg):
    if img_msg.encoding != "bgr8":
        rospy.logerr(
            "This Coral detect node has been hardcoded to the 'bgr8' encoding.  Come change the code if you're actually trying to implement a new camera")
    dtype = np.dtype("uint8")  # Hardcode to 8 bits...
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3),  # and three channels of data. Since OpenCV works with bgr natively, we don't need to reorder the channels.
                              dtype=dtype, buffer=img_msg.data)
    # If the byt order is different between the message and the system.
    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        image_opencv = image_opencv.byteswap().newbyteorder()
    return image_opencv

def init():
    rospy.init_node("object_segment")


def callback_receive_img(data):
    try:
        captureimg = imgmsg_to_cv2(data)
        q.put((0,captureimg,0,rospy.Time.now()))
        q.get() if q.size() > 1 else time.sleep(0.001)
        cv2.imshow(captureimg)
        cv2.waitKey(3)
    except Exception:
        traceback.print_exc()
        
def sub_receive_img():
    rospy.Subscriber("/object_location/object_detection_images",Image,callback_receive_img)
    rospy.spin()
    
def receive_img():
    thread.start_new_thread(sub_receive_img,())
    print("接收图片信息")




@smart_inference_mode()
def run(
    weights=  './yolov5s-seg.pt',  # model.pt path(s)
    source=  './data/images',  # file/dir/URL/glob/screen/0(webcam)
    data=  './data/coco128-seg.yaml',  # dataset.yaml path
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    send_result=False,
    save_txt=False,  # save results to *.txt
    save_conf=False,  # save confidences in --save-txt labels
    save_crop=False,  # save cropped prediction boxes
    nosave=False,  # do not save images/videos
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    augment=False,  # augmented inference
    visualize=False,  # visualize features
    update=False,  # update all models
    project=  './runs/predict-seg',  # save results to project/name
    name='exp',  # save results to project/name
    exist_ok=False,  # existing project/name ok, do not increment
    line_thickness=3,  # bounding box thickness (pixels)
    hide_labels=False,  # hide labels
    hide_conf=False,  # hide confidences
    half=False,  # use FP16 half-precision inference
    dnn=False,  # use OpenCV DNN for ONNX inference
    vid_stride=1,  # video frame-rate stride
    retina_masks=False,
):



    init()
    receive_img()
    object_segment_pub = rospy.Publisher("/object_location/object_detection",ObjectDetect,queue_size=1)
    object_segment_images_pub = rospy.Publisher("/object_location/object_detection_images",Image,queue_size=1)
    
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    # print("size:",imgsz)

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        camera_ids = dataset. camera_ids
        whs = dataset.whs
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    # print("start for path, im, im0s, vid_cap, s in dataset: ")
    for path, im, im0s, vid_cap, s in dataset:
        # print(f"path:{path},vid_cap:{vid_cap},s:{s}")
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0                                              #归一
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred, proto = model(im, augment=augment, visualize=visualize)[:2]

        

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        # Process predictions
        for i, det in enumerate(pred):  # per image
            camera_id = camera_ids[i]
            yw,yh = whs[i]


            # print("从dataset中取出的返回值",camera_id,w,h)
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
                
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string

            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                if retina_masks:
                    # scale bbox first the crop masks
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
                    masks = process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2])  # HWC
                else:
                    masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

                # Segments
                if send_result:
                    segments = [
                        scale_segments(im0.shape if retina_masks else im.shape[2:], x, im0.shape, normalize=True)
                        for x in reversed(masks2segments(masks))]
                        
                if save_txt:
                    segments = [
                        scale_segments(im0.shape if retina_masks else im.shape[2:], x, im0.shape, normalize=True)
                        for x in reversed(masks2segments(masks))]

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    # print("n",n)
                    # print("类别",int(c))
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    # print(33333333333333333,s)
               

                # Mask plotting
                annotator.masks(
                    masks,
                    colors=[colors(x, True) for x in det[:, 5]],
                    im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(device).permute(2, 0, 1).flip(0).contiguous() /
                    255 if retina_masks else im[i])

                
                msg = ObjectDetect ()
                msg.object_ids = []
                msg.object_types = []
                msg.header = Header(stamp = rospy.Time.now())
                # Write results
                for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                    if send_result:
                        seg = segments[j].reshape(-1)  # (n,2) to (n*2)
                        segm = list(seg)
                        pp = 1
                        if cls == 0:            
                                                                                                                                            #top
                            msg.object_ids.append(pp)
                            pp+=1
                            _polygon = Polygon()
                            for x in range(0,len(segm),2):
                                _point = Point32()
                                _point.x = segm[x]*yw
                                _point.y = segm[x+1]*yh
                                _polygon.points.append(_point)
                                # print(j,len(_polygon.points))

                            msg.object_types.append(int(cls))
                            msg.polygons.append(_polygon)
                        
                        if cls == 1:        
                                                                                                                                            ##front
                            msg.object_ids.append(pp)
                            _polygon = Polygon()
                            for x in range(0,len(segm),2):
                                _point = Point32()
                                _point.x = segm[x]*yw
                                _point.y = segm[x+1]*yh
                                _polygon.points.append(_point)
                                # print(j,len(_polygon.points))

                            msg.object_types.append(int(cls))
                            msg.polygons.append(_polygon)
                            
                        if cls == 2:                   
                                                                                                                                ## right_side
                            msg.object_ids.append(pp)
                            _polygon = Polygon()
                            for x in range(0,len(segm),2):
                                _point = Point32()
                                _point.x = segm[x]*yw
                                _point.y = segm[x+1]*yh
                                _polygon.points.append(_point)
                                # print(j,len(_polygon.points))

                            msg.object_types.append(int(cls))
                            msg.polygons.append(_polygon)

                        if cls == 3:                    
                                                                                                                                ## left_side
                            msg.object_ids.append(pp)
                            _polygon = Polygon()
                            # print("camera_id segm: ",segm)
                            for x in range(0,len(segm),2):
                                _point = Point32()
                                _point.x = segm[x]*yw
                                _point.y = segm[x+1]*yh
                                _polygon.points.append(_point)

                            msg.object_types.append(int(cls))
                            msg.polygons.append(_polygon)

                        if cls == 4:
                                                                                                                                            ## behind
                            msg.object_ids.append(pp)
                            _polygon = Polygon()
                            for x in range(0,len(segm),2):
                                _point = Point32()
                                _point.x = segm[x]*yw
                                _point.y = segm[x+1]*yh
                                _polygon.points.append(_point)
                                # print(j,len(_polygon.points))

                            msg.object_types.append(int(cls))
                            msg.polygons.append(_polygon)    
                

                if save_txt:  # Write to file
                    seg = segments[j].reshape(-1)  # (n,2) to (n*2)False
                    line = (cls, *seg, conf) if save_conf else (cls, *seg)  # label format
                    with open(f'{txt_path}.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        

                if save_img or save_crop or view_img:  # Add bbox to image
                    c = int(cls)  # integer class
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    # annotator.draw.polygon(segments[j], outline=colors(c, True), width=3)
                if save_crop:
                    save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)


                    # msg.image = numpy.array(cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,30])[1].tobyteSs())
                Timg = Image()
                Timg.header.frame_id = "1"
                Timg.width = im0.shape[1]
                Timg.height = im0.shape[0]
    
                Timg.encoding = "bgr8"
                Timg.is_bigendian = 0
                Timg.data = im0.tobytes()
                Timg.step = len(Timg.data)

                msg.cam_id = str(camera_id)

                

                print("camera_id: ",msg.cam_id)
                print("Id: ",msg.object_ids)
                print("Type: ",msg.object_types)
                # print("Polygon: ",msg.polygons)
                object_segment_pub.publish(msg)
                object_segment_images_pub.publish(Timg)


                msg.object_ids.clear()
                msg.object_types.clear()
                msg.polygons.clear()
                    

            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    # cv2.namedWindow(str(p),cv2.WINDOW_AUTOSIZE)
                    cv2.resizeWindow(str(p), 640,480)
                cv2.imshow(str(p), im0)
                # cv2.namedWindow(im0,cv2.WINDOW_AUTOSIZE)
                if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                    exit()

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
        
    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default= '/home/yd/duchangchun/tunnel_monitor/src/object_segment/src/weights/best.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='./test/list.streams', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default= './test/data/coco128-seg.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.8, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default = True,action='store_true', help='show results')
    parser.add_argument('--send-result', default = True,action='store_true', help='ros send results')
    parser.add_argument('--save-txt',default=False,action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', default = True,action='store_true', help='do not sav1qe images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default= './runs/predict-seg', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', default = True,action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--retina-masks', action='store_true', help='whether to plot masks in native resolution')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    # check_requirements(exclude=('tensorboard', 'thop'))

   

    run(**vars(opt))

    
    


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
