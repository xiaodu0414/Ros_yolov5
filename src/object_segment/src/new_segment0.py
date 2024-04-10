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
    # print(sys.version_info)
    import Queue
    import thread
else:
    import queue as Queue
    import _thread as thread
    # print(sys.version_info)

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, scale_segments,
                           strip_optimizer)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import masks2segments, process_mask, process_mask_native
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import letterbox


import sys
#sys.path.append("/home/yd/duchangchun/tunnel_monitor/src/")
from object_location_msgs.msg import ObjectImage
from object_location_msgs.msg import ObjectDetect
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from geometry_msgs.msg import Polygon
from geometry_msgs.msg import Point32
import time
import numpy as np
import traceback

q = Queue.Queue(maxsize = 1)   


def init():
    rospy.init_node("object_segment")



def imgmsg_to_cv2(img_msg):
    if img_msg.encoding != "bgr8":
        rospy.logerr(
            "This Coral detect node has been hardcoded to the 'bgr8' encoding.  Come change the code if you're actually trying to implement a new camera")
    dtype = np.dtype("uint8")  # Hardcode to 8 bits...
    dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
    image_opencv = np.ndarray(shape=(img_msg.height, img_msg.width, 3),  # and three channels of data. Since OpenCV works with bgr natively, we don't need to reorder the channels.
                              dtype=dtype, buffer=img_msg.data)
    # If the byt order is different between the message and the system.
    sub_img_cam_id = img_msg.header.frame_id

    print("Received picture id is: ",sub_img_cam_id)
    print("Received picture Width: {}, Height: {} ".format(img_msg.width,img_msg.height))

    if img_msg.is_bigendian == (sys.byteorder == 'little'):
        image_opencv = image_opencv.byteswap().newbyteorder()
    return image_opencv


def callback_receive_img(data):
    try:
        captureimg = imgmsg_to_cv2(data.image)
        q.put((data.object_ids,data.object_types,data.polygons,data.image.width,data.image.height,data.image.header.frame_id,captureimg))
        if q.qsize() > 1:
            q.get() 
        else:
          time.sleep(0.001)

    except Exception:
        traceback.print_exc()


# def callback_receive_img_info(data):
#     try:

#         q.put((data.cam_id, data.objects_ids,data.object_types,data.polygons))     
        
#         if q.qsize() > 1 :
#             q.get()
#         else :
#             time.sleep(0.001)

#     except Exception:
#         traceback.print_exc()


def sub_receive_img():
    #订阅带图节点  只有小车的检测信息   不管有没有检测到小车都会发图片
    rospy.Subscriber("/object_location/trolley_image",ObjectImage,callback_receive_img)

    rospy.spin()
    
def receive_img():
    thread.start_new_thread(sub_receive_img,())



@smart_inference_mode()
def run(
    weights=  './yolov5s-seg.pt',  # model.pt path(s)
    # source=  './data/images',  # file/dir/URL/glob/screen/0(webcam)/
    data=  './data/coco128-seg.yaml',  # dataset.yaml path
    imgsz=(640, 480),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    view_img=False,  # show results
    send_result=True,
    save_txt=False,  # save results to *.txtobject_detection
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
    
    #发布不带图片话题
    object_segment_pub = rospy.Publisher("/object_location/trolley_detection",ObjectDetect,queue_size=1)
    
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)
    bs = 1
    while (not rospy.is_shutdown()):
        try:
            t0 = time.time()
            if q.empty():
                time.sleep(0.01)
                continue
            while not q.empty():
                object_id,types,ploygons,w,h,camera_id, frame = q.get()
                obji = object_id
                tp = types
                ply = ploygons
                cam_id = camera_id
                nw = w
                nh = h
            if frame is None:
                continue
            # if tp == 2:  #如果检测类别等于车的类别那么进行分割
            #     pass
            im0 = frame.copy()
            img = np.array([letterbox(frame, imgsz, stride=stride,auto=pt)[0]])
            # Convert
            img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
            img = np.ascontiguousarray(img)

            print("img.shape",img.shape)
            model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  
            windows = []
            # seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
            im = torch.from_numpy(img).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0                                              #归一qq
            if len(im.shape) == 3:
                im = im[None]
            
            pred, proto = model(im, augment=augment, visualize=visualize)[:2]
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)
            
            for i, det in enumerate(pred):
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    if retina_masks:
                    # scale bbox first the crop masks
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
                        masks = process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2])  # HWC
                    else:
                        masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size
                    #segment
                    if send_result:
                            segments = [
                                scale_segments(im0.shape if retina_masks else im.shape[2:], x, im0.shape, normalize=True)
                                for x in reversed(masks2segments(masks))]
                    #print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum() 

                    #print results
                    annotator.masks(
                            masks,
                            colors=[colors(x, True) for x in det[:, 5]],
                            im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(device).permute(2, 0, 1).flip(0).contiguous() /
                            255 if retina_masks else im[i])

                    msg = ObjectDetect ()
                    msg.object_ids = []
                    msg.object_types = []
                    msg.object_confidences = []
                    msg.header = Header(stamp = rospy.Time.now())
                    # Write results
                    #获取检测消息中的车的检测框
                    xyxy = []
                    for i in range(len(ply)):
                        for point in ply[i].points:
                            xyxy.append(point.x)
                            xyxy.append(point.y)
                    x1,y1,x2,y2 = xyxy[0],xyxy[1],xyxy[2],xyxy[3]
                    print("检测传来的检测框数量:{} ，坐标:{}".format(len(ply),(x1,y1,x2,y2)))

                    for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6])):
                        if send_result:
                            seg = segments[j].reshape(-1)  # (n,2) to (n*2)
                            conf = float(conf)
                            segm = list(seg)
                            pp=1
                            # //cls : 0 , top = 3//cls: 1 , front = 4,//cls : 2 , right_side = 5//  cls: 3, left_side = 6  //cls : 4 ,  behind = 7
                            if cls == 0:                                                                                   
                                msg.object_ids.append(pp)
                                msg.object_types.append(3)
                                msg.object_confidences.append(conf)
                                pp+=1
                                _polygon = Polygon()
                                for x in range(0,len(segm),2):
                                    _point = Point32()
                                    _point.x = segm[x]*nw
                                    _point.y = segm[x+1]*nh
                                    if _point.x>x1 and _point.x<x2 and _point.y>y1 and _point.y<y2:
                                        _polygon.points.append(_point)
                                msg.polygons.append(_polygon)

                            if cls == 1:                                                                                                                     
                                msg.object_ids.append(pp)
                                msg.object_types.append(4)
                                msg.object_confidences.append(conf)
                                _polygon = Polygon()
                                for x in range(0,len(segm),2):
                                    _point = Point32()
                                    _point.x = segm[x]*nw
                                    _point.y = segm[x+1]*nh
                                    if _point.x>x1 and _point.x<x2 and _point.y>y1 and _point.y<y2:
                                        _polygon.points.append(_point)
                                msg.polygons.append(_polygon)                                                                           
 
                            if cls == 2:          
                                msg.object_ids.append(pp)
                                msg.object_types.append(5)
                                msg.object_confidences.append(conf)
                                _polygon = Polygon()
                                for x in range(0,len(segm),2):
                                    _point = Point32()
                                    _point.x = segm[x]*nw
                                    _point.y = segm[x+1]*nh
                                    if _point.x>x1 and _point.x<x2 and _point.y>y1 and _point.y<y2:
                                        _polygon.points.append(_point)
                                msg.polygons.append(_polygon)

                            if cls == 3:                                                                                                                                          
                                msg.object_ids.append(pp)
                                msg.object_types.append(6)
                                msg.object_confidences.append(conf)
                                _polygon = Polygon()
                                for x in range(0,len(segm),2):
                                    _point = Point32()
                                    _point.x = segm[x]*nw
                                    _point.y = segm[x+1]*nh
                                    if _point.x>x1 and _point.x<x2 and _point.y>y1 and _point.y<y2:
                                        _polygon.points.append(_point)
                                msg.polygons.append(_polygon)

                            if cls == 4:                                                                                                                                              
                                msg.object_ids.append(pp)
                                msg.object_types.append(7)
                                msg.object_confidences.append(conf)
                                _polygon = Polygon()
                                for x in range(0,len(segm),2):
                                    _point = Point32()
                                    _point.x = segm[x]*nw
                                    _point.y = segm[x+1]*nh
                                    if _point.x>x1 and _point.x<x2 and _point.y>y1 and _point.y<y2:
                                        _polygon.points.append(_point)
                                msg.polygons.append(_polygon)

                        if view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            # annotator.draw.polygon(segments[j], outline=colors(c, True), width=3)

                    msg.cam_id = cam_id
                    print("camera_id: ",msg.cam_id)
                    print("Id: ",msg.object_ids)
                    print("Confidences",msg.object_confidences)
                    print("Type: ",msg.object_types)
                    print("Polygon: ",msg.polygons)
                    
                    #如果消息列表为空-那么不发送消息
                    if msg.object_ids != []:
                        object_segment_pub.publish(msg)
                    msg.object_ids.clear()
                    msg.object_types.clear()
                    msg.object_confidences.clear()
                    msg.polygons.clear()

    
            # im0 = annotator.result()
            # if view_img:
            #     if platform.system() == 'Linux':
            #         cv2.namedWindow(cam_id, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            #         # cv2.namedWindow(str(p),cv2.WINDOW_AUTOSIZE)
            #         cv2.resizeWindow(cam_id,640,480)
            #     cv2.imshow(cam_id,im0)
            #     # cv2.namedWindow(im0,cv2.WINDOW_AUTOSIZE)
            #     if cv2.waitKey(1) == ord('q'):  # 1 millisecond
            #         exit()
            #stream result
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and cam_id not in windows:
                    windows.append(cam_id)
                    cv2.namedWindow(cam_id, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(cam_id,640,480)
                cv2.imshow(cam_id, im0)
                if cv2.waitKey(1) == ord('q'):  # 1 millisecond
                    exit()
                
        except Exception as e:
            print(e)



def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default= '/home/yd/duchangchun/tunnel_monitor/src/object_segment/src/weights//seg_w/best.pt', help='model path(s)')
    # parser.add_argument('--source', type=str, default='./test/list.streams', help='file/dir/URL/glob/screen/0(webcam)')
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
