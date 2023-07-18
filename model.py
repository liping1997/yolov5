from models.common import DetectMultiBackend
from utils import torch_utils
from cut import cut
import torch
import numpy as np
from utils.augmentations import letterbox
from utils.general import non_max_suppression,scale_boxes
from utils.plots import Annotator,colors
import cv2
from control import mouse_xy,click_mouse_button
import win32gui
import time

weights = './best.pt'
device = torch_utils.select_device(0)
model = DetectMultiBackend(weights, device=device,  data='./data/valorant.yaml')
names=model.names
num=0

while True:

    im0=cut()
    save_path='../img/{}.png'.format(num)
    num+=1
    im = letterbox(im0, [640,640], stride=32, auto=True)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    im = torch.from_numpy(im).to(model.device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim


    pred = model(im, augment=False, visualize=False)

    pred = non_max_suppression(pred, 0.25, 0.45, 1+3, False, max_det=9)




    for i, det in enumerate(pred):

        im0 = np.ascontiguousarray(im0)
        annotator = Annotator(im0, line_width=3, example=str(names))
        if len(det):
                # 将边界框从img_size缩放到im0的尺寸
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                column_six=det[:,4]
                max_value,max_index=torch.max(column_six,dim=0)
                row_with_max_value=det[max_index]
                x, y = win32gui.GetCursorPos()
                mouse_xy(int((row_with_max_value[0]+row_with_max_value[2])/2-x),int((row_with_max_value[1]+row_with_max_value[3])/2)-y)

                # click_mouse_button()
                # click_mouse_button()
                # click_mouse_button()


    # for i, det in enumerate(pred):
    #     imc=im0.copy()
    #     im0 = np.ascontiguousarray(im0)
    #     annotator = Annotator(im0, line_width=3, example=str(names))
    #     if len(det):
    #             # 将边界框从img_size缩放到im0的尺寸
    #             det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
    #             for *xyxy, conf, cls in reversed(det):
    #                 c = int(cls)  # integer class
    #                 label =  f'{names[c]} {conf:.2f}'
    #                 annotator.box_label(xyxy, label, color=colors(c, True))
    #     im0 = annotator.result()
    #     cv2.imwrite(save_path, im0)