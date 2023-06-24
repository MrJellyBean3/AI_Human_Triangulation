import cv2
import matplotlib.pyplot as plt
import keyboard
import time
import multiprocessing
import numpy as np


def triangulate(c1_raw,c2_raw):
    c1_2=0
    c1_4=0
    c2_2=0
    c2_4=0
    const_mult=0
    c2_offset=0

    cam_distance_m=0.85 #distance between cameras in meters, you will need to replace this with the distance between your cameras
    c1_2=447   #camera 1 2 meters away pixel count, you will need to replace this with your value as well as the others below
    c1_4=394   #camera 1 4 meters away pixel count
    c2_2=256   #camera 2 2 meters away pixel count
    c2_4=294   #camera 2 4 meters away pixel count

    angle_4=np.tan((cam_distance_m/2)/4)
    angle_2=np.tan((cam_distance_m/2)/2)

    c2_offset=-((c1_4-c2_4)-(angle_4/angle_2)*(c1_2-c2_2))/((angle_4/angle_2)-1)
    const_mult=(angle_4/(abs(c1_4-(c2_4-c2_offset))/2))
    angle=const_mult*abs(c1_raw-(c2_raw-c2_offset))/2
    y=cam_distance_m/(2*np.tan(angle))
    angle_x=-const_mult*((c1_raw+c2_raw-c2_offset)-400)/2
    x=np.tan(angle_x)*y
    print(angle_4,angle_2,c2_offset,const_mult,angle,y,x)
    return ([x,y])
    
def get_and_display_images(shared_data):
    print("Opening cams")#setup cameras
    cam2=cv2.VideoCapture(2)
    cam1=cv2.VideoCapture(1)
    print("Cams open")

    display_frame=np.array([])#initialize display frame
    #black frame the size of cam2/cam1 image
    ret2,frame2=cam2.read()
    map_frame=np.zeros((frame2.shape[0],frame2.shape[1],3),dtype=np.uint8)


    while True:
        ret1,frame1=cam1.read()#read frames
        ret2,frame2=cam2.read()

        display_frame=np.concatenate((frame1,frame2),axis=1)#concatenate frames
        shared_data["image"]=display_frame
        try:
            x1=shared_data["pos"][0][0]+((shared_data["pos"][0][2]-shared_data["pos"][0][0])*0.5)
            x2=shared_data["pos"][1][0]+((shared_data["pos"][1][2]-shared_data["pos"][1][0])*0.5)
            display_frame[:,int(x1),0]=0
            display_frame[:,int(x1),1]=255
            display_frame[:,int(x1),2]=0
            display_frame[:,int(x2),0]=0
            display_frame[:,int(x2),1]=255
            display_frame[:,int(x2),2]=0
            display_frame[:,int(x1+1),0]=0
            display_frame[:,int(x1+1),1]=255
            display_frame[:,int(x1+1),2]=0
            display_frame[:,int(x2+1),0]=0
            display_frame[:,int(x2+1),1]=255
            display_frame[:,int(x2+1),2]=0
            display_frame[:,int(x1-1),0]=0
            display_frame[:,int(x1-1),1]=255
            display_frame[:,int(x1-1),2]=0
            display_frame[:,int(x2-1),0]=0
            display_frame[:,int(x2-1),1]=255
            display_frame[:,int(x2-1),2]=0
            map_frame=np.zeros((frame2.shape[0],frame2.shape[1],3),dtype=np.uint8)
            map_x=int((shared_data["location"][0])*100)+320  #x
            map_y=int((shared_data["location"][1]+1)*48)+10  #y
            print(map_x,map_y)
            #make a dot on the map
            map_frame[map_y,map_x,0]=255
            map_frame[map_y,map_x,1]=255
            map_frame[map_y,map_x,2]=255
            #enlarge dot
            for i in range(5):
                for j in range(5):
                    map_frame[map_y+i,map_x+j,0]=255
                    map_frame[map_y+i,map_x+j,1]=255
                    map_frame[map_y+i,map_x+j,2]=255
            display_frame=np.concatenate((display_frame,map_frame),axis=1)
        except Exception as e:
            print(e)
        display_frame=cv2.resize(display_frame,(int(display_frame.shape[1]/1.5),int(display_frame.shape[0]/1.5)))

        if ret1 and ret2:#display frames
            cv2.imshow('ims',display_frame)
        if cv2.waitKey(1)==ord('q'):
            break
        
if __name__ == "__main__":
    manager = multiprocessing.Manager()
    shared_data=manager.dict()#data share

    shared_data["pos"]=[]
    shared_data["image"]=[]
    shared_data["location"]=[]

    cam_display = multiprocessing.Process(target=get_and_display_images, args=(shared_data,))
    cam_display.start()


    import torch, torchvision   #AI imports
    import mmdet
    import os, sys
    sys.path.append(r"D:\vscode\mmlab1\mmdetection")
    import mmcv
    import cv2
    from mmcv.ops import get_compiling_cuda_version, get_compiler_version
    from mmcv.runner import load_checkpoint
    import numpy as np
    from mmdet.apis import inference_detector, show_result_pyplot
    from mmdet.models import build_detector
    device="cuda"
    config_file=r"D:\vscode\mmlab1\mmdetection\configs\yolact\yolact_r50_1x8_coco.py"  #yolact
    checkpoint_file=r"D:\vscode\mmlab1\mmdetection\checkpoints\yolact_r50_1x8_coco_20200908-f38d58df.pth"
    config = mmcv.Config.fromfile(config_file)
    config.model.pretrained = None
    model = build_detector(config.model)
    checkpoint = load_checkpoint(model, checkpoint_file, map_location=device)
    model.CLASSES = checkpoint['meta']['CLASSES']
    model.cfg = config
    model.to(device)
    model.eval()

    
    
    time.sleep(1)
    img="im_file.jpg"
    t_e=time.time()
    t_p=time.time()
    while(True):
        try:
            cv2.imwrite(img,shared_data["image"])
            pre_inf=time.time()
            result = inference_detector(model,img)
            post_inf=time.time()
            shared_data["pos"]=[result[0][0][0],result[0][0][1]]
            if shared_data["pos"][0][2]>640:
                cam2_pos=int(shared_data["pos"][0][0]-640+(shared_data["pos"][0][2]-shared_data["pos"][0][0])/2)
                cam1_pos=int(shared_data["pos"][1][0]+(shared_data["pos"][1][2]-shared_data["pos"][1][0])/2)
            else:
                cam2_pos=int(shared_data["pos"][1][0]-640+(shared_data["pos"][1][2]-shared_data["pos"][1][0])/2)
                cam1_pos=int(shared_data["pos"][0][0]+(shared_data["pos"][0][2]-shared_data["pos"][0][0])/2)

            if (time.time()-t_p)>.1:
                print("fps: ",1/(post_inf-pre_inf),"  pos: ",cam1_pos,cam2_pos )
                shared_data["location"]=triangulate(cam1_pos,cam2_pos)
                print(shared_data["location"])
                t_p=time.time()
        except Exception as e:
            if time.time()-t_e>2:
                print(e)
                t_e=time.time()