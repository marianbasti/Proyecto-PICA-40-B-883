import picamera
import datetime as dt
import os

with picamera.PiCamera() as camera:
    camera.resolution = (1600, 1080)
    camera.framerate = 10
    #camera.start_preview()
    camera.annotate_background = picamera.Color('black')
    camera.annotate_text = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    while True:
        now = str(dt.datetime.now())
        timestamp=now.split()
        milis=timestamp[1].split(".")
        date=timestamp[0].split("-")
        time=milis[0].split(":")
        camera.shutter_speed=800
        if int(time[0])>=7 and int(time[0])<=19:
            print (now)
            try:
                os.mkdir("/home/pi/output/"+timestamp[0])
            except OSError as error:
                pass
            try:
                os.mkdir("/home/pi/output/"+timestamp[0]+"/video")
            except OSError as error:
                pass
            #camera.start_recording("/home/pi/output/"+timestamp[0]+"/video/"+timestamp[0]+" "+milis[0]+'.h264')
            camera.start_recording("/home/pi/output/"+timestamp[0]+"/video/"+date[0]+"_"+date[1]+"_"+date[2]+"_"+"_"+time[0]+"_"+time[1]+"_"+time[2]+'.h264')
            start = dt.datetime.now()
            while (dt.datetime.now() - start).seconds < 900:
                camera.annotate_text = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                camera.wait_recording(0.2)
            camera.stop_recording()
            print ("video finished")
        #if int(time[0])==1 and (time[1])==10:
            #exit()
