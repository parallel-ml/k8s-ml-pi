import datetime
from picamera import PiCamera 
import sys
import os

camera = PiCamera()
camera.resolution=(220,220)
camera.start_preview()
# create directory to store images, if not exist
directory  = str(os.getcwd())+'/images'
try:
    os.stat(directory)
except:
    os.mkdir(directory) 

print ("press any key to take picture. Enter 'exit' to exit.")
while (True):
    data=sys.stdin.readline()
    if (data=="exit\n"):
        break
    camera.capture(directory+'/img-'+str(datetime.datetime.now())+'.jpg')

camera.stop_preview()
