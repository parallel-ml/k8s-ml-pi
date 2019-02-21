from picamera import PiCamera 
import sys
import os
import io
import socket
import time
import struct


sender_socket = socket.socket()
sender_socket.connect(('127.0.0.1', 8000))
connection = sender_socket.makefile('wb')
camera = PiCamera()
camera.resolution=(220,220)
camera.start_preview()

print ("press any key to take picture. Enter 'exit' to exit.")
try:
    while (True):
        data=sys.stdin.readline()
        start = time.time()
        stream = io.BytesIO()
        if (data=="exit\n"):
            break
        camera.capture(stream, format='jpeg')
        connection.write(struct.pack('<L', stream.tell()))
        connection.flush()
        stream.seek(0)
        connection.write(stream.read())
        stream.seek(0)
        stream.truncate()
finally:
    camera.stop_preview()
    connection.write(struct.pack('<L', 0))
    sender_socket.close()

