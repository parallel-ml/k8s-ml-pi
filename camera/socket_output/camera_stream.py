from picamera import PiCamera 
import sys
import os
import io
import socket
import struct

# Sending to user specified port number and address
print ("Please entered the destionation IP address.")
ip_addr = sys.stdin.readline().rstrip()
print ("Please enter the destination port number.")
port=sys.stdin.readline().rstrip()
port = int(port)
# initialize socket
sender_socket = socket.socket()
sender_socket.connect((ip_addr, port))
connection = sender_socket.makefile('wb')
# Initialize the camera, set resolution to 220x220
camera = PiCamera()
camera.resolution=(220,220)
camera.start_preview()

print ("press any key to take picture. Enter 'exit' to exit.")
try:
    while (True):
        data=sys.stdin.readline()
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

