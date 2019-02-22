from picamera import PiCamera 
import sys
import os
import avro.ipc as ipc
import avro.protocol as protocol
import numpy 

# Initialize the camera, set resolution to 220x220
camera = PiCamera()
RESOLUTION=[220,220,3]
camera.resolution=(RESOLUTION[0],RESOLUTION[1])
# read data packet format.
DIR_PATH = os.getcwd()
PROTOCAL_PATH='/../resource/protocol/msg.avpr'
PROTOCOL = protocol.parse(open(DIR_PATH + PROTOCAL_PATH).read())
# define default server addresss
SERVER_ADDR = ['localhost', 8000]

def send_frame():
    client = ipc.HTTPTransceiver(SERVER_ADDR[0], SERVER_ADDR[1])
    requestor = ipc.Requestor(PROTOCOL, client)

    output_array = numpy.empty(RESOLUTION, dtype=numpy.uint8)
    camera.capture(output_array, 'rgb')
    packet = dict()
    packet['input'] = output_array.tobytes()
    packet['input_shape'] = list(output_array.shape)
    packet['input_type'] = str(output_array.dtype)

    requestor.request('forward', packet)
    client.close()

def main():
    camera.start_preview()
    print ("Starting Camera...")
    print ("Press any key to take picture. Enter 'exit' to exit.")
    while (True):
        data=sys.stdin.readline()
        if (data=="exit\n"):
            break
        send_frame()
        
    camera.stop_preview()


if __name__ == '__main__':
    # parse arguments from command line
    if len(sys.argv) < 3:
        print ("Please entered the destionation IP address.")
        ip_address = sys.stdin.readline().rstrip()
        if (len(ip_address)> 0): 
            print ("Please enter the destination port number.")
            portNum=sys.stdin.readline().rstrip()
            SERVER_ADDR[0] = ip_address
            SERVER_ADDR[1] = int(portNum)
        else:
            print ("Using default localhost:8000")
    else:
        SERVER_ADDR[0] = sys.argv[1]
        SERVER_ADDR[1] = sys.argv[2]
    main()