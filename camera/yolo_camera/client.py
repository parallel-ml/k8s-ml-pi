import os
import sys
import avro.ipc as ipc
import avro.protocol as protocol
import numpy as np

PATH = os.path.abspath(__file__)
DIR_PATH = os.path.dirname(PATH)

# read data packet format.
PROTOCOL = protocol.parse(open(DIR_PATH + '/../../docker/resource/protocol/msg.avpr').read())

SERVER_ADDR = ['localhost', 8080]


def send_request():
    client = ipc.HTTPTransceiver(SERVER_ADDR[0], SERVER_ADDR[1])
    requestor = ipc.Requestor(PROTOCOL, client)

    data1 = np.random.random_sample([1, 20, 20, 512])
    data1 = data1.astype(np.float32)

    data2 = np.random.random_sample([1, 20, 20, 512])
    data2 = data2.astype(np.float32)

    data3 = np.random.random_sample([1, 40, 40, 256])
    data3 = data3.astype(np.float32)

    packet = dict()
    packet['input'] = [data1.tobytes(), data2.tobytes(), data3.tobytes()]

    requestor.request('forward', packet)
    client.close()


def main():
    for _ in range(1):
        print 'Send request ... ... ',
        send_request()
        print 'Complete'


if __name__ == '__main__':
    # parse arguments from command line
    if len(sys.argv) < 3:
        print 'Using default localhost:8000'
    else:
        SERVER_ADDR[0] = sys.argv[1]
        SERVER_ADDR[1] = sys.argv[2]
    main()
