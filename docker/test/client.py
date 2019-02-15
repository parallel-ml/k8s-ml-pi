import os
import avro.ipc as ipc
import avro.protocol as protocol
import numpy as np

PATH = os.path.abspath(__file__)
DIR_PATH = os.path.dirname(PATH)

# read data packet format.
PROTOCOL = protocol.parse(open(DIR_PATH + '/../resource/protocol/msg.avpr').read())

SERVER_ADDR = ('localhost', 8000)


def send_request():
    client = ipc.HTTPTransceiver(SERVER_ADDR[0], SERVER_ADDR[1])
    requestor = ipc.Requestor(PROTOCOL, client)

    data = np.random.random_sample([220, 220, 3]) * 255
    data = data.astype(np.uint8)

    packet = dict()
    packet['input'] = data.tobytes()
    packet['input_shape'] = list(data.shape)
    packet['input_type'] = str(data.dtype)

    requestor.request('forward', packet)
    client.close()


def main():
    for _ in range(10):
        print 'Send request ... ... ',
        send_request()
        print 'Complete'


if __name__ == '__main__':
    main()
