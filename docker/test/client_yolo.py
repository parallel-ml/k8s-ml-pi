from service.yolo.util import preprocess_input, net_h, net_w, decode_box, draw_boxes, labels, obj_thresh, \
    correct_yolo_boxes
import os
import sys
import avro.ipc as ipc
import avro.protocol as protocol
import cv2
import time

PATH = os.path.abspath(__file__)
DIR_PATH = os.path.dirname(PATH)

# read data packet format.
PROTOCOL = protocol.parse(open(DIR_PATH + '/../resource/protocol/msg.avpr').read())
SERVER_ADDR = ['localhost', 8080]


def send_request():
    client = ipc.HTTPTransceiver(SERVER_ADDR[0], SERVER_ADDR[1])
    requestor = ipc.Requestor(PROTOCOL, client)

    image = cv2.imread(image_path)
    image_h, image_w, _ = image.shape
    data = preprocess_input(image, net_h, net_w)

    packet = dict()
    packet['input'] = data.tobytes()
    packet['input_shape'] = list(data.shape)
    packet['input_type'] = str(data.dtype)

    start = time.time()
    array = requestor.request('forward', packet)
    print 'Latency: %.3f sec' % (time.time() - start)
    boxes = decode_box(array)
    correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)
    draw_boxes(image, boxes, labels, obj_thresh)
    client.close()

    cv2.imshow('Detected image', image)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('\r') or key == ord('\n'):
            break
        time.sleep(0.01)
    cv2.destroyAllWindows()


def main():
    for _ in range(10):
        print 'Send request ... ... ',
        send_request()
        print 'Complete'


if __name__ == '__main__':
    # parse arguments from command line
    print 'Using default localhost:8080'
    global image_path
    try:
        image_path = sys.argv[1]
    except Exception as e:
        print 'python yolo-client.py [image path]'
        exit(1)
    main()
