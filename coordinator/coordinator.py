from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from SocketServer import ThreadingMixIn
import avro.ipc as ipc
import avro.protocol as protocol
from multiprocessing import Queue
import time
from threading import Thread

PROTOCOL = protocol.parse(open('resource/protocol/msg.avpr').read())

LB1_SERVICE = Queue()
LB2_SERVICE = Queue()
BB1_SERVICE = Queue()
BB2_SERVICE = Queue()

INPUT = Queue()
LB1_RESULT = Queue()
LB2_RESULT = Queue()
BB1_RESULT = Queue()
BB2_RESULT = Queue()


class Responder(ipc.Responder):
    """ Responder called by handler when got request. """

    def __init__(self):
        ipc.Responder.__init__(self, PROTOCOL)

    def invoke(self, msg, req):
        try:
            INPUT.put(req['input'])
            return BB2_RESULT.get()
        except Exception, e:
            print 'Exception: %s' % e


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.responder = Responder()
        call_request_reader = ipc.FramedReader(self.rfile)
        call_request = call_request_reader.read_framed_message()
        resp_body = self.responder.respond(call_request)
        self.send_response(200)
        self.send_header('Content-Type', 'avro/binary')
        self.end_headers()
        resp_writer = ipc.FramedWriter(self.wfile)
        resp_writer.write_framed_message(resp_body)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """ Handle requests in separate thread. """


def send(output, ip, id):
    client = ipc.HTTPTransceiver(ip, 8080)
    requestor = ipc.Requestor(PROTOCOL, client)

    packet = dict()
    packet['input'] = output

    result = requestor.request('forward', packet)
    client.close()

    if id == 'lb1':
        LB1_RESULT.put(result)
        LB1_SERVICE.put(ip)
    elif id == 'lb2':
        LB2_RESULT.put(result)
        LB2_SERVICE.put(ip)
    elif id == 'bb1':
        BB1_RESULT.put(result)
        BB1_SERVICE.put(ip)
    else:
        BB2_RESULT.put(result)
        BB2_SERVICE.put(ip)


def lb1_worker():
    while True:
        Thread(target=send, args=(INPUT.get(), LB1_SERVICE.get(), 'lb1')).start()
        time.sleep(0.001)


def lb2_worker():
    while True:
        Thread(target=send, args=(LB1_RESULT.get(), LB2_SERVICE.get(), 'lb2')).start()
        time.sleep(0.001)


def bb1_worker():
    while True:
        Thread(target=send, args=(LB2_RESULT.get(), BB1_SERVICE.get(), 'bb1')).start()
        time.sleep(0.001)


def bb2_worker():
    while True:
        Thread(target=send, args=(BB1_RESULT.get(), BB2_SERVICE.get(), 'bb2')).start()
        time.sleep(0.001)


def main():
    LB1_SERVICE.put('192.168.1.101')
    LB2_SERVICE.put('192.168.1.102')
    BB1_SERVICE.put('192.168.1.103')
    BB2_SERVICE.put('192.168.1.104')

    Thread(target=lb1_worker).start()
    Thread(target=lb2_worker).start()
    Thread(target=bb1_worker).start()
    Thread(target=bb2_worker).start()

    server = ThreadedHTTPServer(('0.0.0.0', 8080), Handler)
    server.allow_reuse_address = True
    server.serve_forever()


if __name__ == '__main__':
    main()
