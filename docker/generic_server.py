"""
    A generic server for handling traffic. This could be inherited in each sub module.
"""
import argparse
import os
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from SocketServer import ThreadingMixIn
import avro.ipc as ipc
import avro.protocol as protocol

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

PATH = os.path.abspath(__file__)
DIR_PATH = os.path.dirname(PATH)

# read data packet format.
PROTOCOL = protocol.parse(open(DIR_PATH + '/resource/protocol/msg.avpr').read())


class GenericResponder(ipc.Responder):
    """ Responder called by handler when got request. """

    def __init__(self):
        ipc.Responder.__init__(self, PROTOCOL)

    def invoke(self, msg, req):
        """
            This function will be automatically triggered by the Handler. req contains the
            actual data packet.

            Args:
                msg: Meta data.
                req: Contains data packet.
            Returns:
                None: It just acts as confirmation for sender.
            Raises:
                NotImplementedException
        """
        raise NotImplementedError('Invoke should be implemented by inherited class!')


def responder_factory():
    pass


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        """
            do_POST is automatically invoked by ThreadedHTTPServer. It creates a new
            responder for each request. The responder generates response and write
            response to data sent back.
        """
        self.responder = GenericResponder()
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


def main():
    server = ThreadedHTTPServer(('0.0.0.0', 8080), Handler)
    server.allow_reuse_address = True
    server.serve_forever()


if __name__ == '__main__':
    global ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument('service', help='specific service for the server')
    parser.add_argument('-d', '--debug', action='store_true', default=False,
                        help='set to debug mode')
    ARGS = parser.parse_args()
    main()
