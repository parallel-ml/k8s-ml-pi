from docker.generic_server import GenericResponder


class Responder(GenericResponder):
    def invoke(self, msg, req):
        print msg
