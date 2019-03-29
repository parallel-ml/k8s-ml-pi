from service.generic_server import GenericResponder
import time


class Responder(GenericResponder):
    def __init__(self, service_cls):
        GenericResponder.__init__(self)
        self.service = service_cls()

    def invoke(self, msg, req):
        try:
            start = time.time()
            output = self.service.predict(req['input'])
            print '%s latency: %.3f sec' % (self.service, (time.time() - start))
            return output

        except Exception as e:
            print e
