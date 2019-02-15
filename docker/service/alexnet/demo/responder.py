from service.generic_server import GenericResponder


class Responder(GenericResponder):
    def __init__(self, service_cls):
        GenericResponder.__init__(self)
        self.service = service_cls()

    def invoke(self, msg, req):
        try:
            print self.service
        except Exception as e:
            print e
