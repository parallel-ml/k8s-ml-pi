from service.generic_server import GenericResponder


class Responder(GenericResponder):
    def __init__(self, service_cls):
        GenericResponder.__init__(self)
        self.service = service_cls()

    def invoke(self, msg, req):
        try:
            output = self.service.predict(req['input'])
            print 'finish prediction'
            next_result = self.service.send(output)
            print 'get response from next layer'
            return next_result

        except Exception as e:
            print e
