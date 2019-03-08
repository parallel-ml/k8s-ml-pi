from service.generic_server import GenericResponder
import numpy as np


class Responder(GenericResponder):
    def __init__(self, service_cls):
        GenericResponder.__init__(self)
        self.service = service_cls()

    def invoke(self, msg, req):
        try:
            input_shape = req['input_shape']
            data = np.fromstring(req['input'], req['input_type']).reshape(input_shape)
            output = self.service.predict(data)
            self.service.send(output)

        except Exception as e:
            print e
