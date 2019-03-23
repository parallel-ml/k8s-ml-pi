from service.generic_service import GenericService
from service.yolo.util import load_yolo_model


class Service(GenericService):
    def __init__(self):
        GenericService.__init__(self)
        self.model = load_yolo_model([94, 153])

    def predict(self, input):
        return self.model.predict(input)

    def __repr__(self):
        return 'yolo.demo.lb2-service'
