class Singleton(type):
    """ Singleton implementation to avoid creating same model multiple times. """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class GenericService(object):
    __metaclass__ = Singleton

    def __init__(self):
        # do nothing
        pass

    def predict(self, input):
        """
            Predict method takes input from message and return the inference result.

            Args:
                input: Input array from message protocol in Numpy format.
            Returns:
                np.array(): Numpy array for inference result.
        """
        raise NotImplementedError('Predict method must be implemented!')

    def send(self, output):
        """
            Send the output to next service. This method is better to be implemented as non-blocking.

            Args:
                output: Output array in numpy format, and this function should convert this to bytes.
            Returns:
                None
        """
        raise NotImplementedError('Send method must be implemented!')
