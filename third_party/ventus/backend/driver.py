from triton.backends.driver import DriverBase


class VentusDriver(DriverBase):

    _ERROR = "Ventus runtime is not implemented"

    @staticmethod
    def is_active():
        return False

    @classmethod
    def _unsupported(cls):
        raise RuntimeError(cls._ERROR)

    def map_python_to_cpp_type(self, ty: str) -> str:
        self._unsupported()

    def get_current_target(self):
        self._unsupported()

    def get_active_torch_device(self):
        self._unsupported()

    def get_benchmarker(self):
        self._unsupported()
