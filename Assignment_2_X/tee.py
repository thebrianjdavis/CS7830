import sys
from contextlib import ContextDecorator


class Tee(ContextDecorator):
    """
    Context manager that duplicates writes to original_stdout and a file handle
    simultaneously. The same as used in your linear regression project.
    """
    def __init__(self, file_path, mode='w'):
        self.file_path = file_path
        self.mode = mode
        self._file = None
        self._original_stdout = None

    def __enter__(self):
        self._file = open(self.file_path, self.mode)
        self._original_stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        sys.stdout = self._original_stdout
        self._file.close()

    def write(self, data):
        self._original_stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._original_stdout.flush()
        self._file.flush()
