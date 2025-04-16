import sys
from contextlib import ContextDecorator


class Tee(ContextDecorator):
    """
    Context manager that duplicates writes to original_stdout and a
    file handle simultaneously. Hopefully this ignores stderr!
    """
    def __init__(self, file_path, mode='w'):
        self.file_path = file_path
        self.mode = mode
        self._file = None
        self._original_stdout = None

    def __enter__(self):
        # Open the output file
        self._file = open(self.file_path, self.mode)
        # Save original stdout
        self._original_stdout = sys.stdout
        # Replace sys.stdout with specified object
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore the original stdout
        sys.stdout = self._original_stdout
        # Close the file
        self._file.close()

    def write(self, data):
        # Write data to both original_stdout and file
        self._original_stdout.write(data)
        self._file.write(data)

    def flush(self):
        # Ensure both streams are flushed
        self._original_stdout.flush()
        self._file.flush()
