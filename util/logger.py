import sys

class Logger:
    def __init__(self, filename):
        self.filename = filename
        self.log_file = open(filename, 'w')

    def write(self, message):
        self.log_file.write(message)
        sys.__stdout__.write(message)

    def writelines(self, messages):
        self.log_file.writelines(messages)
        sys.__stdout__.writelines(messages)

    def flush(self):
        self.log_file.flush()
        sys.__stdout__.flush()

    def __enter__(self):
        self.original_stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        self.log_file.close()

    def __del__(self):
        self.log_file.close()