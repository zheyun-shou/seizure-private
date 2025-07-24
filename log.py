import sys
import logging

class OutputCapture:
    def __init__(self, filename, also_console=True):
        self.filename = filename
        self.also_console = also_console
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Set up file logging
        logging.basicConfig(
            filename=filename,
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            filemode='w'
        )
        self.logger = logging.getLogger('output_capture')
    
    def write_stdout(self, text):
        if text.strip():  # Only log non-empty lines
            self.logger.info(f"STDOUT: {text.strip()}")
        if self.also_console:
            self.original_stdout.write(text)
    
    def write_stderr(self, text):
        if text.strip():
            self.logger.error(f"STDERR: {text.strip()}")
        if self.also_console:
            self.original_stderr.write(text)
    
    def flush(self):
        if self.also_console:
            self.original_stdout.flush()
            self.original_stderr.flush()

class StdoutCapture:
    def __init__(self, capture_obj):
        self.capture = capture_obj
    def write(self, text):
        self.capture.write_stdout(text)
    def flush(self):
        self.capture.flush()

class StderrCapture:
    def __init__(self, capture_obj):
        self.capture = capture_obj
    def write(self, text):
        self.capture.write_stderr(text)
    def flush(self):
        self.capture.flush()