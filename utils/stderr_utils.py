import os
import sys
import threading

def suppress_stderr_keywords(keywords):
    r, w = os.pipe()
    original_stderr_fd = sys.stderr.fileno()
    saved_stderr_fd = os.dup(original_stderr_fd)
    os.dup2(w, original_stderr_fd)

    def filter_thread():
        with os.fdopen(r) as read_pipe:
            for line in read_pipe:
                if not any(k in line for k in keywords):
                    os.write(saved_stderr_fd, line.encode())

    t = threading.Thread(target=filter_thread, daemon=True)
    t.start()
