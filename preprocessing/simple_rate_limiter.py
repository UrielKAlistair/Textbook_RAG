import time


class SimpleRateLimiter:
    def __init__(self, requests_per_minute):
        self.requests_per_minute = requests_per_minute
        self.interval = 60 / requests_per_minute
        self.last_request_time = 0

    def wait_if_needed(self):
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self.last_request_time = time.time()
