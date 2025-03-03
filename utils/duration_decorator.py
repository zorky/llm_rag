import time
from functools import wraps

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        print(f"\n## Temps écoulé pour {func.__name__}: {duration:.4f} secondes ##\n")
        return result
    return wrapper