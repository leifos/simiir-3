import time

def retry(max_retries, wait_time):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            if retries < max_retries:
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    retries += 1
                    time.sleep(wait_time)
            else:
              raise Exception(f"Max retries of function {func} exceeded")
        return wrapper
    return decorator
