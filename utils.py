def exception(exc_type, msg):
    def decorator(func):
        def wrap_func(*args, **kwargs):
            try:
                func(*args, **kwargs)
            except exc_type:
                print(msg)
        
        return wrap_func
    
    return decorator