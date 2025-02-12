from functools import wraps


def partial(f):
    """Decorate a function so that the first call saves the arguments in a closure"""
    @wraps(f)
    def closure(*p_args, **p_kwargs):
        @wraps(f)
        def wrapper(*args, **kwargs):
            args = (*p_args, *args)
            kwargs.update(p_kwargs)
            return f(*args, **kwargs)
        return wrapper
    return closure


def cutoff(f):
    """Filter out any segments shorter than the cutoff n samples"""
    @wraps(f)
    def closure(*args, cutoff=0, **kwargs):
        @partial
        @wraps(f)
        def with_cutoff(*args, **kwargs):
            for t,y,*l in f(*args, **kwargs):
                if len(t) > cutoff:
                    yield t,y,*l
        if cutoff > 0:
            return with_cutoff(*args, **kwargs)
        else:
            return f(*args, **kwargs)
    return closure
