from functools import wraps

from anytree import PreOrderIter


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
    def wrapper(*args, cutoff=0, **kwargs):
        for t, y, *l in f(*args, **kwargs):
            if len(t) > cutoff:
                yield t, y, *l
    return wrapper


def requires_children(f):
    """Used in root nodes to generate children before they are accessed"""
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        if not self.children:
            # Generate the tree
            for _ in PreOrderIter(self):
                pass
        return f(self, *args, **kwargs)
    return wrapper
