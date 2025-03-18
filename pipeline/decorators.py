__copyright__ = """
Copyright 2025 Matthijs Tadema

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
from functools import wraps
from typing import Callable, Any, Generator

import numpy as np

logger = logging.getLogger(__name__)


def catch_errors(n=1) -> Callable:
    """Catch errors and return nan instead of breaking"""

    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return f(*args, **kwargs)
            except Exception as e:
                logger.error(e)
                if n > 1:
                    return [np.nan] * n
                else:
                    return np.nan

        return wrapper

    return decorator


def partial(f: Callable) -> Callable:
    """Decorate a function so that the first call saves the arguments in a closure"""

    @wraps(f)
    def closure(*p_args, **p_kwargs) -> Callable:
        @wraps(f)
        def wrapper(*args, **kwargs) -> Any:
            args = (*p_args, *args)
            kwargs.update(p_kwargs)
            return f(*args, **kwargs)

        return wrapper

    return closure


def cutoff(f: Callable) -> Callable:
    """Filter out any segments shorter than the cutoff n samples"""

    @wraps(f)
    def wrapper(*args, cutoff=0, **kwargs) -> Generator[tuple[np.array, np.array, list]]:
        for t, y, *l in f(*args, **kwargs):
            if len(t) > cutoff:
                yield t, y, *l

    return wrapper


def requires_children(f: Callable) -> Callable:
    """Used in root nodes to generate children before they are accessed"""

    @wraps(f)
    def wrapper(self, *args, **kwargs) -> Any:
        if not self.children:
            # Generate the tree
            self.derive_children()
        return f(self, *args, **kwargs)

    return wrapper
