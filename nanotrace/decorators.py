"""
Decorators are "syntactic" for a function wrapping another function.
The decorators defined here add some convenient functionality to some
of the functions defined elsewhere.
"""
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
from inspect import signature, Parameter
from typing import Callable, Any, Generator

import numpy as np

logger = logging.getLogger(__name__)


def catch_errors(n=1) -> Callable:
    """
    Catch errors and return nan instead of breaking.
    Handy for feature extractors.

    :param n: number of expected values so we know how many nans to return
    """

    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapper(*args, **kwargs) -> Any:
            try:
                with np.errstate(all='ignore'):
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
    """
    Decorate a function so that the first call saves the arguments in a closure.
    """
    @wraps(f)
    def closure(*p_args, **p_kwargs) -> Callable:
        @wraps(f)
        def wrapper(*args, **kwargs) -> Any:
            args = (*p_args, *args)
            kwargs.update(p_kwargs)
            return f(*args, **kwargs)

        return wrapper

    return closure



