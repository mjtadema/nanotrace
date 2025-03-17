# Copyright 2025 Matthijs Tadema
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
            self.derive_children()
        return f(self, *args, **kwargs)
    return wrapper
