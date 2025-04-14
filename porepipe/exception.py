"""
Exceptions used in porepipe. All inherit from the same base
so it's easier to catch them.
"""

class PorePipeException(Exception):
    pass


class StageError(Exception): pass
