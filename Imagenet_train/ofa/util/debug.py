__all__ = ['alpha_print']

import os

def alpha_print(*args, **kwargs):
    if os.environ.get('ALPHATRION_VERBOSE') in ['1', 'true', 'True', 'TRUE']:
        print(*args, **kwargs)
