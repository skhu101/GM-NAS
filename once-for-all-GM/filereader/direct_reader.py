__all__ = ['DirectReader']

class DirectReader:

    def __call__(self, path):
        with open(path, 'rb') as f:
            content = f.read()
        return content
