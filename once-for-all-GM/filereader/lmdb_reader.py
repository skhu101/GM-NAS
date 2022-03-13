__all__ = ['LMDBReader']

class LMDBReader:

    def __init__(self, lmdb_path, utf8_decode = False):
        self.lmdb_env = lmdb.open(
            lmdb_path,
            readonly = True,
            lock = False,
            readahead = False,
            meminit = False,
        )
        if not self.lmdb_env:
            raise Exception('cannot open lmdb from %s' % (lmdb_path))
        self.utf8_decode = False

    def __del__(self):
        self.lmdb_env.close()

    def __call__(self, path):
        key = path.encode('utf8')
        with lmdb_env.begin(write = False) as lmdb_txn:
            value = lmdb_txn.get(key.encode('utf8'))
        if self.utf8_decode:
            value = value.decode('utf8')
        return value
