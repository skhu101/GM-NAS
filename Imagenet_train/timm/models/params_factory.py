import logging
import collections

logger = logging.getLogger(__name__)
def unify_args(aargs):
    """ Return a dict of args """
    if aargs is None:
        return {}
    if isinstance(aargs, str):
        return {"name": aargs}
    assert isinstance(aargs, dict)
    return aargs


def merge_unify_args(*args):
    from collections import ChainMap

    unified_args = [unify_args(x) for x in args]
    ret = dict(ChainMap(*unified_args))
    return ret

def filter_kwargs(func, kwargs, log_skipped=True):
    """ Filter kwargs based on signature of `func`
        Return arguments that matches `func`
    """
    import inspect

    sig = inspect.signature(func)
    filter_keys = [
        param.name
        for param in sig.parameters.values()
        if param.kind == param.POSITIONAL_OR_KEYWORD
    ]

    if log_skipped:
        skipped_args = [x for x in kwargs.keys() if x not in filter_keys]
        if skipped_args:
            logger.warning("Arguments {skipped_args} skipped for op {func.__name__}")

    filtered_dict = {
        filter_key: kwargs[filter_key]
        for filter_key in filter_keys
        if filter_key in kwargs
    }
    return filtered_dict

def update_dict(dest, src):
    """ Update the dict 'dest' recursively.
        Elements in src could be a callable function with signature
            f(key, curr_dest_val)
    """
    for key, val in src.items():
        if isinstance(val, collections.Mapping):
            # dest[key] could be None in the case of a dict
            cur_dest = dest.get(key, {}) or {}
            assert isinstance(cur_dest, dict), cur_dest
            dest[key] = update_dict(cur_dest, val)
        else:
            if callable(val) and key in dest:
                dest[key] = val(key, dest[key])
            else:
                dest[key] = val
    return dest

def merge(kwargs, **all_args):
    """ kwargs will override other arguments """
    return update_dict(all_args, kwargs)