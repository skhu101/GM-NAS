import yaml
import json5

###############################################################
# utility functions
###############################################################

def consume_dots(config, key, create_default):
    sub_keys = key.split('.', 1)
    sub_key = sub_keys[0]

    if not dict.__contains__(config, sub_key) and len(sub_keys) == 2:
        if create_default:
            dict.__setitem__(config, sub_key, Config())
        else:
            raise KeyError('%s not exists' % str(key))

    if len(sub_keys) == 1:
        return config, sub_key
    else:
        sub_config = dict.__getitem__(config, sub_key)
        if type(sub_config) != Config:
            if create_default:
                sub_config = Config()
                dict.__setitem__(config, sub_key, sub_config)
            else:
                raise KeyError('%s not exists' % str(key))
        return consume_dots(sub_config, sub_keys[1], create_default)

def traverse_dfs(root, mode, continue_type, key_prefix = ''):
    for key, value in root.items():
        full_key = '.'.join([key_prefix, key]).strip('.')
        yield { 'key': full_key, 'value': value, 'item': (full_key, value) }[mode]
        if type(value) == continue_type:
            for kv in traverse_dfs(value, mode, continue_type, full_key):
                yield kv

def traverse_bfs(root, mode):
    q = [(root, '')]
    while len(q) > 0:
        child, key_prefix = q.pop(0)
        for key, value in child.items():
            full_key = '.'.join([key_prefix, key]).strip('.')
            yield { 'key': full_key, 'value': value, 'item': (full_key, value) }[mode]
            if type(value) == continue_type:
                q.append((value, full_key))

def init_assign(config, d, traverse):
    for full_key, value in traverse_dfs(d, 'item', continue_type = dict):
        # skip non-empty dict
        if type(value) == dict and len(value) > 0: continue
        sub_cfg, sub_key = consume_dots(config, full_key, create_default = True)
        sub_cfg[sub_key] = value

###############################################################
# main class
###############################################################

class Config(dict):

    def __init__(self, *args, **kwargs):
        super(Config, self).__init__()
        for arg in args:
            if isinstance(arg, str):
                if arg.endswith('.json') or arg.endswith('.json5'):
                    with open(arg) as f:
                        raw_dict = json5.load(f)
                elif arg.endswith('.yaml'):
                    with open(arg) as f:
                        raw_dict = yaml.load(f)
                else:
                    raise Exception('unknown file format %s' % arg)
                init_assign(self, raw_dict, traverse = True)
            elif isinstance(arg, dict):
                init_assign(self, arg, traverse = True)
            else:
                raise TypeError('arg should be an instance of <str> or <dict>')
        if kwargs:
            init_assign(self, kwargs, traverse = False)

    def __call__(self, *args, **kwargs):
        return Config(self, *args, **kwargs)

    ###########################################################
    # support for pickle
    ###########################################################

    def __setstate__(self, state):
        init_assign(self, state, traverse = True)

    def __getstate__(self):
        d = dict()
        for key, value in self.items():
            if type(value) is Config:
                value = value.__getstate__()
            d[key] = value
        return d

    ###########################################################
    # access by '.' -> access by '[]'
    ###########################################################

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]

    ###########################################################
    # access by '[]'
    ###########################################################

    def __getitem__(self, key):
        sub_cfg, sub_key = consume_dots(self, key, create_default = False)
        return dict.__getitem__(sub_cfg, sub_key)

    def __setitem__(self, key, value):
        sub_cfg, sub_key = consume_dots(self, key, create_default = True)
        dict.__setitem__(sub_cfg, sub_key, value)

    def __delitem__(self, key):
        sub_cfg, sub_key = consume_dots(self, key, create_default = False)
        dict.__delitem__(sub_cfg, sub_key)
        #del self.__dict__[key]

    ###########################################################
    # access by 'in'
    ###########################################################

    def __contains__(self, key):
        try:
            sub_cfg, sub_key = consume_dots(self, key, create_default = False)
        except KeyError:
            return False
        return dict.__contains__(sub_cfg, sub_key)

    ###########################################################
    # traverse keys / values/ items
    ###########################################################

    def all_keys(self, order = 'dfs'):
        traverse = { 'dfs': traverse_dfs, 'bfs': traverse_bfs }[order]
        for key in traverse(self, 'key', continue_type = Config):
            yield key

    def all_values(self, order = 'dfs'):
        traverse = { 'dfs': traverse_dfs, 'bfs': traverse_bfs }[order]
        for value in traverse(self, 'value', continue_type = Config):
            yield value

    def all_items(self, order = 'dfs'):
        traverse = { 'dfs': traverse_dfs, 'bfs': traverse_bfs }[order]
        for key, value in traverse(self, 'item', continue_type = Config):
            yield key, value

    ###########################################################
    # for command line arguments
    ###########################################################

    def parse_args(self, cmd_args = None, strict = True):
        unknown_args = []
        if cmd_args is None:
            import sys
            cmd_args = sys.argv[1:]
        index = 0
        while index < len(cmd_args):
            arg = cmd_args[index]
            err_msg = 'invalid command line argument pattern: %s' % arg
            assert arg.startswith('--'), err_msg
            assert len(arg) > 2, err_msg
            assert arg[2] != '-', err_msg

            arg = arg[2:]
            if '=' in arg:
                key, full_value_str = arg.split('=')
                index += 1
            else:
                assert len(cmd_args) > index + 1, \
                        'incomplete command line arguments'
                key = arg
                full_value_str = cmd_args[index + 1]
                index += 2
            if ':' in full_value_str:
                value_str, value_type_str = full_value_str.split(':')
                value_type = eval(value_type_str)
            else:
                value_str = full_value_str
                value_type = None

            if key not in self:
                if strict:
                    raise KeyError('%s not exists in config' % key)
                else:
                    unknown_args.extend(['--' + key, full_value_str])
                    continue

            if value_type is None:
                value_type = type(self[key])

            if value_type is bool:
                self[key] = {
                    'true' : True,
                    'True' : True,
                    '1'    : True,
                    'false': False,
                    'False': False,
                    '0'    : False,
                }[value_str]
            else:
                self[key] = value_type(value_str)

        return unknown_args

    ###########################################################
    # for key reference
    ###########################################################

    def parse_refs(self, subconf = None, stack_depth = 1, max_stack_depth = 10):
        if stack_depth > max_stack_depth:
            raise Exception((
                'Recursively calling `parse_refs` too many times with stack depth > %d. A circular reference may exists in your config.\n'
                'If deeper calling stack is really needed, please call `parse_refs` with extra argument like: `parse_refs(max_stack_depth = 9999)`'
                ) % max_stack_depth
            )
        if subconf is None:
            subconf = self
        for key in subconf.keys():
            value = subconf[key]
            if type(value) is str and value.startswith('@{') and value.endswith('}'):
                ref_key = value[2:-1]
                ref_value = self[ref_key]
                if type(ref_value) is str and ref_value.startswith('@{') and value.endswith('}'):
                    raise Exception('Refering key %s to %s, but the value of %s is another reference value %s' % (
                        repr(key), repr(value), repr(ref_key), repr(ref_value),
                    ))
                subconf[key] = ref_value
        for key in subconf.keys():
            value = subconf[key]
            if type(value) is Config:
                self.parse_refs(value, stack_depth + 1)
