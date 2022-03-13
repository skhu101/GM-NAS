from collections import namedtuple

Dataset = namedtuple('Dataset',
                     ['num_classes',
                      'num_channels',
                      'hw',
                      'mean',
                      'std',
                      'initial_channels_factor',
                      'is_ImageFolder',
                      'def_resize'])

datasets = {'CIFAR10':
              Dataset(10, 3, [32, 32],
                      [0.49139968, 0.48215827, 0.44653124],
                      [0.24703233, 0.24348505, 0.26158768],
                      1,
                      False,
                      None),

            'ImageNet':
              Dataset(1000, 3, [224, 224],
                      [0.485, 0.456, 0.406],
                      [0.229, 0.224, 0.225],
                      1,
                      True,
                      None),
            }