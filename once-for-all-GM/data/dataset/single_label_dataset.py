from glob import glob

from filereader import DirectReader
from util import alpha_print

__all__ = ['SingleLabelDataset']

import io
import os
import sys
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


def to_imglist():
    '''
    by huangguowei: generate train or val meta file.
    :param root:
    :return:
    '''
    import pydash
    root = "/home/huang/huang/data/ILSVRC/Data/CLS-LOC/train"
    meta_train_path = "/home/huang/huang/data/ILSVRC/meta/train.txt"

    mpa_clsloc = "/home/huang/huang/data/ILSVRC/devkit/data/map_clsloc.txt"

    root = root if root.endswith("/") else root + "/"
    lines = open(mpa_clsloc, "r").readlines()
    # lines ['n02119789 1 kit_fox\n', 'n02100735 2 English_setter\n', 'n02110185 3 Siberian_husky\n']
    print("lines", lines[:3])

    def get_class_name_id(line):
        line_split = line.strip().split(maxsplit=2)
        # class_id starts from 0.
        class_id = int(line_split[1]) - 1
        return line_split[0], class_id

    class_name_id_dict = dict(pydash.map_(lines, get_class_name_id))

    paths = sorted(glob(os.path.join(root, '*/*.JPEG')))
    print("paths", paths[:3])


    def parse_jpeg_path(full_path):
        relative_path = pydash.replace(full_path, root, "")
        last_slash_index = pydash.last_index_of(relative_path, "/")
        class_name = relative_path[:last_slash_index]
        # print("class_name", class_name)
        class_id = class_name_id_dict[class_name]
        return relative_path, class_id

    paths_with_id = pydash.map_(paths, parse_jpeg_path)

    with open(meta_train_path, "w") as file:
        pydash.map_(paths_with_id, lambda kv: file.write(pydash.join(kv, " ") + '\n'))

    return 0


class SingleLabelDataset(Dataset):
    def __init__(self, imglist, root='', image_mode='RGB', transform=transforms.Compose([]), reader=DirectReader(),
                 psudo_index=-1, skip_broken=True):
        super(SingleLabelDataset, self).__init__()
        self.root = root
        self.image_mode = image_mode
        self.transform = transform
        self.reader = reader
        self.psudo_index = psudo_index
        self.skip_broken = skip_broken

        self.imglist = []
        with open(imglist) as f:
            for line in f.readlines():
                path, label = line.strip().split(maxsplit=1)
                label = int(label)
                self.imglist.append((path, label))
        alpha_print('single label dataset samples: %d' % len(self), flush=True)

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        if index == self.psudo_index:
            index = random.randrange(len(self))
            psudo = 1
        else:
            psudo = 0

        while True:
            try:
                path, label = self.imglist[index]
                path = os.path.join(self.root, path)
                filebytes = self.reader(path)
                buff = io.BytesIO(filebytes)
                image = Image.open(buff)
                image = image.convert(self.image_mode)
                break
            except Exception as e:
                if self.skip_broken:
                    print('Warning: skip [%s]，using next index' % path, flush=True)
                    index = (index + 1) % len(self)
                else:
                    print('file [%s] is broken' % path, flush=True)
                    raise e

        image = self.transform(image)

        # return {
        #     'data': image,
        #     'label': label,
        #     'index': index,
        #     'psudo': psudo,
        # }
        return image, label
