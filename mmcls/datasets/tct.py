# -*- coding: utf-8 -*-
# @Author  : zhaoyang

import numpy as np
import mmcv

from .builder import DATASETS
from .base_dataset import BaseDataset


@DATASETS.register_module()
class TCT(BaseDataset):
    # CLASSES = ["normal", "ascus", "asch", "lsil", "hsil_scc_omn", "agc_adenocarcinoma_em",
    #            "vaginalis", "monilia", "dysbacteriosis_herpes_act", "ec"]

    def load_annotations(self):
        assert isinstance(self.ann_file, str)

        data_infos = []
        with open(self.ann_file) as f:
            samples = [x.strip().split(' ') for x in f.readlines()]
            for filename, gt_label in samples:
                info = {'img_prefix': self.data_prefix}
                info['img_info'] = {'filename': filename}
                info['gt_label'] = np.array(gt_label, dtype=np.int64)
                data_infos.append(info)
            return data_infos
