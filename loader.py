import csv
import os
import pdb
import pickle

import numpy as np
import torch
from sklearn import model_selection
from dataclasses import dataclass

import torch.nn.functional as F

from utils import save_json, load_json

KEY_TO_SEMITONE = {'c': 0, 'c#': 1, 'db': 1, 'd-': 1, 'c##': 2, 'd': 2, 'e--': 2, 'd#': 3, 'eb': 3, 'e-': 3, 'd##': 4,
                   'e': 4, 'f-': 4, 'e#': 5, 'f': 5, 'g--': 5, 'e##': 6, 'f#': 6, 'gb': 6, 'g-': 6, 'f##': 7, 'g': 7,
                   'a--': 7, 'g#': 8, 'ab': 8, 'a-': 8, 'g##': 9, 'a': 9, 'b--': 9, 'a#': 10, 'bb': 10, 'b-': 10,
                   'a##': 11, 'b': 11, 'b#': 12, 'c-': -1, 'x': None}

pfd_test = ['001-1_fingering.txt', '001-2_fingering.txt', '001-5_fingering.txt', '001-8_fingering.txt',
            '002-1_fingering.txt', '002-2_fingering.txt', '002-5_fingering.txt', '002-8_fingering.txt',
            '003-1_fingering.txt', '003-2_fingering.txt', '003-5_fingering.txt', '003-8_fingering.txt',
            '004-1_fingering.txt', '004-2_fingering.txt', '004-5_fingering.txt', '004-8_fingering.txt',
            '005-1_fingering.txt', '005-2_fingering.txt', '005-5_fingering.txt', '005-8_fingering.txt',
            '006-1_fingering.txt', '006-2_fingering.txt', '006-5_fingering.txt', '006-8_fingering.txt',
            '007-1_fingering.txt', '007-2_fingering.txt', '007-5_fingering.txt', '007-8_fingering.txt',
            '008-1_fingering.txt', '008-2_fingering.txt', '008-5_fingering.txt', '008-8_fingering.txt',
            '009-1_fingering.txt', '009-2_fingering.txt', '009-5_fingering.txt', '009-8_fingering.txt',
            '010-1_fingering.txt', '010-2_fingering.txt', '010-5_fingering.txt', '010-8_fingering.txt',
            '011-1_fingering.txt', '011-3_fingering.txt', '011-4_fingering.txt', '011-5_fingering.txt',
            '011-6_fingering.txt', '011-7_fingering.txt', '012-1_fingering.txt', '012-3_fingering.txt',
            '012-4_fingering.txt', '012-5_fingering.txt', '012-6_fingering.txt', '012-7_fingering.txt',
            '013-1_fingering.txt', '013-3_fingering.txt', '013-4_fingering.txt', '013-5_fingering.txt',
            '013-6_fingering.txt', '013-7_fingering.txt', '014-1_fingering.txt', '014-3_fingering.txt',
            '014-4_fingering.txt', '014-5_fingering.txt', '014-6_fingering.txt', '014-7_fingering.txt',
            '015-1_fingering.txt', '015-3_fingering.txt', '015-4_fingering.txt', '015-5_fingering.txt',
            '015-6_fingering.txt', '015-7_fingering.txt', '016-1_fingering.txt', '016-3_fingering.txt',
            '016-4_fingering.txt', '016-5_fingering.txt', '016-6_fingering.txt', '016-7_fingering.txt',
            '017-1_fingering.txt', '017-3_fingering.txt', '017-4_fingering.txt', '017-5_fingering.txt',
            '017-6_fingering.txt', '017-7_fingering.txt', '018-1_fingering.txt', '018-3_fingering.txt',
            '018-4_fingering.txt', '018-5_fingering.txt', '018-6_fingering.txt', '018-7_fingering.txt',
            '019-1_fingering.txt', '019-3_fingering.txt', '019-4_fingering.txt', '019-5_fingering.txt',
            '019-6_fingering.txt', '019-7_fingering.txt', '020-1_fingering.txt', '020-3_fingering.txt',
            '020-4_fingering.txt', '020-5_fingering.txt', '020-6_fingering.txt', '020-7_fingering.txt',
            '021-1_fingering.txt', '021-3_fingering.txt', '021-4_fingering.txt', '021-5_fingering.txt',
            '021-6_fingering.txt', '022-1_fingering.txt', '022-3_fingering.txt', '022-4_fingering.txt',
            '022-5_fingering.txt', '022-6_fingering.txt', '023-1_fingering.txt', '023-3_fingering.txt',
            '023-4_fingering.txt', '023-5_fingering.txt', '023-6_fingering.txt', '024-1_fingering.txt',
            '024-3_fingering.txt', '024-4_fingering.txt', '024-5_fingering.txt', '024-6_fingering.txt',
            '025-1_fingering.txt', '025-3_fingering.txt', '025-4_fingering.txt', '025-5_fingering.txt',
            '025-6_fingering.txt', '026-1_fingering.txt', '026-3_fingering.txt', '026-4_fingering.txt',
            '026-5_fingering.txt', '026-6_fingering.txt', '027-1_fingering.txt', '027-3_fingering.txt',
            '027-4_fingering.txt', '027-5_fingering.txt', '027-6_fingering.txt', '028-1_fingering.txt',
            '028-3_fingering.txt', '028-4_fingering.txt', '028-5_fingering.txt', '028-6_fingering.txt',
            '029-1_fingering.txt', '029-3_fingering.txt', '029-4_fingering.txt', '029-5_fingering.txt',
            '029-6_fingering.txt', '030-1_fingering.txt', '030-3_fingering.txt', '030-4_fingering.txt',
            '030-5_fingering.txt', '030-6_fingering.txt'
            ]

pfd_sliced_test = [
    '001-1_fingering.txt', '001-2_fingering.txt', '001-5_fingering.txt', '001-8_fingering.txt',
    '201-1_fingering.txt', '201-2_fingering.txt', '201-5_fingering.txt', '201-8_fingering.txt',
    '202-1_fingering.txt', '202-2_fingering.txt', '202-5_fingering.txt', '202-8_fingering.txt',
    '003-1_fingering.txt', '003-2_fingering.txt', '003-5_fingering.txt', '003-8_fingering.txt',
    '004-1_fingering.txt', '004-2_fingering.txt', '004-5_fingering.txt', '004-8_fingering.txt',
    '005-1_fingering.txt', '005-2_fingering.txt', '005-5_fingering.txt', '005-8_fingering.txt',
    '006-1_fingering.txt', '006-2_fingering.txt', '006-5_fingering.txt', '006-8_fingering.txt',
    '007-1_fingering.txt', '007-2_fingering.txt', '007-5_fingering.txt', '007-8_fingering.txt',
    '008-1_fingering.txt', '008-2_fingering.txt', '008-5_fingering.txt', '008-8_fingering.txt',
    '009-1_fingering.txt', '009-2_fingering.txt', '009-5_fingering.txt', '009-8_fingering.txt',
    '010-1_fingering.txt', '010-2_fingering.txt', '010-5_fingering.txt', '010-8_fingering.txt',
    '011-1_fingering.txt', '011-3_fingering.txt', '011-4_fingering.txt', '011-5_fingering.txt',
    '011-6_fingering.txt', '011-7_fingering.txt', '012-1_fingering.txt', '012-3_fingering.txt',
    '012-4_fingering.txt', '012-5_fingering.txt', '012-6_fingering.txt', '012-7_fingering.txt',
    '013-1_fingering.txt', '013-3_fingering.txt', '013-4_fingering.txt', '013-5_fingering.txt',
    '013-6_fingering.txt', '013-7_fingering.txt', '014-1_fingering.txt', '014-3_fingering.txt',
    '014-4_fingering.txt', '014-5_fingering.txt', '014-6_fingering.txt', '014-7_fingering.txt',
    '015-1_fingering.txt', '015-3_fingering.txt', '015-4_fingering.txt', '015-5_fingering.txt',
    '015-6_fingering.txt', '015-7_fingering.txt', '016-1_fingering.txt', '016-3_fingering.txt',
    '016-4_fingering.txt', '016-5_fingering.txt', '016-6_fingering.txt', '016-7_fingering.txt',
    '017-1_fingering.txt', '017-3_fingering.txt', '017-4_fingering.txt', '017-5_fingering.txt',
    '017-6_fingering.txt', '017-7_fingering.txt', '018-1_fingering.txt', '018-3_fingering.txt',
    '018-4_fingering.txt', '018-5_fingering.txt', '018-6_fingering.txt', '018-7_fingering.txt',
    '019-1_fingering.txt', '019-3_fingering.txt', '019-4_fingering.txt', '019-5_fingering.txt',
    '019-6_fingering.txt', '019-7_fingering.txt', '020-1_fingering.txt', '020-3_fingering.txt',
    '020-4_fingering.txt', '020-5_fingering.txt', '020-6_fingering.txt', '020-7_fingering.txt',
    '021-1_fingering.txt', '021-3_fingering.txt', '021-4_fingering.txt', '021-5_fingering.txt',
    '021-6_fingering.txt', '022-1_fingering.txt', '022-3_fingering.txt', '022-4_fingering.txt',
    '022-5_fingering.txt', '022-6_fingering.txt', '023-1_fingering.txt', '023-3_fingering.txt',
    '023-4_fingering.txt', '023-5_fingering.txt', '023-6_fingering.txt', '024-1_fingering.txt',
    '024-3_fingering.txt', '024-4_fingering.txt', '024-5_fingering.txt', '024-6_fingering.txt',
    '025-1_fingering.txt', '025-3_fingering.txt', '025-4_fingering.txt', '025-5_fingering.txt',
    '025-6_fingering.txt', '026-1_fingering.txt', '026-3_fingering.txt', '026-4_fingering.txt',
    '026-5_fingering.txt', '026-6_fingering.txt', '027-1_fingering.txt', '027-3_fingering.txt',
    '027-4_fingering.txt', '027-5_fingering.txt', '027-6_fingering.txt', '028-1_fingering.txt',
    '028-3_fingering.txt', '028-4_fingering.txt', '028-5_fingering.txt', '028-6_fingering.txt',
    '029-1_fingering.txt', '029-3_fingering.txt', '029-4_fingering.txt', '029-5_fingering.txt',
    '029-6_fingering.txt', '030-1_fingering.txt', '030-3_fingering.txt', '030-4_fingering.txt',
    '030-5_fingering.txt', '030-6_fingering.txt'
]

pfd_train_val = ['031-1_fingering.txt', '032-2_fingering.txt', '032-3_fingering.txt', '033-1_fingering.txt',
                 '034-4_fingering.txt',
                 '035-1_fingering.txt', '036-1_fingering.txt', '037-1_fingering.txt', '038-1_fingering.txt',
                 '039-1_fingering.txt', '040-1_fingering.txt', '041-1_fingering.txt', '042-1_fingering.txt',
                 '043-1_fingering.txt', '043-2_fingering.txt', '044-1_fingering.txt', '045-1_fingering.txt',
                 '045-2_fingering.txt', '046-1_fingering.txt', '046-2_fingering.txt', '047-1_fingering.txt',
                 '047-2_fingering.txt', '048-1_fingering.txt', '048-2_fingering.txt', '049-1_fingering.txt',
                 '049-2_fingering.txt', '050-1_fingering.txt', '050-2_fingering.txt', '051-1_fingering.txt',
                 '052-1_fingering.txt', '053-1_fingering.txt', '054-1_fingering.txt', '055-1_fingering.txt',
                 '056-1_fingering.txt', '057-1_fingering.txt', '058-1_fingering.txt', '059-1_fingering.txt',
                 '060-1_fingering.txt', '061-1_fingering.txt', '061-2_fingering.txt', '062-1_fingering.txt',
                 '062-2_fingering.txt', '063-1_fingering.txt', '063-2_fingering.txt', '064-1_fingering.txt',
                 '064-2_fingering.txt', '065-1_fingering.txt', '065-2_fingering.txt', '066-1_fingering.txt',
                 '066-2_fingering.txt', '067-1_fingering.txt', '067-2_fingering.txt', '068-1_fingering.txt',
                 '068-2_fingering.txt', '069-1_fingering.txt', '069-2_fingering.txt', '070-1_fingering.txt',
                 '070-2_fingering.txt', '071-1_fingering.txt', '071-2_fingering.txt', '072-1_fingering.txt',
                 '073-1_fingering.txt', '074-1_fingering.txt', '074-2_fingering.txt', '075-1_fingering.txt',
                 '076-1_fingering.txt', '076-2_fingering.txt', '077-1_fingering.txt', '077-2_fingering.txt',
                 '078-1_fingering.txt', '079-1_fingering.txt', '079-2_fingering.txt', '080-1_fingering.txt',
                 '081-1_fingering.txt', '082-1_fingering.txt', '083-1_fingering.txt', '084-1_fingering.txt',
                 '085-1_fingering.txt', '086-1_fingering.txt', '087-1_fingering.txt', '088-1_fingering.txt',
                 '089-1_fingering.txt', '090-1_fingering.txt', '091-1_fingering.txt', '092-1_fingering.txt',
                 '093-1_fingering.txt', '094-1_fingering.txt', '095-1_fingering.txt', '096-1_fingering.txt',
                 '097-1_fingering.txt', '098-1_fingering.txt', '099-1_fingering.txt', '100-1_fingering.txt',
                 '101-1_fingering.txt', '102-1_fingering.txt', '103-1_fingering.txt', '104-1_fingering.txt',
                 '105-1_fingering.txt', '106-1_fingering.txt', '107-1_fingering.txt', '108-1_fingering.txt',
                 '109-1_fingering.txt', '110-1_fingering.txt', '111-1_fingering.txt', '112-1_fingering.txt',
                 '113-1_fingering.txt', '113-2_fingering.txt', '114-1_fingering.txt', '115-1_fingering.txt',
                 '115-2_fingering.txt', '116-1_fingering.txt', '117-1_fingering.txt', '118-1_fingering.txt',
                 '119-1_fingering.txt', '120-1_fingering.txt', '121-1_fingering.txt', '121-2_fingering.txt',
                 '122-1_fingering.txt', '122-2_fingering.txt', '123-1_fingering.txt', '123-2_fingering.txt',
                 '124-1_fingering.txt', '124-2_fingering.txt', '125-1_fingering.txt', '125-2_fingering.txt',
                 '126-1_fingering.txt', '126-2_fingering.txt', '127-1_fingering.txt', '127-2_fingering.txt',
                 '128-1_fingering.txt', '128-2_fingering.txt', '129-1_fingering.txt', '129-2_fingering.txt',
                 '130-1_fingering.txt', '130-2_fingering.txt', '131-1_fingering.txt', '131-2_fingering.txt',
                 '132-1_fingering.txt', '132-2_fingering.txt', '133-1_fingering.txt', '134-1_fingering.txt',
                 '135-1_fingering.txt', '136-1_fingering.txt', '137-1_fingering.txt', '138-1_fingering.txt',
                 '139-1_fingering.txt', '140-1_fingering.txt', '140-2_fingering.txt', '141-1_fingering.txt',
                 '142-1_fingering.txt', '142-2_fingering.txt', '143-1_fingering.txt', '144-1_fingering.txt',
                 '145-1_fingering.txt', '146-1_fingering.txt', '147-1_fingering.txt', '148-1_fingering.txt',
                 '149-1_fingering.txt', '150-1_fingering.txt'
                 ]

pfd_train = ['031-1_fingering.txt', '032-2_fingering.txt', '032-3_fingering.txt', '033-1_fingering.txt',
             '034-4_fingering.txt', '035-1_fingering.txt', '037-1_fingering.txt', '038-1_fingering.txt',
             '039-1_fingering.txt', '040-1_fingering.txt', '041-1_fingering.txt', '042-1_fingering.txt',
             '043-1_fingering.txt', '043-2_fingering.txt', '044-1_fingering.txt', '046-1_fingering.txt',
             '046-2_fingering.txt', '047-1_fingering.txt', '047-2_fingering.txt', '049-1_fingering.txt',
             '049-2_fingering.txt', '050-1_fingering.txt', '050-2_fingering.txt', '051-1_fingering.txt',
             '052-1_fingering.txt', '053-1_fingering.txt', '054-1_fingering.txt', '055-1_fingering.txt',
             '056-1_fingering.txt', '057-1_fingering.txt', '058-1_fingering.txt', '059-1_fingering.txt',
             '060-1_fingering.txt', '061-1_fingering.txt', '061-2_fingering.txt', '062-1_fingering.txt',
             '062-2_fingering.txt', '063-1_fingering.txt', '063-2_fingering.txt', '064-1_fingering.txt',
             '064-2_fingering.txt', '065-1_fingering.txt', '065-2_fingering.txt', '066-1_fingering.txt',
             '066-2_fingering.txt', '068-1_fingering.txt', '068-2_fingering.txt', '069-1_fingering.txt',
             '069-2_fingering.txt', '070-1_fingering.txt', '070-2_fingering.txt', '071-1_fingering.txt',
             '071-2_fingering.txt', '072-1_fingering.txt', '074-1_fingering.txt', '074-2_fingering.txt',
             '075-1_fingering.txt', '076-1_fingering.txt', '076-2_fingering.txt', '078-1_fingering.txt',
             '079-1_fingering.txt', '079-2_fingering.txt', '080-1_fingering.txt', '081-1_fingering.txt',
             '082-1_fingering.txt', '083-1_fingering.txt', '084-1_fingering.txt', '085-1_fingering.txt',
             '086-1_fingering.txt', '087-1_fingering.txt', '088-1_fingering.txt', '089-1_fingering.txt',
             '090-1_fingering.txt', '092-1_fingering.txt', '094-1_fingering.txt', '095-1_fingering.txt',
             '097-1_fingering.txt', '099-1_fingering.txt', '100-1_fingering.txt', '101-1_fingering.txt',
             '103-1_fingering.txt', '104-1_fingering.txt', '105-1_fingering.txt', '106-1_fingering.txt',
             '108-1_fingering.txt', '109-1_fingering.txt', '110-1_fingering.txt', '113-1_fingering.txt',
             '113-2_fingering.txt', '114-1_fingering.txt', '115-1_fingering.txt', '115-2_fingering.txt',
             '117-1_fingering.txt', '118-1_fingering.txt', '120-1_fingering.txt', '121-1_fingering.txt',
             '121-2_fingering.txt', '124-1_fingering.txt', '124-2_fingering.txt', '126-1_fingering.txt',
             '126-2_fingering.txt', '127-1_fingering.txt', '127-2_fingering.txt', '128-1_fingering.txt',
             '128-2_fingering.txt', '129-1_fingering.txt', '129-2_fingering.txt', '130-1_fingering.txt',
             '130-2_fingering.txt', '131-1_fingering.txt', '131-2_fingering.txt', '132-1_fingering.txt',
             '132-2_fingering.txt', '134-1_fingering.txt', '135-1_fingering.txt', '136-1_fingering.txt',
             '137-1_fingering.txt', '138-1_fingering.txt', '139-1_fingering.txt', '140-1_fingering.txt',
             '140-2_fingering.txt', '142-1_fingering.txt', '142-2_fingering.txt', '143-1_fingering.txt',
             '146-1_fingering.txt', '147-1_fingering.txt', '148-1_fingering.txt', '149-1_fingering.txt']

pfd_sliced_train = [
    '311-1_fingering.txt', '312-1_fingering.txt', '313-1_fingering.txt', '314-1_fingering.txt',
    '315-1_fingering.txt', '321-2_fingering.txt', '321-3_fingering.txt', '322-2_fingering.txt',
    '322-3_fingering.txt', '331-1_fingering.txt', '332-1_fingering.txt', '341-4_fingering.txt',
    '342-4_fingering.txt', '343-4_fingering.txt', '344-4_fingering.txt', '345-4_fingering.txt',
    '035-1_fingering.txt', '037-1_fingering.txt', '038-1_fingering.txt', '039-1_fingering.txt',
    '040-1_fingering.txt', '041-1_fingering.txt', '042-1_fingering.txt', '043-1_fingering.txt',
    '043-2_fingering.txt', '044-1_fingering.txt', '046-1_fingering.txt',
    '046-2_fingering.txt', '047-1_fingering.txt', '047-2_fingering.txt', '049-1_fingering.txt',
    '049-2_fingering.txt', '050-1_fingering.txt', '050-2_fingering.txt', '051-1_fingering.txt',
    '052-1_fingering.txt', '053-1_fingering.txt', '054-1_fingering.txt', '055-1_fingering.txt',
    '056-1_fingering.txt', '057-1_fingering.txt', '058-1_fingering.txt', '059-1_fingering.txt',
    '060-1_fingering.txt', '061-1_fingering.txt', '061-2_fingering.txt', '062-1_fingering.txt',
    '062-2_fingering.txt', '063-1_fingering.txt', '063-2_fingering.txt', '064-1_fingering.txt',
    '064-2_fingering.txt', '065-1_fingering.txt', '065-2_fingering.txt', '066-1_fingering.txt',
    '066-2_fingering.txt', '068-1_fingering.txt', '068-2_fingering.txt', '069-1_fingering.txt',
    '069-2_fingering.txt', '070-1_fingering.txt', '070-2_fingering.txt', '071-1_fingering.txt',
    '071-2_fingering.txt', '072-1_fingering.txt', '074-1_fingering.txt', '074-2_fingering.txt',
    '075-1_fingering.txt', '076-1_fingering.txt', '076-2_fingering.txt', '078-1_fingering.txt',
    '079-1_fingering.txt', '079-2_fingering.txt', '080-1_fingering.txt', '081-1_fingering.txt',
    '082-1_fingering.txt', '083-1_fingering.txt', '084-1_fingering.txt', '085-1_fingering.txt',
    '086-1_fingering.txt', '087-1_fingering.txt', '088-1_fingering.txt', '089-1_fingering.txt',
    '090-1_fingering.txt', '092-1_fingering.txt', '094-1_fingering.txt', '095-1_fingering.txt',
    '097-1_fingering.txt', '099-1_fingering.txt', '100-1_fingering.txt', '101-1_fingering.txt',
    '103-1_fingering.txt', '104-1_fingering.txt', '105-1_fingering.txt', '106-1_fingering.txt',
    '108-1_fingering.txt', '109-1_fingering.txt', '110-1_fingering.txt', '113-1_fingering.txt',
    '113-2_fingering.txt', '114-1_fingering.txt', '115-1_fingering.txt', '115-2_fingering.txt',
    '117-1_fingering.txt', '118-1_fingering.txt', '120-1_fingering.txt', '121-1_fingering.txt',
    '121-2_fingering.txt', '124-1_fingering.txt', '124-2_fingering.txt', '126-1_fingering.txt',
    '126-2_fingering.txt', '127-1_fingering.txt', '127-2_fingering.txt', '128-1_fingering.txt',
    '128-2_fingering.txt', '129-1_fingering.txt', '129-2_fingering.txt', '130-1_fingering.txt',
    '130-2_fingering.txt', '131-1_fingering.txt', '131-2_fingering.txt', '132-1_fingering.txt',
    '132-2_fingering.txt', '134-1_fingering.txt', '135-1_fingering.txt', '136-1_fingering.txt',
    '137-1_fingering.txt', '138-1_fingering.txt', '139-1_fingering.txt', '140-1_fingering.txt',
    '140-2_fingering.txt', '142-1_fingering.txt', '142-2_fingering.txt', '143-1_fingering.txt',
    '146-1_fingering.txt', '147-1_fingering.txt', '148-1_fingering.txt', '149-1_fingering.txt']

pfd_val = ['036-1_fingering.txt', '045-1_fingering.txt', '045-2_fingering.txt', '048-1_fingering.txt',
           '048-2_fingering.txt', '067-1_fingering.txt', '067-2_fingering.txt', '073-1_fingering.txt',
           '077-1_fingering.txt', '077-2_fingering.txt', '091-1_fingering.txt', '093-1_fingering.txt',
           '096-1_fingering.txt', '098-1_fingering.txt', '102-1_fingering.txt', '107-1_fingering.txt',
           '111-1_fingering.txt', '112-1_fingering.txt', '116-1_fingering.txt', '119-1_fingering.txt',
           '122-1_fingering.txt', '122-2_fingering.txt', '123-1_fingering.txt', '123-2_fingering.txt',
           '125-1_fingering.txt', '125-2_fingering.txt', '133-1_fingering.txt', '141-1_fingering.txt',
           '144-1_fingering.txt', '145-1_fingering.txt', '150-1_fingering.txt']


def save_note2ids():
    dict_id = {}
    for path in os.listdir("PianoFingeringDataset_v1.02/FingeringFiles/"):
        id_piece = f"{int(path[:3])}-{int(path[4])}"
        path = f"PianoFingeringDataset_v1.02/FingeringFiles/{path}"
        with open(path, mode='r') as csvfile:
            r = list(csv.reader(csvfile, delimiter='\t'))[1:]
            r_h = {int(row[0]): KEY_TO_SEMITONE[row[3][:-1].lower()] + int(row[3][-1]) * 12
                   for row in r if row[6] == '0'}
            l_h = {int(row[0]): KEY_TO_SEMITONE[row[3][:-1].lower()] + int(row[3][-1]) * 12
                   for row in r if row[6] == '1'}

            dict_id[id_piece] = {
                'right': r_h,
                'left': l_h
            }
    save_json(dict_id, f"data/note2ids.json")


def load_note2ids():
    return load_json(f"data/note2ids.json")


# save_note2ids()


def create_val():
    piece = [(int(p[:3]), p) for p in pfd_train_val]

    train, val = model_selection.train_test_split(list(range(31, 151)), test_size=0.20)

    train_pfd, test_pfd = [], []
    for n, piece_name in piece:
        if n in train:
            train_pfd.append(piece_name)
        else:
            test_pfd.append(piece_name)
    print(len(train_pfd), len(test_pfd))
    print(train_pfd)
    print(test_pfd)


# create_val()

FINGER_TO_NUM = {
    '-5': 4,
    '-4': 3,
    '-3': 2,
    '-2': 1,
    '-1': 0,
    '0': 10,
    '1': 0,
    '2': 1,
    '3': 2,
    '4': 3,
    '5': 4,
}


def next_onset(onset, sequence_notes, channel):
    # -1 is a impossible value then there is no next
    ans = '-1'
    hand_onsets = list(set([s[1] for s in sequence_notes if int(s[6]) == channel]))
    hand_onsets.sort(key=lambda a: float(a))
    for idx in range(len(hand_onsets)):
        if float(hand_onsets[idx]) > float(onset):
            ans = hand_onsets[idx]
            break
    return ans


def compute_edge_list(sequence_notes, condition):
    edges = []
    for idx, row in enumerate(sequence_notes):
        if row[6] in condition:
            # TODO test maybe with next_same_hand and next_other_hand
            # next labels of right hand
            next_right_hand = next_onset(row[1], sequence_notes, 0)
            next_labels = [(idx, jdx, "next") for jdx, e in enumerate(sequence_notes) if
                           int(row[6]) == 0 and e[1] == next_right_hand and idx != jdx]
            edges.extend(next_labels)
            # next labels of left hand
            next_left_hand = next_onset(row[1], sequence_notes, 1)
            next_labels = [(idx, jdx, "next") for jdx, e in enumerate(sequence_notes) if
                           int(row[6]) == 1 and e[1] == next_left_hand and idx != jdx]
            edges.extend(next_labels)
            # onset labels
            onset_edges = [(idx, jdx, "onset") for jdx, e in enumerate(sequence_notes) if row[1] == e[1] and idx != jdx]
            edges.extend(onset_edges)

    return edges


def nakamura_dataset(set_name, only_left=False, only_right=False, sliced=False):
    if set_name == 'train':
        print("train")
        set = pfd_train
    elif 'validation_experiment' in set_name:
        print(set_name)
        subset, _, _, num = set_name.split('_')
        set = load_json('PianoFingeringDataset_v1.02/validation_experiments_splits.json')[num][subset]
    elif set_name == 'train_sliced':
        print("train_sliced")
        set = pfd_sliced_train
    elif set_name == "val":
        print("val")
        set = pfd_val
    elif set_name == "test_sliced":
        print("test_sliced")
        set = pfd_sliced_test
    elif set_name == "train_val":
        print("train_val")
        set = pfd_train_val
    elif set_name == "test":
        print("test")
        set = pfd_test
    elif set_name == "train_official":
        print("train_official")
        set = load_json('PianoFingeringDataset_v1.02/official_split.json')['train']
    elif set_name == "val_official":
        print("val_official")
        set = load_json('PianoFingeringDataset_v1.02/official_split.json')['val']
    elif set_name == "test_official":
        print("test_official")
        set = load_json('PianoFingeringDataset_v1.02/official_split.json')['test']

    if only_left:
        condition = ['1']
    elif only_right:
        condition = ['0']
    else:
        condition = ['0', '1']

    if sliced:
        main_path = 'FingeringFilesSliced'
    else:
        main_path = 'FingeringFiles'

    note, onset, duration, finger, ids, lengths, edges = [], [], [], [], [], [], []

    for piece in set:
        with open(f"PianoFingeringDataset_v1.02/{main_path}/{piece}", mode='r') as csvfile:
            r = list(csv.reader(csvfile, delimiter='\t'))[1:]
            n, o, d, f, rr = [], [], [], [], []
            for row in r:
                if row[6] in condition:
                    n.append(KEY_TO_SEMITONE[row[3][:-1].lower()] + int(row[3][-1]) * 12)
                    o.append(float(row[1]))
                    d.append(float(row[2]) - float(row[1]))
                    # TODO how to manage the change of fingers? e.g.  '-5_-1'
                    f.append(FINGER_TO_NUM[row[7].split('_')[0]])
                    rr.append(row)
        note.append(normalize_midi(np.array(n)))
        onset.append(np.array(o))
        duration.append(np.array(d))
        finger.append(np.array(f))
        ids.append((int(piece[:3]), int(piece[4])))
        lengths.append(len(n))
        edges.append(compute_edge_list(rr, condition))

    return note, onset, duration, finger, ids, lengths, edges


def next_onset_window(onset, onsets):
    # -1 is a impossible value then there is no next
    ans = '-1'
    hand_onsets = list(set(onsets))
    hand_onsets.sort(key=lambda a: float(a))
    for idx in range(len(hand_onsets)):
        if float(hand_onsets[idx]) > float(onset):
            ans = hand_onsets[idx]
            break
    return ans


def compute_edge_list_window(w):
    edges = []
    for idx, (current_onset, current_pitch) in enumerate(zip(w['onsets'], w['pitchs'])):
        # pdb.set_trace()
        if current_pitch != 0:
            # next labels of right hand
            next_right_hand = next_onset_window(current_onset, w['onsets'])
            next_labels = [(idx, jdx, "next") for jdx, onset in enumerate(w['onsets']) if
                           onset == next_right_hand and idx != jdx]
            edges.extend(next_labels)
            # onset labels
            onset_edges = [(idx, jdx, "onset") for jdx, onset in enumerate(w['onsets']) if
                           current_onset == onset and idx != jdx]
            edges.extend(onset_edges)
    return edges


def normalize_data_tensor(data):
    ans = torch.nan_to_num((data - torch.min(data)) / (torch.max(data) - torch.min(data)), nan=0)
    return ans


def normalize_data(data):
    ans = np.nan_to_num((data - np.min(data)) / (np.max(data) - np.min(data)), nan=0)
    return ans


def normalize_midi(data):
    return data / 127.0


def musescore_dataset(set_name, hand='right', w_type='w11'):
    if w_type == 'w11':
        windows = load_binary('data/musescore_fingers.pickle')[hand][set_name]
    elif w_type == 'random':
        windows = load_binary('data/musescore_fingers_w_up64.pickle')[hand][set_name]

    note, onset, duration, finger, ids, lengths, edges = [], [], [], [], [], [], []
    for w in windows:
        note.append(normalize_midi(np.array(w['pitchs'], dtype=float)))
        onset.append(normalize_data(np.array(w['onsets'], dtype=float)))
        duration.append(normalize_data(np.array(w['offsets'], dtype=float) - np.array(w['onsets'], dtype=float)))
        finger.append(np.array(w['fingers']) - 1)
        ids.append((w['alias'].split('.')[0], w['alias'][-1]))
        lengths.append(len(w['pitchs']))
        edges.append(compute_edge_list_window(w))
    return note, onset, duration, finger, ids, lengths, edges


def save_binary(dictionary, name_file):
    with open(name_file, 'wb') as fp:
        pickle.dump(dictionary, fp, protocol=pickle.HIGHEST_PROTOCOL)


def load_binary(name_file):
    data = None
    with open(name_file, 'rb') as fp:
        data = pickle.load(fp)
    return data


def save_load_pfd():
    train = ('train')
    validation = nakamura_dataset('val')
    test = nakamura_dataset('test')

    data = (train, validation, test)
    save_binary(data, "data/nak1_0.pickle")


def save_load_pfd_sliced():
    train = nakamura_dataset('train_sliced', sliced=True)
    validation = nakamura_dataset('val', sliced=True)
    test = nakamura_dataset('test_sliced', sliced=True)

    data = (train, validation, test)
    save_binary(data, "data/nak1_0_sliced.pickle")


def salami(arr, window_size, hop_size, normalize=False):
    new_arr = []
    arr = np.concatenate(([0] * 5, arr, ([0] * 5)))
    for idx in range(0, len(arr) - window_size + 1, hop_size):
        if normalize:
            new_arr.append(normalize_data(arr[idx:idx + window_size]))
        else:
            new_arr.append(arr[idx:idx + window_size])
    return new_arr


def salami_tensor(arr, window_size, hop_size, normalize=False, device=None, padding_value=0):
    new_arr = []
    if device is None:
        zero_p = torch.ones(arr.shape[0], 5, arr.shape[2]) * padding_value
    else:
        zero_p = torch.ones(arr.shape[0], 5, arr.shape[2]).to(device) * padding_value
    arr = torch.cat((zero_p, arr, zero_p), dim=1)
    for idx in range(0, arr.shape[1] - window_size + 1, hop_size):
        if normalize:
            new_arr.append(normalize_data_tensor(arr[:, idx: idx + window_size, :]))
        else:
            new_arr.append(arr[:, idx: idx + window_size, :])
    return new_arr


def filter_edges(edges, window_size, hop_size, len_notes):
    new_edges = []

    for idx in range(0, len_notes, hop_size):
        edges_segment = [
            (e[0] - idx + 5, e[1] - idx + 5, e[2])
            for e in edges if idx <= e[0] + 5 < idx + window_size and idx <= e[1] + 5 < idx + window_size
        ]
        new_edges.append(edges_segment)
    # pdb.set_trace()
    return new_edges


def salamizer(dataset, ws, hs):
    # ws := window_size, hs := hop_size
    new_notes, new_onsets, new_durations, new_fingers, new_ids, new_lengths, new_edges = [], [], [], [], [], [], []
    for notes, onsets, durations, fingers, ids, lengths, edges in zip(*dataset):
        print(ids)
        notes_windowed = salami(notes, window_size=ws, hop_size=hs, normalize=False)
        new_notes.extend(notes_windowed)
        new_onsets.extend(salami(onsets, window_size=ws, hop_size=hs, normalize=True))
        new_durations.extend(salami(durations, window_size=ws, hop_size=hs, normalize=False))
        new_fingers.extend(salami(fingers, window_size=ws, hop_size=hs, normalize=False))
        new_edges.extend(filter_edges(edges, window_size=ws, hop_size=hs, len_notes=len(notes_windowed)))
        new_ids.extend([ids] * len(notes_windowed))
        new_lengths.extend([ws] * len(notes_windowed))
    return new_notes, new_onsets, new_durations, new_fingers, new_ids, new_lengths, new_edges


def save_load_pfd_right_w11():
    train = nakamura_dataset('train', only_right=True)
    validation = nakamura_dataset('val', only_right=True)
    test = nakamura_dataset('test', only_right=True)

    windowed = salamizer(train, 11, 1)

    data = (train, validation, test, windowed)
    save_binary(data, "data/nak1_0_right_w11.pickle")


# save_load_pfd_right_w11()


def save_load_pfd_right():
    train = nakamura_dataset('train', only_right=True)
    validation = nakamura_dataset('val', only_right=True)
    test = nakamura_dataset('test', only_right=True)

    data = (train, validation, test)
    save_binary(data, "data/nak1_0_right.pickle")


# save_load_pfd_right()

# save_load_pfd_right_w11()


def save_load_pfd_w59():
    train = nakamura_dataset('train')
    validation = nakamura_dataset('val')
    test = nakamura_dataset('test')

    train = salamizer(train, 59, 1)
    validation = salamizer(validation, 59, 1)
    test = salamizer(test, 59, 1)

    data = (train, validation, test)
    save_binary(data, "data/nak1_0_w59.pickle")


def save_pfd_right_noisy_w11():
    train = nakamura_dataset('train', only_right=True)
    validation = nakamura_dataset('val', only_right=True)
    test = nakamura_dataset('test', only_right=True)
    windowed = salamizer(train, 11, 1)
    noisy_train = musescore_dataset('train', hand='right')
    noisy_validation = musescore_dataset('validation', hand='right')

    data = (train, validation, test, windowed)
    save_binary(data, "data/nakamura_right_w11.pickle")
    data_noisy = (noisy_train, noisy_validation)
    save_binary(data_noisy, "data/musescore_right_w11.pickle")


# save_pfd_right_noisy_w11()

def first_note_symmetric(note, from_hand='left'):
    right2left_pitch_class_symmetric = {
        0: 4,
        1: 2,
        2: 0,
        3: -2,
        4: -4,
        5: -6,
        6: -8,
        7: -10,
        8: -12,
        9: -14,
        10: -16,
        11: -18
    }
    left2right_pitch_class_symmetric = {
        0: 16,
        1: 14,
        2: 12,
        3: 10,
        4: 8,
        5: 6,
        6: 4,
        7: 2,
        8: 0,
        9: -2,
        10: -4,
        11: -6
    }
    pitch_class = note % 12  # 4
    d_oct = (note - 60) // 12  # -1

    if from_hand == 'left':
        ans = note + left2right_pitch_class_symmetric[pitch_class] - (2 * d_oct * 12) - 24
    else:
        ans = note + right2left_pitch_class_symmetric[pitch_class] - (2 * d_oct * 12)
    return ans


# print("sim", first_note_symmetric(68))

def _surpass_bounds(notes):
    surpass = False
    for n in notes:
        if not (n == 0 or (21 <= n < 108)):
            surpass = True
    return surpass


def reverse_hand(data, bounds=False):
    list_notes, list_onsets, list_durations, list_fingers, list_ids, list_lengths, list_edges = [], [], [], [], [], [], []
    for notes, onsets, durations, fingers, ids, lengths, edges in zip(*data):
        new_notes = []
        notes = notes * 127
        jdx = 0
        for idx, n in enumerate(notes):
            if n == 0:
                jdx += 1
                new_notes.append(0)
            elif idx == jdx:
                new_notes.append(first_note_symmetric(notes[idx]))
            else:
                is_black_current = (n % 12) in [1, 3, 6, 8, 10]
                distance = n - notes[idx - 1]
                new_n = new_notes[-1] - distance
                is_black_new = (new_n % 12) in [1, 3, 6, 8, 10]
                new_notes.append(new_n)
                assert is_black_current == is_black_new, " is not working symmetric hand data augmentation " \
                                                         f"original seq = {np.array(notes)} " \
                                                         f"new seq = {np.array(new_notes)}"

        new_notes = np.array(new_notes)
        if bounds:
            if _surpass_bounds(new_notes):
                print(f"surpass piano keyboard bounds "
                      f"original seq = {np.array(notes)} "
                      f"new seq = {np.array(new_notes)}")
                continue
        list_notes.append(new_notes / 127)
        list_onsets.append(onsets)
        list_durations.append(durations)
        list_fingers.append(fingers)
        list_ids.append(ids)
        list_lengths.append(lengths)
        list_edges.append(edges)
    return list_notes, list_onsets, list_durations, list_fingers, list_ids, list_lengths, list_edges


def time_symmetry(notes, onsets, durations, fingers, idx, lengths, edges):
    notes = np.flip(notes)
    fingers = np.flip(fingers)
    # pdb.set_trace()
    # print(notes, edges)
    edges = [(jdx, idx, t) for idx, jdx, t in edges]
    new_onsets = []
    for o in np.flip(onsets):
        assert 0 <= o <= 1.2, f"{o} is not normalized"
        d = o - 0.5
        new_onsets.append(o - (2 * d))
    return np.array(notes), np.array(new_onsets), np.array(durations), np.array(fingers), idx, lengths, edges


def octave_symmetry(notes, onsets, durations, fingers, idx, lengths, edges):
    list_notes, list_onsets, list_durations, list_fingers, list_ids, list_lengths, list_edges = [], [], [], [], [], [], []
    octaves = [-9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for octave in octaves:
        new_notes = [((n * 127) + octave * 12) if n != 0 else 0 for n in notes]
        if not _surpass_bounds(new_notes):
            list_notes.append(np.array(new_notes) / 127)
            list_onsets.append(onsets)
            list_durations.append(durations)
            list_fingers.append(fingers)
            list_ids.append(idx)
            list_lengths.append(lengths)
            list_edges.append(edges)
    return list_notes, list_onsets, list_durations, list_fingers, list_ids, list_lengths, list_edges


def merge_windows(windowed_lh, windowed_rh):
    list_notes, list_onsets, list_durations, list_fingers, list_ids, list_lengths, list_edges = windowed_lh[0] + \
                                                                                                windowed_rh[0], \
                                                                                                windowed_lh[1] + \
                                                                                                windowed_rh[1], \
                                                                                                windowed_lh[2] + \
                                                                                                windowed_rh[2], \
                                                                                                windowed_lh[3] + \
                                                                                                windowed_rh[3], \
                                                                                                windowed_lh[4] + \
                                                                                                windowed_rh[4], \
                                                                                                windowed_lh[5] + \
                                                                                                windowed_rh[5], \
                                                                                                windowed_lh[6] + \
                                                                                                windowed_rh[6]

    return list_notes, list_onsets, list_durations, list_fingers, list_ids, list_lengths, list_edges


def data_augmentation(windowed):
    windowed_augmented = []
    # pdb.set_trace()
    for notes, onsets, durations, fingers, ids, lengths, edges in zip(*windowed):
        augmentations = []
        octave_augmentations = octave_symmetry(notes, onsets, durations, fingers, ids, lengths, edges)
        for nn, o, d, f, idx, l, e in zip(*octave_augmentations):
            augmentations.append((nn, o, d, f, idx, l, e))
            augmentations.append(time_symmetry(nn, o, d, f, idx, l, e))
        for idx in range(len(augmentations)):
            for midi_note in (augmentations[idx][0] * 127):
                assert midi_note.is_integer(), "something wrong with midi notes"
        windowed_augmented.append(augmentations)
    return windowed_augmented


def normalize_seq2seq(data):
    list_notes, list_onsets, list_durations, list_fingers, list_ids, list_lengths, list_edges = data
    new_list_onsets, new_list_durations = [], []
    for o, d in zip(list_onsets, list_durations):
        new_list_onsets.append(normalize_data(o))
        new_list_durations.append(normalize_data(d))
    return list_notes, new_list_onsets, new_list_durations, list_fingers, list_ids, list_lengths, list_edges


def save_pfd_seq2seq_w11():
    train_lh = reverse_hand(nakamura_dataset('train', only_left=True))
    val_lh = reverse_hand(nakamura_dataset('val', only_left=True), bounds=True)
    test_lh = reverse_hand(nakamura_dataset('test', only_left=True), bounds=True)

    train_rh = nakamura_dataset('train', only_right=True)
    val_rh = nakamura_dataset('val', only_right=True)
    test_rh = nakamura_dataset('test', only_right=True)

    train = merge_windows(train_lh, train_rh)
    train = normalize_seq2seq(train)
    train = data_augmentation(train)

    data = (train, train_rh, train_lh, val_rh, val_lh, test_rh, test_lh)
    save_binary(data, "data/augmented/naka_seq2seq_augmented_w11.pickle")


# save_pfd_seq2seq_w11()


def save_pfd_nakamura():
    train_lh = reverse_hand(nakamura_dataset('train', only_left=True))
    val_lh = reverse_hand(nakamura_dataset('val', only_left=True), bounds=True)
    test_lh = reverse_hand(nakamura_dataset('test', only_left=True), bounds=True)

    train_rh = nakamura_dataset('train', only_right=True)
    val_rh = nakamura_dataset('val', only_right=True)
    test_rh = nakamura_dataset('test', only_right=True)

    train = merge_windows(train_lh, train_rh)
    train = normalize_seq2seq(train)
    train = data_augmentation(train)

    data = (train, train_rh, train_lh, val_rh, val_lh, test_rh, test_lh)
    save_binary(data, "data/augmented/naka_seq2seq_augmented_w11.pickle")


# save_pfd_nakamura()


def load_pfd_seq2seq_w11():
    data = load_binary("data/augmented/naka_seq2seq_augmented_w11.pickle")
    print("dataset seq2seqaugmented_w11!!")
    return data


def save_pfd_full_augmented_w11():
    # load nakamura left hand and right hand
    train_lh = reverse_hand(nakamura_dataset('train', only_left=True))
    val_lh = reverse_hand(nakamura_dataset('val', only_left=True), bounds=True)
    test_lh = reverse_hand(nakamura_dataset('test', only_left=True), bounds=True)

    train_rh = nakamura_dataset('train', only_right=True)
    val_rh = nakamura_dataset('val', only_right=True)
    test_rh = nakamura_dataset('test', only_right=True)

    # create windows
    windowed_lh = salamizer(train_lh, 11, 1)
    windowed_rh = salamizer(train_rh, 11, 1)

    windowed = merge_windows(windowed_lh, windowed_rh)
    windowed = data_augmentation(windowed)

    # data = (train_lh, validation_lh, test_lh, train_rh, validation_rh, test_rh, windowed)
    data = (windowed, val_rh, val_lh, test_rh, test_lh)
    save_binary(data, "data/augmented/nakamura_augmented_w11.pickle")


def save_validation_experiments():
    for exp_num in range(1, 6):
        # load nakamura left hand and right hand
        train_lh = reverse_hand(nakamura_dataset(f'train_validation_experiment_{exp_num}', only_left=True))
        val_lh = reverse_hand(nakamura_dataset(f'val_validation_experiment_{exp_num}', only_left=True), bounds=True)
        test_lh = reverse_hand(nakamura_dataset(f'test_validation_experiment_{exp_num}', only_left=True), bounds=True)
        test_fair_lh = reverse_hand(nakamura_dataset(f'test-fair_validation_experiment_{exp_num}', only_left=True),
                                    bounds=True)

        train_rh = nakamura_dataset(f'train_validation_experiment_{exp_num}', only_right=True)
        val_rh = nakamura_dataset(f'val_validation_experiment_{exp_num}', only_right=True)
        test_rh = nakamura_dataset(f'test_validation_experiment_{exp_num}', only_right=True)
        test_fair_rh = nakamura_dataset(f'test-fair_validation_experiment_{exp_num}', only_right=True)

        # create windows
        windowed_lh = salamizer(train_lh, 11, 1)
        windowed_rh = salamizer(train_rh, 11, 1)

        windowed = merge_windows(windowed_lh, windowed_rh)
        windowed = data_augmentation(windowed)

        # data = (train_lh, validation_lh, test_lh, train_rh, validation_rh, test_rh, windowed)
        data = (windowed, val_rh, val_lh, test_rh, test_lh, test_fair_rh, test_fair_lh)
        save_binary(data, f"data/augmented/validation_experiment_{exp_num}.pickle")


def save_validation_experiments_seq2seq():
    for exp_num in range(1, 6):
        # load nakamura left hand and right hand
        train_lh = reverse_hand(nakamura_dataset(f'train_validation_experiment_{exp_num}', only_left=True))
        val_lh = reverse_hand(nakamura_dataset(f'val_validation_experiment_{exp_num}', only_left=True), bounds=True)
        test_lh = reverse_hand(nakamura_dataset(f'test_validation_experiment_{exp_num}', only_left=True), bounds=True)

        train_rh = nakamura_dataset(f'train_validation_experiment_{exp_num}', only_right=True)
        val_rh = nakamura_dataset(f'val_validation_experiment_{exp_num}', only_right=True)
        test_rh = nakamura_dataset(f'test_validation_experiment_{exp_num}', only_right=True)
        # create windows
        train_rh = normalize_seq2seq(train_rh)
        train_lh = normalize_seq2seq(train_lh)
        train = merge_windows(train_rh, train_lh)
        train = data_augmentation(train)
        # data = (train_lh, validation_lh, test_lh, train_rh, validation_rh, test_rh, windowed)
        data = (train, train_rh, train_lh, val_rh, val_lh, test_rh, test_lh)
        save_binary(data, f"data/augmented/validation_experiment_seq2seq_{exp_num}.pickle")


# save_validation_experiments_seq2seq()


def load_validation_experiment(name_exp):
    data = load_binary(f"data/augmented/{name_exp}.pickle")
    print(name_exp)
    return data


def save_noisy_full_augmented_w11():
    # load nakamura left hand and right hand
    test_lh = reverse_hand(nakamura_dataset('test', only_left=True), bounds=True)
    test_rh = nakamura_dataset('test', only_right=True)

    validation_lh = reverse_hand(nakamura_dataset('train_val', only_left=True), bounds=True)
    validation_lh = salamizer(validation_lh, 11, 1)
    validation_rh = nakamura_dataset('train_val', only_right=True)
    validation_rh = salamizer(validation_rh, 11, 1)

    # create windows
    noisy_train_rh = musescore_dataset('train', hand='right', w_type='w11')
    noisy_train_lh = reverse_hand(musescore_dataset('train', hand='left', w_type='w11'))

    noisy_validation_rh = musescore_dataset('validation', hand='right', w_type='w11')
    noisy_validation_lh = reverse_hand(musescore_dataset('validation', hand='left', w_type='w11'), bounds=True)

    noisy_windowed = merge_windows(noisy_train_rh, noisy_train_lh)
    noisy_windowed = merge_windows(noisy_windowed, noisy_validation_rh)
    noisy_windowed = merge_windows(noisy_windowed, noisy_validation_lh)
    noisy_windowed = data_augmentation(noisy_windowed)

    data = (test_rh, test_lh, validation_rh, validation_lh, noisy_windowed)
    save_binary(data, "data/augmented/musescore_augmented_w11.pickle")


# save_noisy_full_augmented_w11()


def save_nakamura_seq2seq_merged():
    train_lh = reverse_hand(nakamura_dataset(f'train_official', only_left=True))
    val_lh = reverse_hand(nakamura_dataset(f'val_official', only_left=True), bounds=True)
    test_lh = reverse_hand(nakamura_dataset(f'test_official', only_left=True), bounds=True)

    train_rh = nakamura_dataset(f'train_official', only_right=True)
    val_rh = nakamura_dataset(f'val_official', only_right=True)
    test_rh = nakamura_dataset(f'test_official', only_right=True)
    # create windows
    train_rh = normalize_seq2seq(train_rh)
    train_lh = normalize_seq2seq(train_lh)
    train = merge_windows(train_rh, train_lh)
    train = data_augmentation(train)

    data = (train, train_rh, train_lh, val_rh, val_lh, test_rh, test_lh)
    save_binary(data, "data/augmented/nakamura_augmented_seq2seq_merged.pickle")

    train_rh_augmented = data_augmentation(train_rh)
    train_lh_augmented = data_augmentation(train_lh)
    data = (train_rh_augmented, train_lh_augmented, train_rh, train_lh, val_rh, val_lh, test_rh, test_lh)
    save_binary(data, "data/augmented/nakamura_augmented_seq2seq_no_merged.pickle")


def save_nakamura_no_augmented_seq2seq():
    train_lh = reverse_hand(nakamura_dataset(f'train_official', only_left=True))
    val_lh = reverse_hand(nakamura_dataset(f'val_official', only_left=True), bounds=True)
    test_lh = reverse_hand(nakamura_dataset(f'test_official', only_left=True), bounds=True)

    train_rh = nakamura_dataset(f'train_official', only_right=True)
    val_rh = nakamura_dataset(f'val_official', only_right=True)
    test_rh = nakamura_dataset(f'test_official', only_right=True)
    # create windows
    train_rh = normalize_seq2seq(train_rh)
    train_lh = normalize_seq2seq(train_lh)
    train = merge_windows(train_rh, train_lh)

    data = (train, train_rh, train_lh, val_rh, val_lh, test_rh, test_lh)
    save_binary(data, "data/augmented/nakamura_no_augmented_seq2seq_no_merged.pickle")


def load_nakamura_no_augmented_seq2seq():
    data = load_binary("data/augmented/nakamura_no_augmented_seq2seq_no_merged.pickle")
    print("dataset merged!!")
    return data

# save_nakamura_no_augmented_seq2seq()


def save_generalization():
    test_lh = reverse_hand(nakamura_dataset('train_val', only_left=True), bounds=True)
    test_rh = nakamura_dataset('train_val', only_right=True)

    train_lh = reverse_hand(nakamura_dataset('test', only_left=True), bounds=True)
    train_lh = normalize_seq2seq(train_lh)

    train_rh = nakamura_dataset('test', only_right=True)
    train_rh = normalize_seq2seq(train_rh)

    train = merge_windows(train_rh, train_lh)
    train = data_augmentation(train)

    data = (train, test_rh, test_lh)
    save_binary(data, "data/augmented/nakamura_generalization.pickle")


# save_generalization()

def load_generalization():
    data = load_binary("data/augmented/nakamura_generalization.pickle")
    print("dataset generalizationn experiment!!")
    return data


def load_nakamura_augmented_seq2seq_merged():
    data = load_binary("data/augmented/nakamura_augmented_seq2seq_merged.pickle")
    print("dataset merged!!")
    return data


def load_nakamura_augmented_seq2seq_no_merged():
    data = load_binary("data/augmented/nakamura_augmented_seq2seq_merged.pickle")
    print("dataset no merged!!")
    return data


def save_noisy_random_seq2seq():
    # load nakamura left hand and right hand
    test_lh = reverse_hand(nakamura_dataset('test', only_left=True), bounds=True)
    test_rh = nakamura_dataset('test', only_right=True)
    validation_lh = reverse_hand(nakamura_dataset('train_val', only_left=True), bounds=True)
    validation_rh = nakamura_dataset('train_val', only_right=True)
    # create windows
    noisy_train_rh = musescore_dataset('train', hand='right', w_type='random')
    noisy_train_lh = reverse_hand(musescore_dataset('train', hand='left', w_type='random'))

    noisy_validation_rh = musescore_dataset('validation', hand='right', w_type='random')
    noisy_validation_lh = reverse_hand(musescore_dataset('validation', hand='left', w_type='random'), bounds=True)

    noisy_windowed = merge_windows(noisy_train_rh, noisy_train_lh)
    noisy_windowed = merge_windows(noisy_windowed, noisy_validation_rh)
    noisy_windowed = merge_windows(noisy_windowed, noisy_validation_lh)
    noisy_windowed = data_augmentation(noisy_windowed)

    data = (test_rh, test_lh, validation_rh, validation_lh, noisy_windowed)
    save_binary(data, "data/augmented/musescore_augmented_random_seq2seq.pickle")


def save_no_augmented_noisy_random_seq2seq():
    # load nakamura left hand and right hand
    test_lh = reverse_hand(nakamura_dataset('test', only_left=True), bounds=True)
    test_rh = nakamura_dataset('test', only_right=True)
    validation_lh = reverse_hand(nakamura_dataset('train_val', only_left=True), bounds=True)
    validation_rh = nakamura_dataset('train_val', only_right=True)
    # create windows
    noisy_train_rh = musescore_dataset('train', hand='right', w_type='random')
    noisy_train_lh = reverse_hand(musescore_dataset('train', hand='left', w_type='random'))

    noisy_validation_rh = musescore_dataset('validation', hand='right', w_type='random')
    noisy_validation_lh = reverse_hand(musescore_dataset('validation', hand='left', w_type='random'), bounds=True)

    noisy_windowed = merge_windows(noisy_train_rh, noisy_train_lh)
    noisy_windowed = merge_windows(noisy_windowed, noisy_validation_rh)
    noisy_windowed = merge_windows(noisy_windowed, noisy_validation_lh)

    data = (test_rh, test_lh, validation_rh, validation_lh, noisy_windowed)
    save_binary(data, "data/augmented/musescore_no_augmented_random_seq2seq.pickle")


# save_no_augmented_noisy_random_seq2seq()


def save_dasaem_test():
    # load nakamura left hand and right hand
    test_lh = reverse_hand(nakamura_dataset('test', only_left=True), bounds=True)
    test_rh = nakamura_dataset('test', only_right=True)

    data = (test_rh, test_lh)
    save_binary(data, "data/dasaem_test.pickle")


# save_dasaem_test()


def load_dasaem_test():
    data = load_binary("data/dasaem_test.pickle")
    print("dataset dasaem!!")
    return data


def load_noisy_full_augmented_w11():
    data = load_binary("data/augmented/musescore_augmented_w11.pickle")
    print("dataset noisy_full_noisy_augmented_w11!!")
    return data


def load_noisy_random_seq2seq():
    data = load_binary("data/augmented/musescore_augmented_random_seq2seq.pickle")
    print("dataset noisy_random_seq2seq!!")
    return data


def load_no_augmented_noisy_random_seq2seq():
    data = load_binary("data/augmented/musescore_no_augmented_random_seq2seq.pickle")
    print("dataset no augmented noisy_random_seq2seq!!")
    return data


# load_noisy_random_seq2seq()

def load_pfd_right_noisy_w11():
    train, validation, test, windowed = load_binary("data/nakamura_right_w11.pickle")
    noisy_train, noisy_validation = load_binary("data/musescore_right_w11.pickle")
    print("dataset right_noisy_w11!!")
    return train, validation, test, windowed, noisy_train, noisy_validation


# load_pfd_right_noisy_w11()

def load_pfd():
    data = load_binary("data/nak1_0.pickle")
    print("dataset loaded!!")
    return data


def load_pfd_full_augmented_w11():
    data = load_binary("data/augmented/nakamura_augmented_w11.pickle")
    print("dataset loaded!!")

    return data


# pfd_full_augmented_w11()


def load_pfd_sliced():
    data = load_binary("data/nak1_0_sliced.pickle")
    print("dataset loaded!!")
    return data


def load_pfd_w59():
    data = load_binary("data/nak1_0_w59.pickle")
    print("dataset loaded!!")
    return data


def load_pfd_right_w11():
    data = load_binary("data/nak1_0_right_w11.pickle")
    print("dataset loaded!!")
    return data


# load_pfd_right_w11()

def load_pfd_right():
    data = load_binary("data/nak1_0_right.pickle")
    print("dataset loaded!!")
    return data

# load_pfd_right()
# save_load_pfd_w59()
# data = load_pfd_w59()
# print()

# note, onset, duration, finger, ids, lengths, edges = load_pfd('train')
# a = []
# max_length = int(max(lengths))
# for edge, length in zip(edges, lengths):
#     aa = edges_to_matrix(edge, length)
#     new_aa = np.pad(aa, ((0, 0), (0, max_length), (0, max_length)), 'constant')
#     a.append(new_aa)
# pdb.set_trace()
# print(note[0], onset[0], duration[0], finger[0], ids[0], lengths[0])


# for file in pfd_train_val:
#     copyfile(f"PianoFingeringDataset_v1.02/FingeringFiles/{file}", f"nakamura_SourceCode/train/{file}")
#
# for file in pfd_test:
#     print(file)


# for file in pfd_test:
#     copyfile(f"PianoFingeringDataset_v1.02/FingeringFiles/{file}", f"nakamura_SourceCode/test/{file}")

if __name__ == '__main__':
    save_noisy_random_seq2seq()
    save_nakamura_seq2seq_merged()