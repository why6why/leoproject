# -*- coding: utf-8 -*-
# Wei Wang (ww8137@mail.ustc.edu.cn)
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file, You
# can obtain one at http://mozilla.org/MPL/2.0/.
# ==============================================================================
import binascii
import os
import random
import threading
import time

import cv2
import numpy
import numpy as np

PNG_SIZE = 28
labels = ["DDoS", "FTP-Patator", "normal", "PortScan", "SSH-Patator", "web_scan_dir"]


def parse_pcap(filename, width):
    with open(filename, 'rb') as f:
        content = f.read()
    hexst = binascii.hexlify(content)
    fh = numpy.array([int(hexst[i:i + 2], 16) for i in range(0, len(hexst), 2)])
    start, end = 0, width * width
    while end < len(fh):
        try:
            image = numpy.reshape(fh[start:end], (width, width))
            start = end
            end = end + width * width
            save_name = f'{save_path}{int(time.time()):03d}-{int(random.random() * 10000):04d}.png'
            cv2.imwrite(save_name, np.expand_dims(image, axis=-1))
            print(f'保存 {save_name}')
        except Exception as e:
            break


def parse_one_label(path: str, save_path: str):
    pcap_file_list = os.listdir(path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i, pcap_file in enumerate(pcap_file_list):
        print(f'{i + 1}/{len(pcap_file_list)} 读入 {pcap_file}')
        parse_pcap(path + pcap_file, PNG_SIZE)


for label in labels:
    path = f'../../../../datasets/CIC-IDS2017/MachineLearningCVE/real/package/{label}/'
    save_path = f'../../../../datasets/LEO/image28/{label}/'

    thread = threading.Thread(target=parse_one_label, args=(path, save_path))
    thread.start()
