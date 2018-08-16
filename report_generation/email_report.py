#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script to generate an e-mail report of benchmarks from Benchmark.AI"""

import logging
import pickle
import sys

from metrics import gather_benchmarks, parse_metadata
from pprint import pprint

def main():
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger('boto3').setLevel(logging.CRITICAL)
    logging.getLogger('botocore').setLevel(logging.CRITICAL)
    logging.info("Gathering metrics")

    debug = False
    if debug:
        benchmarks = pickle.load(open("benchmarks.pkl", "rb"))
    else:
        benchmarks = gather_benchmarks(use_cache=True)
        pickle.dump(benchmarks, open("benchmarks.pkl", "wb"))


    return 0

if __name__ == '__main__':
    main()

