#!/bin/bash
time MAX_JOBS=16 MAKEFLAGS="-j$(nproc)" python setup.py develop