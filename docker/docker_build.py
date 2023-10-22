#!/usr/bin/env python3
import os

if __name__=="__main__":
    cmd = "nvidia-docker build -t panda_docker_env . "
    code = os.system(cmd)
