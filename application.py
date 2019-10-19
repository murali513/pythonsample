from flask import Flask
from flask import request, Flask
from flask_cors import CORS



# Setup configuration file
import configparser

config = configparser.ConfigParser()
config.read('api_config.ini')

# Setup logs
import logging


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

import cv2
from json2xml import readfromstring, json2xml
import json
import re
import xml.etree.ElementTree as xml


app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"
