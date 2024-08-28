from os import path
import sys

import alog

sys.path.append(path.abspath(path.join(path.dirname(__file__), "../")))

import text_sentiment

alog.configure(default_level="debug")
from caikit.runtime import grpc_server

grpc_server.main()
