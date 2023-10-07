import inspect
import threading
import logging
import os

import datetime as dt

from termcolor import cprint
from pyfiglet import figlet_format


class MetisASCIIArt(object):

    @classmethod
    def print(cls):
        # Print 'METIS Federated Learning' on console as an ASCII-Art pattern.
        cprint(figlet_format('METIS', font='greek'),
               'blue', None, attrs=['bold'], flush=True)
        cprint(figlet_format('Federated Learning Framework', width=150),
               'blue', None, attrs=['bold'], flush=True)

