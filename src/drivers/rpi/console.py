# -*- coding: utf-8 -*-
"""
Copyright (c) 2023 The uos_sess6072_build Authors.
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import datetime
import getpass
import logging
import os
from pathlib import Path
import socket
import sys
import timeit

import pkg_resources


logger = None  # Public logger
verbose = False

# Fix for Windows terminal
os.system("")


class BColors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class CodeTimer:
    def __init__(self, name=None):
        self.name = " '" + name + "'" if name else ""

    def __enter__(self):
        self.start = timeit.default_timer()

    def __exit__(self, exc_type, exc_value, traceback):
        self.took = (timeit.default_timer() - self.start) * 1000.0
        print(
            BColors.OKBLUE + self.name + " took > " + BColors.ENDC + str(self.took) + " ms"
        )


class Console:
    """Console utility functions."""

    @staticmethod
    def set_verbosity(verbosity) -> None:
        global verbose
        verbose = verbosity

    @staticmethod
    def warn(*args, **kwargs) -> None:
        """Print a warning message."""
        print(BColors.WARNING + "WARN > " + BColors.ENDC + " ".join(map(str, args)), **kwargs)
        if logger is not None:
            logger.warning(" ".join(map(str, args)), **kwargs)

    @staticmethod
    def warn_verbose(*args, **kwargs) -> None:
        """Print a warning message if in verbose mode. Log in either case."""
        if verbose:
            print(
                BColors.WARNING + "WARN > " + BColors.ENDC + " ".join(map(str, args)),
                **kwargs,
            )
        if logger is not None:
            logger.warning(" ".join(map(str, args)), **kwargs)

    @staticmethod
    def error(*args, **kwargs) -> None:
        """Print an error message."""
        print(BColors.FAIL + "ERROR > " + BColors.ENDC + " ".join(map(str, args)), **kwargs)
        if logger is not None:
            logger.error(" ".join(map(str, args)), **kwargs)

    @staticmethod
    def info(*args, **kwargs) -> None:
        """Print an information message."""
        print(BColors.OKBLUE + "INFO > " + BColors.ENDC + " ".join(map(str, args)), **kwargs)
        if logger is not None:
            logger.info(" ".join(map(str, args)), **kwargs)

    @staticmethod
    def info_verbose(*args, **kwargs) -> None:
        """Print an information message if in verbose mode. Log in either case."""
        if verbose:
            print(
                BColors.OKBLUE + "INFO > " + BColors.ENDC + " ".join(map(str, args)),
                **kwargs,
            )
        if logger is not None:
            logger.info(" ".join(map(str, args)), **kwargs)

    @staticmethod
    def quit(*args, **kwargs) -> None:
        """Print a FAIL message and stop execution."""
        print("\n")
        print(BColors.FAIL + "**** " + BColors.ENDC + "Exiting.")
        print(
            BColors.FAIL
            + "**** "
            + BColors.ENDC
            + "Reason: "
            + " ".join(map(str, args)),
            **kwargs,
        )
        if logger is not None:
            logger.warning(" ".join(map(str, args)), **kwargs)
        quit()

    @staticmethod
    def banner() -> None:
        """Displays Ocean Perception banner and copyright."""
        print(" ")
        print(BColors.OKBLUE + "     **" + BColors.ENDC + " Ocean Perception")
        print(
            BColors.OKBLUE
            + "     *"
            + BColors.WARNING
            + ">"
            + BColors.ENDC
            + " University of Southampton"
        )
        print(" ")
        print(" Copyright (C) 2022 University of Southampton   ")
        print(" This program comes with ABSOLUTELY NO WARRANTY.")
        print(" This is free software, and you are welcome to  ")
        print(" redistribute it.                               ")
        print(" ")

    @staticmethod
    def get_username() -> str:
        """Returns the computer username."""
        return getpass.getuser()

    @staticmethod
    def get_hostname():
        """Return the hostname."""
        return socket.gethostname()

    @staticmethod
    def get_date():
        """Returns current date."""
        return str(datetime.datetime.now())

    @staticmethod
    def get_stamp():
        """Returns current epoch."""
        return str(datetime.datetime.now().timestamp())

    @staticmethod
    def get_version(pkg_name="uos_sess6072_build"):
        """Returns pkg_name version number."""
        return str(pkg_resources.require(pkg_name)[0].version)

    @staticmethod
    def write_metadata():
        """Writes all metadata to a string."""
        return (
            'date: "'
            + Console.get_date()
            + '" \n'
            + 'user: "'
            + Console.get_username()
            + '" \n'
            + 'host: "'
            + Console.get_hostname()
            + '" \n'
            + 'version: "'
            + Console.get_version()
            + '" \n'
        )

    @staticmethod
    def progress(
        iteration,
        total,
        prefix="Progress:",
        suffix="Complete",
        length=50,
        decimals=1,
        fill="#",
    ):
        """Call in a loop to create a progress bar in the terminal."""
        percent = ("{0:." + str(decimals) + "f}").format(
            100 * (iteration / float(total))
        )
        filled_length = int(length * iteration // total)
        bar = fill * filled_length + "-" * (length - filled_length)
        print("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix), end="\r")
        if iteration >= total - 1:
            print()

    @staticmethod
    def set_logging_file(folder_path: str, name: str = "console") -> None:
        global logger
        stamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        folder_path = Path(folder_path)
        if not folder_path.exists():
            folder_path.mkdir(parents=True)
        filename = folder_path / f"{stamp}_{name}.log"
        fh = logging.FileHandler(filename)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(formatter)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
        if logger is not None:
            logger.info("uos_sess6072_build version: " + str(Console.get_version()))

    @staticmethod
    def query_yes_no(question, default="yes"):
        """Ask a yes/no question via input() and return True/False."""
        valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
        if default is None:
            prompt = " [y/n] "
        elif default == "yes":
            prompt = " [Y/n] "
        elif default == "no":
            prompt = " [y/N] "
        else:
            raise ValueError("invalid default answer: '%s'" % default)

        while True:
            sys.stdout.write(question + prompt)
            choice = input().lower()
            if default is not None and choice == "":
                return valid[default]
            if choice in valid:
                return valid[choice]
            sys.stdout.write("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")
