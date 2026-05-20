# -*- coding: utf-8 -*-
"""
Copyright (c) 2023 The uos_sess6072_build Authors.
All rights reserved.
Licensed under the BSD 3-Clause License.
See LICENSE.md file in the project root for full license information.
"""

import asyncio
import os

__version__ = "1.2.1"

if os.name == "nt":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
