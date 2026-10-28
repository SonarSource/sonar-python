#
# SonarQube Python Plugin
# Copyright (C) 2011-2024 SonarSource SA
# mailto:info AT sonarsource DOT com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3 of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#

import atomicwrites
import backports
import backports_abc
import certifi
import characteristic
import click
import contextvar
import cryptography
import dataclasses
import emoji
import enum34
import fastapi
import fb303
import filelock
import first
import flask
import flask_cors
import freezegun
import frozendict
import futures
import geoip2
import ipaddress
import itsdangerous
import Jinja2
import kazoo
import MarkupSafe
import maxminddb
import orjson
import pandas
import pathlib
import PyJWT
import pymssql
import pyspark
import requests
import Routes
import scribes
import termcolor
import tornado
import WerkZeug
import xxhash
import yaml


# Replacement typing sources: import public submodules explicitly.
from PIL import (
    Image, ImageOps, ImageDraw, ImageFont, ImageChops, ImageEnhance,
    ImageCms, ImageColor, ImageGrab, ImageMorph, ImageStat, ImageTk, ImageTransform,
)
import sqlalchemy
import sqlalchemy.engine.url
import sqlalchemy.orm
import babel
import babel.core
import passlib.hash
import passlib.context
import passlib.handlers.sha2_crypt
import passlib.handlers.pbkdf2

# Flask-SQLAlchemy 3.x uses SQLAlchemy 2.0 inline typing.
import flask_sqlalchemy
import sqlalchemy.ext.asyncio
import sqlalchemy.dialects.postgresql
import sqlalchemy.dialects.mysql
import sqlalchemy.dialects.sqlite
import sqlalchemy.inspection
import babel.dates
import babel.numbers
import babel.support
import babel.messages.catalog

# Typed pyOpenSSL replaces the removed Typeshed package.
from OpenSSL import SSL, crypto

# These public modules are no longer transitively imported by cryptography 50.
import cryptography.hazmat.primitives.kdf.scrypt
import cryptography.hazmat.primitives.serialization.pkcs12
import cryptography.hazmat.primitives.serialization.pkcs7

# Libraries that now publish their own typing instead of Typeshed distributions.
import stripe
import influxdb_client
import paho.mqtt.client
import chardet
import chardet.universaldetector
import dj_database_url
import invoke
import slugify
import tzlocal
import ujson

import annoy
import exifread
import flake8_plugin_utils
import fpdf
import pyrfc3339
import datemath
import pyVmomi
import pyVim.connect
import stdlib_list
import tree_sitter
