/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.plugins.python.telemetry.collectors;

import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import org.sonar.plugins.python.api.tree.AliasedName;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.ImportName;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.dependency.model.Dependency;

/**
 * Collects top-level module names from Python import statements across all analyzed files.
 * Thread-safe: may be called from parallel file scans.
 */
public class ImportsTelemetryCollector {

  private static final Set<String> PYTHON_STDLIB_MODULES = Set.of(
    "os", "json", "datetime", "typing", "logging", "sys", "re", "time", "pathlib",
    "collections", "dataclasses", "uuid", "enum", "urllib", "functools", "io",
    "argparse", "base64", "contextlib", "asyncio", "subprocess", "hashlib",
    "tempfile", "threading", "importlib", "math", "abc", "shutil", "traceback",
    "random", "copy", "types", "csv", "concurrent", "decimal", "itertools", "inspect",
    "socket", "http", "secrets", "warnings", "string", "zipfile", "ast", "zoneinfo",
    "glob", "xml", "html", "contextvars", "signal", "hmac", "email", "operator", "ssl",
    "unicodedata", "gzip", "multiprocessing", "calendar", "textwrap", "pickle",
    "platform", "struct", "queue", "ipaddress", "statistics", "pprint", "difflib",
    "mimetypes", "gc", "binascii", "sqlite3", "smtplib", "fnmatch", "atexit",
    "configparser", "shlex", "getpass", "stat", "builtins", "zlib", "pkgutil",
    "tarfile", "ctypes", "codecs", "errno", "timeit", "locale", "tkinter",
    "tracemalloc", "socketserver", "getopt", "optparse", "filecmp", "ftplib",
    "imaplib", "xmlrpc", "wsgiref", "colorsys", "pdb", "tomllib", "pydoc", "ntpath",
    "sysconfig", "faulthandler", "site", "msvcrt", "curses", "cprofile", "-future-",
    "unittest");

  private static final Set<String> PYTHON_PYPI_MODULES = Set.of(
    "pytest", "requests", "pydantic", "boto3", "fastapi", "pandas", "numpy", "yaml",
    "dotenv", "httpx", "sqlalchemy", "starlette", "pydantic-settings", "dateutil",
    "google", "jwt", "uvicorn", "azure", "cryptography", "opentelemetry", "pytz",
    "redis", "urllib3", "flask", "pil", "pyspark", "openpyxl", "psycopg2", "jinja2",
    "aiohttp", "typing-extensions", "openai", "tenacity", "structlog", "matplotlib",
    "sentry-sdk", "setuptools", "alembic", "sklearn", "scipy", "moto", "django",
    "aws-lambda-powertools", "bs4", "click", "pymongo", "jsonschema", "loguru",
    "langchain-core", "psutil", "celery", "tqdm", "cachetools", "werkzeug",
    "prometheus-client", "rest-framework", "pyarrow", "pydantic-core", "freezegun",
    "lxml", "airflow", "rich", "snowflake", "torch", "docx", "pytest-asyncio",
    "psycopg", "jose", "pythonjsonlogger", "typer", "bson", "pyodbc", "paramiko",
    "plotly", "mcp", "apscheduler", "pendulum", "pypdf", "langgraph", "databricks",
    "polars", "msal", "faker", "langchain-openai", "googleapiclient", "orjson",
    "cv2", "ddtrace", "playwright", "fitz", "anthropic", "asyncpg", "reportlab",
    "confluent-kafka", "streamlit", "docker", "langchain-community", "deepdiff",
    "xgboost", "unidecode", "datadog", "motor", "pika", "github", "kombu", "stripe",
    "pypdf2", "hypothesis", "colorama", "firebase-admin", "qrcode",
    "prometheus-fastapi-instrumentator", "ldclient", "pycountry", "pydantic-ai",
    "flask-wtf", "pandera", "markupsafe", "croniter", "aiobotocore",
    "langchain-text-splitters", "magic", "aiokafka", "geopandas", "oracledb", "nltk",
    "onnxruntime", "psycopg-pool", "rest-framework-simplejwt", "aws-xray-sdk",
    "oauthlib", "pyotp", "tensorflow", "statsmodels", "holidays", "s3fs", "pdf2image",
    "sqlglot", "attr", "omegaconf", "pytesseract", "decouple", "gevent",
    "requests-aws4auth", "aws-lambda-typing", "fakeredis", "retry", "twilio",
    "langchain-aws", "pgvector", "chardet", "dns", "tomli", "mypy-boto3-dynamodb",
    "spacy", "requests-oauthlib", "hvac", "fpdf", "functions-framework", "aio-pika",
    "aiocache", "dj-database-url", "schedule", "ujson", "invoke", "keyring",
    "pypdfium2", "langchain-anthropic", "gnupg", "langsmith", "watchdog",
    "simple-salesforce", "simple-history", "asgi-correlation-id", "async-lru",
    "redshift-connector", "datadog-api-client", "uvloop", "markdownify", "langdetect",
    "faiss", "websocket", "pg8000", "pynamodb", "geoalchemy2", "tldextract",
    "coverage", "atlassian", "ulid", "flask-caching", "minio", "feedparser",
    "levenshtein", "mypy-boto3-secretsmanager", "jmespath", "channels", "dateparser",
    "pygments", "optuna", "dlt", "langchain-mcp-adapters", "debugpy", "numba",
    "msgpack", "pydantic-extra-types", "oauth2client", "attrs", "pysftp", "xarray",
    "chromadb", "flask-limiter", "argon2", "wtforms", "json-repair", "opencensus",
    "mongoengine", "mypy-boto3-ssm", "nh3", "slack-bolt", "llama-index", "prefect",
    "tornado", "vcr", "pdfkit", "aiosqlite", "docxtpl", "slack", "graphviz",
    "mongomock", "ray", "deepeval", "pyathena", "aws-msk-iam-sasl-signer",
    "importlib-metadata", "gcsfs", "assertpy", "docutils", "markitdown", "pip",
    "inflection", "timezonefinder", "office365", "voluptuous", "pulumi", "tzlocal",
    "rq", "valkey", "deltalake", "avro", "datadog-lambda", "prettytable",
    "webdriver-manager", "django-redis", "strenum", "allure", "colorlog", "opik",
    "nacl", "pikepdf", "cfnresponse", "flask-session", "rasterio", "sympy", "pydub",
    "pydicom", "ipywidgets", "textual", "uuid-extensions", "keras", "beanie",
    "mypy-boto3-sns", "flask-jwt-extended", "hamcrest", "python-http-client",
    "httpcore", "pillow-heif", "pyparsing", "unleashclient", "altair",
    "starlette-context", "fire", "posthog", "xlwings", "aws-requests-auth",
    "botocore", "awsglue", "mshell-python-core", "regex", "urllib2");

  private static final Set<String> WHITELIST;

  static {
    WHITELIST = new HashSet<>(PYTHON_STDLIB_MODULES);
    WHITELIST.addAll(PYTHON_PYPI_MODULES);
  }

  private final Set<String> importedModules = ConcurrentHashMap.newKeySet();

  public void collect(FileInput rootTree) {
    var visitor = new CollectorVisitor();
    rootTree.accept(visitor);
    Set<String> collected = visitor.getCollected();
    collected.retainAll(WHITELIST);
    importedModules.addAll(collected);
  }

  public ImportsTelemetry getTelemetry() {
    return new ImportsTelemetry(importedModules);
  }

  private static class CollectorVisitor extends BaseTreeVisitor {
    private final Set<String> collected = new HashSet<>();

    @Override
    public void visitImportName(ImportName importName) {
      // import X.Y.Z  ->  top-level is X
      for (AliasedName aliasedName : importName.modules()) {
        List<Name> names = aliasedName.dottedName().names();
        if (!names.isEmpty()) {
          addNormalized(names.get(0).name());
        }
      }
      super.visitImportName(importName);
    }

    @Override
    public void visitImportFrom(ImportFrom importFrom) {
      // Skip relative imports (from . import foo, from ..utils import bar)
      if (!importFrom.dottedPrefixForModule().isEmpty()) {
        super.visitImportFrom(importFrom);
        return;
      }
      // module() can be null for bare relative imports
      if (importFrom.module() == null) {
        super.visitImportFrom(importFrom);
        return;
      }
      // Collect the top-level module name — works for both regular and wildcard imports
      List<Name> names = importFrom.module().names();
      if (!names.isEmpty()) {
        addNormalized(names.get(0).name());
      }
      super.visitImportFrom(importFrom);
    }

    private void addNormalized(String rawName) {
      // Reuse Dependency normalization (lowercase + [._-]+ -> -)
      collected.add(new Dependency(rawName).name());
    }

    Set<String> getCollected() {
      return collected;
    }
  }
}
