/*
 * Sonar Python Plugin
 * Copyright (C) 2011 Waleri Enns
 * dev@sonar.codehaus.org
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02
 */

package org.sonar.plugins.python;

import org.sonar.api.resources.AbstractLanguage;

public class Python extends AbstractLanguage {
  // The keywords. Reference: keyword.kwlist of Python 2.6.5
  protected static final String[] KEYWORDS =
  {
    "and", "as", "assert", "break", "class", "continue", "def", "del", "elif", "else", "except",
    "exec", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda", "not", "or", "pass", "print", "raise", "return",
    "try", "while", "with", "yield"
  };

  // The builtins. Reference: __builtin__-members of Python 2.6.5
  protected static final String[] BUILTINS =
  {
    "ArithmeticError", "AssertionError", "AttributeError", "BaseException", "BufferError",
    "BytesWarning", "DeprecationWarning", "EOFError", "Ellipsis", "EnvironmentError", "Exception", "False", "FloatingPointError",
    "FutureWarning", "GeneratorExit", "IOError", "ImportError", "ImportWarning", "IndentationError", "IndexError", "KeyError",
    "KeyboardInterrupt", "LookupError", "MemoryError", "NameError", "None", "NotImplemented", "NotImplementedError", "OSError",
    "OverflowError", "PendingDeprecationWarning", "ReferenceError", "RuntimeError", "RuntimeWarning", "StandardError", "StopIteration",
    "SyntaxError", "SyntaxWarning", "SystemError", "SystemExit", "TabError", "True", "TypeError", "UnboundLocalError",
    "UnicodeDecodeError", "UnicodeEncodeError", "UnicodeError", "UnicodeTranslateError", "UnicodeWarning", "UserWarning", "ValueError",
    "Warning", "ZeroDivisionError", "_", "__debug__", "__doc__", "__import__", "__name__", "__package__", "abs", "all", "any", "apply",
    "basestring", "bin", "bool", "buffer", "bytearray", "bytes", "callable", "chr", "classmethod", "cmp", "coerce", "compile", "complex",
    "copyright", "credits", "delattr", "dict", "dir", "divmod", "enumerate", "eval", "execfile", "exit", "file", "filter", "float",
    "format", "frozenset", "getattr", "globals", "hasattr", "hash", "help", "hex", "id", "input", "int", "intern", "isinstance",
    "issubclass", "iter", "len", "license", "list", "locals", "long", "map", "max", "min", "next", "object", "oct", "open",
    "ord", "pow", "print", "property", "quit", "range", "raw_input", "reduce", "reload", "repr", "reversed", "round", "set", "setattr",
    "slice", "sorted", "staticmethod", "str", "sum", "super", "tuple", "type", "unichr", "unicode", "vars", "xrange", "zip"
  };

  // Some language constants suitable for highlighting
  protected static final String[] CONSTANTS = { "None", "False", "True" };

  private static final String[] SUFFIXES = { "py" };
  protected static final String KEY = "py";
  protected static final Python INSTANCE = new Python();

  public Python() {
    super(KEY, "Python");
  }

  public String[] getFileSuffixes() {
    return SUFFIXES;
  }
}
