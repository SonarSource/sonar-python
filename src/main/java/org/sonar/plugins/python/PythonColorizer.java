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

import java.util.ArrayList;
import java.util.List;

import org.sonar.api.web.CodeColorizerFormat;
import org.sonar.colorizer.KeywordsTokenizer;
import org.sonar.colorizer.StringTokenizer;
import org.sonar.colorizer.Tokenizer;

public final class PythonColorizer extends CodeColorizerFormat {

  // The keywords. Reference: keyword.kwlist of Python 2.6.6
  // self is 'quasy-keyword', the standard emacs-highlighting highlights
  // it like one
  public static final String[] KEYWORDS = { "and", "as", "assert", "break", "class", "continue", "def", "del", "elif", "else", "except",
      "exec", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda", "not", "or", "pass", "print", "raise", "return",
      "try", "while", "with", "yield", "self" };

  // The builtins. Reference: builtin__-members of Python 2.7.2
  public static final String[] BUILTINS = { "ArithmeticError", "AssertionError", "AttributeError", "BaseException", "BufferError",
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
      "issubclass", "iter", "len", "license", "list", "locals", "long", "map", "max", "memoryview", "min", "next", "object", "oct", "open",
      "ord", "pow", "print", "property", "quit", "range", "raw_input", "reduce", "reload", "repr", "reversed", "round", "set", "setattr",
      "slice", "sorted", "staticmethod", "str", "sum", "super", "tuple", "type", "unichr", "unicode", "vars", "xrange", "zip" };

  // Constants
  public static final String[] CONSTANTS = { "None", "False", "True" };

  private List<Tokenizer> tokens;

  public PythonColorizer() {
    super(Python.KEY);
  }

  public List<Tokenizer> getTokenizers() {
    if (tokens == null) {
      tokens = new ArrayList<Tokenizer>();
      tokens.add(new KeywordsTokenizer("<span class=\"k\">", "</span>", KEYWORDS));
      tokens.add(new StringTokenizer("<span class=\"s\">", "</span>"));
      tokens.add(new PythonDocTokenizer("<span class=\"cd\">", "</span>"));
      tokens.add(new PythonDocStringTokenizer("<span class=\"s\">", "</span>"));

      // the following tokenizers don't work, for some reason.
      // tokens.add(new KeywordsTokenizer("<span class=\"c\">", "</span>", CONSTANTS));
      // tokens.add(new KeywordsTokenizer("<span class=\"h\">", "</span>", BUILTINS));

      // TODO:
      // use regexptokenizer to match functions or classes
    }
    return tokens;
  }
}
