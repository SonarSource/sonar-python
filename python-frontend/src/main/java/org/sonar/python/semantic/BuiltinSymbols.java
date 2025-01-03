/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource SA.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.semantic;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

public class BuiltinSymbols {

  private BuiltinSymbols() {
    // empty constructor
  }


  public static final String CLASS_METHOD_DECORATOR = "classmethod";
  public static final String STATIC_METHOD_DECORATOR = "staticmethod";

  /**
   * See https://docs.python.org/3/library/constants.html#built-in-consts
   */
  public static final Set<String> CONSTANTS = Collections.unmodifiableSet(new HashSet<>(Arrays.asList(
    "True", "False", "None", "NotImplemented", "__debug__", "copyright", "credits", "license", "quit", "exit", "Ellipsis")));

  /**
   * See https://docs.python.org/3/reference/import.html?highlight=__package__#import-related-module-attributes
   */
  public static final Set<String> MODULE_ATTRIBUTES = Collections.unmodifiableSet(new HashSet<>(Arrays.asList(
    "__name__", "__loader__", "__package__", "__spec__", "__path__", "__file__", "__cached__", "__doc__", "__builtins__")));

  /**
   * See https://docs.python.org/3/library/functions.html
   */
  public static final Set<String> FUNCTIONS = Collections.unmodifiableSet(new HashSet<>(Arrays.asList(
    "abs",
    "delattr",
    "hash",
    "memoryview",
    "set",
    "all",
    "dict",
    "help",
    "min",
    "setattr",
    "any",
    "dir",
    "hex",
    "next",
    "slice",
    "ascii",
    "divmod",
    "id",
    "object",
    "sorted",
    "bin",
    "enumerate",
    "input",
    "oct",
    STATIC_METHOD_DECORATOR,
    "bool",
    "eval",
    "int",
    "open",
    "str",
    "breakpoint",
    "exec",
    "isinstance",
    "ord",
    "sum",
    "bytearray",
    "filter",
    "issubclass",
    "pow",
    "super",
    "bytes",
    "float",
    "iter",
    "print",
    "tuple",
    "callable",
    "format",
    "len",
    "property",
    "type",
    "chr",
    "frozenset",
    "list",
    "range",
    "vars",
    CLASS_METHOD_DECORATOR,
    "getattr",
    "locals",
    "repr",
    "zip",
    "compile",
    "globals",
    "map",
    "reversed",
    "__import__",
    "complex",
    "hasattr",
    "max",
    "round")));

  /**
   * See https://docs.python.org/2.7/library/functions.html
   */
  public static final Set<String> FUNCTIONS_PYTHON2 = Collections.unmodifiableSet(new HashSet<>(Arrays.asList(
    "abs",
    "divmod",
    "input",
    "open",
    STATIC_METHOD_DECORATOR,
    "all",
    "enumerate",
    "int",
    "ord",
    "str",
    "any",
    "eval",
    "isinstance",
    "pow",
    "sum",
    "basestring",
    "execfile",
    "issubclass",
    "print",
    "super2",
    "bin",
    "file",
    "iter",
    "property",
    "tuple",
    "bool",
    "filter",
    "len",
    "range",
    "type",
    "bytearray",
    "float",
    "list",
    "raw_input",
    "unichr",
    "callable",
    "format",
    "locals",
    "reduce",
    "unicode",
    "chr",
    "frozenset",
    "long",
    "reload",
    "vars",
    CLASS_METHOD_DECORATOR,
    "getattr",
    "map",
    "repr",
    "xrange",
    "cmp",
    "globals",
    "max",
    "reversed",
    "zip",
    "compile",
    "hasattr",
    "memoryview",
    "round",
    "__import__",
    "complex",
    "hash",
    "min",
    "set",
    "delattr",
    "help",
    "next",
    "setattr",
    "dict",
    "hex",
    "object",
    "slice",
    "dir",
    "id",
    "oct",
    "sorted",
    "apply",
    "buffer",
    "coerce",
    "intern"
  )));

  /**
   * See https://docs.python.org/3/library/exceptions.html
   */
  public static final Set<String> EXCEPTIONS = Collections.unmodifiableSet(new HashSet<>(Arrays.asList(
    "BaseException",
    "SystemExit",
    "KeyboardInterrupt",
    "GeneratorExit",
    "Exception",
    "StopIteration",
    "StopAsyncIteration",
    "ArithmeticError",
    "FloatingPointError",
    "OverflowError",
    "ZeroDivisionError",
    "AssertionError",
    "AttributeError",
    "BufferError",
    "EOFError",
    "ImportError",
    "ModuleNotFoundError",
    "LookupError",
    "IndexError",
    "KeyError",
    "MemoryError",
    "NameError",
    "UnboundLocalError",
    "OSError",
    "BlockingIOError",
    "ChildProcessError",
    "ConnectionError",
    "BrokenPipeError",
    "ConnectionAbortedError",
    "ConnectionRefusedError",
    "ConnectionResetError",
    "FileExistsError",
    "FileNotFoundError",
    "InterruptedError",
    "IsADirectoryError",
    "NotADirectoryError",
    "PermissionError",
    "ProcessLookupError",
    "TimeoutError",
    "ReferenceError",
    "RuntimeError",
    "NotImplementedError",
    "RecursionError",
    "SyntaxError",
    "IndentationError",
    "TabError",
    "SystemError",
    "TypeError",
    "ValueError",
    "UnicodeError",
    "UnicodeDecodeError",
    "UnicodeEncodeError",
    "UnicodeTranslateError",
    "Warning",
    "DeprecationWarning",
    "PendingDeprecationWarning",
    "RuntimeWarning",
    "SyntaxWarning",
    "UserWarning",
    "FutureWarning",
    "ImportWarning",
    "UnicodeWarning",
    "BytesWarning",
    "ResourceWarning"
  )));

  /**
   * See https://docs.python.org/2.7/library/exceptions.html
   */
  public static final Set<String> EXCEPTIONS_PYTHON2 = Collections.unmodifiableSet(new HashSet<>(Arrays.asList(
    "BaseException",
    "SystemExit",
    "KeyboardInterrupt",
    "GeneratorExit",
    "Exception",
    "StopIteration",
    "StandardError",
    "BufferError",
    "ArithmeticError",
    "FloatingPointError",
    "OverflowError",
    "ZeroDivisionError",
    "AssertionError",
    "AttributeError",
    "EnvironmentError",
    "IOError",
    "OSError",
    "WindowsError",
    "VMSError",
    "EOFError",
    "ImportError",
    "LookupError",
    "IndexError",
    "KeyError",
    "MemoryError",
    "NameError",
    "UnboundLocalError",
    "ReferenceError",
    "RuntimeError",
    "NotImplementedError",
    "SyntaxError",
    "IndentationError",
    "TabError",
    "SystemError",
    "TypeError",
    "ValueError",
    "UnicodeError",
    "UnicodeDecodeError",
    "UnicodeEncodeError",
    "UnicodeTranslateError",
    "Warning",
    "DeprecationWarning",
    "PendingDeprecationWarning",
    "RuntimeWarning",
    "SyntaxWarning",
    "UserWarning",
    "FutureWarning",
    "ImportWarning",
    "UnicodeWarning",
    "BytesWarning"
  )));

  public static Set<String> all() {
    Set<String> all = new HashSet<>();
    all.addAll(BuiltinSymbols.CONSTANTS);
    all.addAll(BuiltinSymbols.FUNCTIONS);
    all.addAll(BuiltinSymbols.FUNCTIONS_PYTHON2);
    all.addAll(BuiltinSymbols.EXCEPTIONS);
    all.addAll(BuiltinSymbols.EXCEPTIONS_PYTHON2);
    all.addAll(BuiltinSymbols.MODULE_ATTRIBUTES);
    return all;
  }

  public static final Set<String> STATIC_AND_CLASS_METHOD_DECORATORS = Set.of(CLASS_METHOD_DECORATOR, STATIC_METHOD_DECORATOR);
}
