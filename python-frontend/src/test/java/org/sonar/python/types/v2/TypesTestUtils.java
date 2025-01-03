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
package org.sonar.python.types.v2;

import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.ProjectLevelTypeTable;
import org.sonar.python.semantic.v2.SymbolTableBuilderV2;
import org.sonar.python.semantic.v2.TypeInferenceV2;

public class TypesTestUtils {

  public static final ProjectLevelTypeTable PROJECT_LEVEL_TYPE_TABLE = new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty());
  public static final PythonType BUILTINS = PROJECT_LEVEL_TYPE_TABLE.getBuiltinsModule();

  public static final PythonType INT_TYPE = BUILTINS.resolveMember("int").get();
  public static final PythonType FLOAT_TYPE = BUILTINS.resolveMember("float").get();
  public static final PythonType COMPLEX_TYPE = BUILTINS.resolveMember("complex").get();
  public static final PythonType BOOL_TYPE = BUILTINS.resolveMember("bool").get();
  public static final PythonType STR_TYPE = BUILTINS.resolveMember("str").get();
  public static final PythonType LIST_TYPE = BUILTINS.resolveMember("list").get();
  public static final PythonType TUPLE_TYPE = BUILTINS.resolveMember("tuple").get();
  public static final PythonType SET_TYPE = BUILTINS.resolveMember("set").get();
  public static final PythonType FROZENSET_TYPE = BUILTINS.resolveMember("frozenset").get();
  public static final PythonType DICT_TYPE = BUILTINS.resolveMember("dict").get();
  public static final PythonType NONE_TYPE = BUILTINS.resolveMember("NoneType").get();
  public static final PythonType TYPE_TYPE = BUILTINS.resolveMember("type").get();
  public static final PythonType EXCEPTION_TYPE = BUILTINS.resolveMember("Exception").get();

  public static FileInput parseAndInferTypes(String... code) {
    return parseAndInferTypes(PythonTestUtils.pythonFile("mod"), code);
  }

  public static FileInput parseAndInferTypes(PythonFile pythonFile, String... code) {
    return parseAndInferTypes(PROJECT_LEVEL_TYPE_TABLE, pythonFile, code);
  }

  public static FileInput parseAndInferTypes(ProjectLevelTypeTable typeTable, PythonFile pythonFile, String... code) {
    FileInput fileInput = PythonTestUtils.parseWithoutSymbols(code);
    var symbolTable = new SymbolTableBuilderV2(fileInput).build();
    new TypeInferenceV2(typeTable, pythonFile, symbolTable, "my_package").inferTypes(fileInput);
    return fileInput;
  }

  public static ClassDef lastClassDef(String... code) {
    FileInput fileInput = parseAndInferTypes(code);
    return PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Tree.Kind.CLASSDEF));
  }

  public static FunctionDef lastFunctionDef(String... code) {
    FileInput fileInput = parseAndInferTypes(code);
    return PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Tree.Kind.FUNCDEF));
  }

  public static ClassDef lastClassDefWithName(String name, String... code) {
    FileInput fileInput = parseAndInferTypes(code);
    return PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Tree.Kind.CLASSDEF) && ((ClassDef) t).name().name().equals(name));
  }

  public static Name lastName(String... code) {
    FileInput fileInput = parseAndInferTypes(code);
    Tree tree = PythonTestUtils.getLastDescendant(fileInput, t -> t.is(Tree.Kind.NAME));
    return (Name) tree;
  }

}
