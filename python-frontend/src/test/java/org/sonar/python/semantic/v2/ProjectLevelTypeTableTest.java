/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.python.semantic.v2;

import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.types.v2.ClassType;
import org.sonar.python.types.v2.FunctionType;
import org.sonar.python.types.v2.LazyTypeWrapper;
import org.sonar.python.types.v2.ModuleType;
import org.sonar.python.types.v2.PythonType;
import org.sonar.python.types.v2.TriBool;
import org.sonar.python.types.v2.TypeChecker;
import org.sonar.python.types.v2.TypeWrapper;
import org.sonar.python.types.v2.UnknownType;

import static org.assertj.core.api.Assertions.assertThat;
import static org.sonar.python.PythonTestUtils.parseWithoutSymbols;
import static org.sonar.python.PythonTestUtils.pythonFile;
import static org.sonar.python.types.v2.TypesTestUtils.parseAndInferTypes;

class ProjectLevelTypeTableTest {
  
  @Test
  void getBuiltinTypeTest() {
    var symbolTable = ProjectLevelSymbolTable.empty();
    var table = new ProjectLevelTypeTable(symbolTable);

    var listClassType = table.getType("list");
    Assertions.assertThat(listClassType).isNotNull().isInstanceOf(ClassType.class);

    var builtinsModuleType = table.getType();
    Assertions.assertThat(builtinsModuleType).isNotNull().isInstanceOf(ModuleType.class);
    Assertions.assertThat(builtinsModuleType.resolveMember("list")).isPresent().containsSame(listClassType);

    Assertions.assertThat(table.getType("list.something")).isSameAs(PythonType.UNKNOWN);
  }

  @Test
  void getTypeshedTypeTest() {
    var symbolTable = ProjectLevelSymbolTable.empty();
    var table = new ProjectLevelTypeTable(symbolTable);

    var generatorClassType = table.getType("typing.Generator");
    Assertions.assertThat(generatorClassType).isNotNull().isInstanceOf(ClassType.class);

    var typingModuleType = table.getType("typing");
    Assertions.assertThat(typingModuleType).isNotNull().isInstanceOf(ModuleType.class);
    Assertions.assertThat(typingModuleType.resolveMember("Generator")).isPresent().containsSame(generatorClassType);
  }

  @Test
  void updateTypeTableDuringGetTypeTest() {
    var symbolTable = ProjectLevelSymbolTable.empty();
    var table = new ProjectLevelTypeTable(symbolTable);

    var rootModuleType = table.getType();
    Assertions.assertThat(rootModuleType.hasMember("typing")).isNotEqualTo(TriBool.TRUE);

    var generatorClassType = table.getType("typing.Generator");
    Assertions.assertThat(generatorClassType).isNotNull().isInstanceOf(ClassType.class);

    var typingModuleType = table.getType("typing");
    Assertions.assertThat(typingModuleType).isNotNull().isInstanceOf(ModuleType.class);
    Assertions.assertThat(typingModuleType.resolveMember("Generator")).isPresent().containsSame(generatorClassType);

    Assertions.assertThat(rootModuleType.hasMember("typing")).isEqualTo(TriBool.TRUE);
    Assertions.assertThat(rootModuleType.resolveMember("typing")).containsSame(typingModuleType);
  }

  @Test
  void nameConflictBetweenSubmoduleAndInitNameFromStubs() {
    FileInput fileInput = parseAndInferTypes("""
      import dateutil.parser
      isoparse_function = dateutil.parser.isoparse
      isoparse_function
      date = isoparse_function(date)
      date
      """);
    // We expect the "isoparse_function" to come from dateutil/parser/__init__.pyi, defined as an actual function
    FunctionType isoparseFunctionType = (FunctionType) ((ExpressionStatement) fileInput.statements().statements().get(2)).expressions().get(0).typeV2();
    assertThat(isoparseFunctionType.name()).isEqualTo("isoparse");
    assertThat(isoparseFunctionType.owner()).isNull();
    PythonType dateType = ((ExpressionStatement) fileInput.statements().statements().get(4)).expressions().get(0).typeV2();
    assertThat(dateType.unwrappedType().unwrappedType().name()).isEqualTo("datetime");
  }

  @Test
  void nameConflictBetweenSubmoduleAndInitNameFromStubs2() {
    ProjectLevelTypeTable projectLevelTypeTable = new ProjectLevelTypeTable(ProjectLevelSymbolTable.empty());
    TypeChecker typeChecker = new TypeChecker(projectLevelTypeTable);
    FileInput fileInput = parseAndInferTypes(projectLevelTypeTable, pythonFile("main.py"), """
      from dateutil.parser.isoparser import isoparser
      isoparser
      """);
    ClassType isoparserClass = (ClassType) ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2();
    // With this import syntax, we expect to retrieve the class defined in "dateutil/parser/isoparser.pyi", as "dateutil.parser.isoparser" must refer to a module
    assertThat(isoparserClass.name()).isEqualTo("isoparser");
    assertThat(typeChecker.typeCheckBuilder().isTypeOrInstanceWithName("dateutil.parser.isoparser.isoparser").check(isoparserClass)).isEqualTo(TriBool.TRUE);
    assertThat(typeChecker.typeCheckBuilder().isTypeOrInstanceWithName("dateutil.parser.isoparser").check(isoparserClass)).isEqualTo(TriBool.TRUE);

    fileInput = parseAndInferTypes(projectLevelTypeTable, pythonFile("main.py"), """
      import dateutil.parser as parser_module
      parser_module
      """);
    ModuleType parserModuleType = (ModuleType) ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2();
    PythonType isoParserMember = parserModuleType.resolveMember("isoparser").get();
    assertThat(isoParserMember).isInstanceOf(ClassType.class);
    // Should be True
    assertThat(typeChecker.typeCheckBuilder().isTypeOrInstanceWithName("dateutil.parser.isoparser.isoparser").check(isoParserMember)).isEqualTo(TriBool.UNKNOWN);

    fileInput = parseAndInferTypes(projectLevelTypeTable, pythonFile("main.py"), """
      from dateutil.parser.isoparser import isoparser
      isoparser
      """);
    PythonType isoparserClass2 = ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2();
    // We no longer resolve the class, now that the LazyType is resolved
    // SONARPY-2176: We should properly resolve either submodules or names from __init__.py depending on how the import is performed
    assertThat(isoparserClass2).isSameAs(isoparserClass);
  }

  @Test
  void importingSubmodulesTest() {
    ProjectLevelSymbolTable projectLevelSymbolTable = ProjectLevelSymbolTable.empty();

    FileInput libTree = parseWithoutSymbols(
      """
      class A: ...
      """
    );
    PythonFile pythonFile = pythonFile("lib.py");
    projectLevelSymbolTable.addModule(libTree, "my_package", pythonFile);
    ProjectLevelTypeTable projectLevelTypeTable = new ProjectLevelTypeTable(projectLevelSymbolTable);

    FileInput initTree = parseWithoutSymbols("");
    PythonFile initFile = pythonFile("__init__.py");
    projectLevelSymbolTable.addModule(initTree, "my_package", initFile);

    PythonFile mainFile = pythonFile("main.py");
    var fileInput = parseAndInferTypes(projectLevelTypeTable, mainFile, """
      from my_package import lib
      lib
      """
    );
    var libType = ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2();
    // SONARPY-2230: lib should be registered as a member/submodule of "my_package" in project table and we should fall back to it when failing to find the member declared in "my_package"
    assertThat(libType).isInstanceOf(UnknownType.UnresolvedImportType.class);
    assertThat(((UnknownType.UnresolvedImportType) libType).importPath()).isEqualTo("my_package.lib");


    mainFile = pythonFile("main.py");
    fileInput = parseAndInferTypes(projectLevelTypeTable, mainFile, """
      import my_package.lib as mylib
      mylib
      """
    );
    ModuleType moduleLibType = (ModuleType) ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2();
    assertThat(moduleLibType.name()).isEqualTo("lib");
    assertThat(moduleLibType.members().values()).extracting(TypeWrapper::type).extracting(PythonType::name).containsExactly("A");
    assertThat(moduleLibType.members().values()).extracting(TypeWrapper::type).allMatch(ClassType.class::isInstance);
  }

  @Test
  void importingRedefinedSubmodulesTest() {
    ProjectLevelSymbolTable projectLevelSymbolTable = ProjectLevelSymbolTable.empty();

    FileInput libTree = parseWithoutSymbols(
      """
      class A: ...
      """
    );
    PythonFile pythonFile = pythonFile("lib.py");
    projectLevelSymbolTable.addModule(libTree, "my_package", pythonFile);
    ProjectLevelTypeTable projectLevelTypeTable = new ProjectLevelTypeTable(projectLevelSymbolTable);

    FileInput initTree = parseWithoutSymbols("""
      from my_package.lib import A as lib
      """);
    PythonFile initFile = pythonFile("__init__.py");
    projectLevelSymbolTable.addModule(initTree, "my_package", initFile);

    PythonFile mainFile = pythonFile("main.py");
    var fileInput = parseAndInferTypes(projectLevelTypeTable, mainFile, """
      from my_package import lib
      lib
      """
    );
    var libType = ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2();
    // SONARPY-2230 lib should be resolved as the renamed class "A" here (but not present in project table at all)
    assertThat(libType).isInstanceOf(UnknownType.UnresolvedImportType.class);
    assertThat(((UnknownType.UnresolvedImportType) libType).importPath()).isEqualTo("my_package.lib");


    mainFile = pythonFile("main.py");
    fileInput = parseAndInferTypes(projectLevelTypeTable, mainFile, """
      import my_package.lib as mylib
      mylib
      """
    );
    ModuleType moduleLibType = (ModuleType) ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2();
    assertThat(moduleLibType.name()).isEqualTo("lib");
    // SONARPY-2176 lib should be resolved as the renamed class "A" here
    assertThat(moduleLibType.members().values()).extracting(TypeWrapper::type).extracting(PythonType::name).containsExactly("A");
    assertThat(moduleLibType.members().values()).extracting(TypeWrapper::type).allMatch(ClassType.class::isInstance);
  }

  @Test
  void importingRedefinedSubmodules2Test() {
    ProjectLevelSymbolTable projectLevelSymbolTable = ProjectLevelSymbolTable.empty();

    FileInput libTree = parseWithoutSymbols(
      """
      class A: ...
      """
    );
    PythonFile pythonFile = pythonFile("lib.py");
    projectLevelSymbolTable.addModule(libTree, "my_package", pythonFile);
    ProjectLevelTypeTable projectLevelTypeTable = new ProjectLevelTypeTable(projectLevelSymbolTable);

    FileInput initTree = parseWithoutSymbols("""
      class lib: ...
      """);
    PythonFile initFile = pythonFile("__init__.py");
    projectLevelSymbolTable.addModule(initTree, "my_package", initFile);

    PythonFile mainFile = pythonFile("main.py");
    var fileInput = parseAndInferTypes(projectLevelTypeTable, mainFile, """
      from my_package.lib import A
      A
      """
    );
    ClassType aType = (ClassType) ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2();
    // SONARPY-2176 lib should be resolved as the renamed class "A" here
    assertThat(aType.name()).isEqualTo("A");
  }

  @Test
  void resolveStubsWithImportedModuleVariableDescriptor() {
    var symbolTable = ProjectLevelSymbolTable.empty();
    var table = new ProjectLevelTypeTable(symbolTable);

    var nnModuleType = table.getType("torch.nn");

    Assertions.assertThat(nnModuleType).isNotNull().isInstanceOf(ModuleType.class);
  }

  @Test
  void importFunctionWithDecorators() {
    var projectLevelSymbolTable = ProjectLevelSymbolTable.empty();
    var libTree = parseWithoutSymbols(
      """
      def lib_decorator(): ...
      @lib_decorator
      def foo(): ...
      """
    );
    projectLevelSymbolTable.addModule(libTree, "", pythonFile("lib.py"));

    var projectLevelTypeTable = new ProjectLevelTypeTable(projectLevelSymbolTable);
    var mainFile = pythonFile("main.py");
    var fileInput = parseAndInferTypes(projectLevelTypeTable, mainFile, """
      from lib import foo
      foo
      """
    );
    var fooType = (FunctionType) ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2();
    var typeWrapper = (LazyTypeWrapper) fooType.decorators().get(0);
    assertThat(typeWrapper.hasImportPath("lib.lib_decorator")).isTrue();
    var decoratorType = typeWrapper.type();
    assertThat(decoratorType).isInstanceOfSatisfying(FunctionType.class, functionType -> assertThat(functionType.name()).isEqualTo("lib_decorator"));
  }

  @Test
  void importFunctionWithImportedDecorators() {
    var projectLevelSymbolTable = ProjectLevelSymbolTable.empty();
    var libTree = parseWithoutSymbols(
      """
      def lib_decorator(): ...
      """
    );
    projectLevelSymbolTable.addModule(libTree, "", pythonFile("lib.py"));
    var lib2Tree = parseWithoutSymbols(
      """
      import lib as l
      
      @l.lib_decorator
      def foo(): ...
      """
    );
    projectLevelSymbolTable.addModule(lib2Tree, "", pythonFile("lib2.py"));

    var projectLevelTypeTable = new ProjectLevelTypeTable(projectLevelSymbolTable);
    var mainFile = pythonFile("main.py");
    var fileInput = parseAndInferTypes(projectLevelTypeTable, mainFile, """
      from lib2 import foo
      foo
      """
    );
    var fooType = (FunctionType) ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2();
    var typeWrapper = (LazyTypeWrapper) fooType.decorators().get(0);
    assertThat(typeWrapper.hasImportPath("lib.lib_decorator")).isTrue();
  }
  
  @Test
  void importedFunctionDecoratorNamesTest() {
    var projectLevelSymbolTable = ProjectLevelSymbolTable.empty();
    var libTree = parseWithoutSymbols(
      """
      from abc import abstractmethod
      
      class A:
        @abstractmethod
        def foo(self): ...
      """
    );
    projectLevelSymbolTable.addModule(libTree, "", pythonFile("lib.py"));

    var projectLevelTypeTable = new ProjectLevelTypeTable(projectLevelSymbolTable);
    var mainFile = pythonFile("main.py");
    var fileInput = parseAndInferTypes(projectLevelTypeTable, mainFile, """
      from lib import A
      from datetime import tzinfo
      A.foo
      tzinfo.tzname
      """
    );
    var fooType = (FunctionType) ((ExpressionStatement) fileInput.statements().statements().get(2)).expressions().get(0).typeV2();
    var typeWrapper = (LazyTypeWrapper) fooType.decorators().get(0);
    assertThat(typeWrapper.hasImportPath("abc.abstractmethod")).isTrue();
    var tznameType = (FunctionType) ((ExpressionStatement) fileInput.statements().statements().get(3)).expressions().get(0).typeV2();
    typeWrapper = (LazyTypeWrapper) tznameType.decorators().get(0);
    // SONARPY-2300 - need to fix serializer to use fully qualified names
    assertThat(typeWrapper.hasImportPath("abstractmethod")).isTrue();
  }

  @Test
  void relativeImports() {
    var projectLevelSymbolTable = ProjectLevelSymbolTable.empty();
    FileInput initTree = parseWithoutSymbols("");
    PythonFile initFile = pythonFile("__init__.py");
    projectLevelSymbolTable.addModule(initTree, "my_package", initFile);

    var libTree = parseWithoutSymbols(
      """
      def foo(): ...
      """
    );
    projectLevelSymbolTable.addModule(libTree, "my_package", pythonFile("lib.py"));

    var projectLevelTypeTable = new ProjectLevelTypeTable(projectLevelSymbolTable);
    var mainFile = pythonFile("main.py");
    var fileInput = parseAndInferTypes(projectLevelTypeTable, mainFile, """
      from .lib import foo
      from . import lib
      foo
      lib
      """
    );
    PythonType fooType = ((ExpressionStatement) fileInput.statements().statements().get(2)).expressions().get(0).typeV2();
    assertThat(fooType).isInstanceOf(FunctionType.class);
    assertThat(fooType.name()).isEqualTo("foo");
    PythonType libType = ((ExpressionStatement) fileInput.statements().statements().get(3)).expressions().get(0).typeV2();
    assertThat(libType).isInstanceOf(ModuleType.class);
    assertThat(libType.name()).isEqualTo("lib");
  }

  @Test
  void importFunctionWithUnresolvedImportParameterTypes() {
    var projectLevelSymbolTable = ProjectLevelSymbolTable.empty();
    var libTree = parseWithoutSymbols(
      """
      def imported_function(params: list[str]): ...
      """
    );
    projectLevelSymbolTable.addModule(libTree, "", pythonFile("lib.py"));

    var projectLevelTypeTable = new ProjectLevelTypeTable(projectLevelSymbolTable);
    var mainFile = pythonFile("main.py");
    var fileInput = parseAndInferTypes(projectLevelTypeTable, mainFile, """
      from lib import imported_function
      imported_function
      """
    );
    var importedFunctionType = (FunctionType) ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2();
    var parameterType = importedFunctionType.parameters().get(0).declaredType().type().unwrappedType();
    assertThat(parameterType).isInstanceOfSatisfying(ClassType.class, parameterClassType -> assertThat(parameterClassType.name()).isEqualTo("list"));
  }

  @Test
  void classTypeAndAlias() {
    var projectLevelSymbolTable = ProjectLevelSymbolTable.empty();
    var libTree = parseWithoutSymbols(
      """
      class MyClass():
        def foo(self): ...
      my_alias = MyClass
      """
    );
    projectLevelSymbolTable.addModule(libTree, "", pythonFile("lib.py"));

    var projectLevelTypeTable = new ProjectLevelTypeTable(projectLevelSymbolTable);
    var mainFile = pythonFile("main.py");
    var fileInput = parseAndInferTypes(projectLevelTypeTable, mainFile, """
      from lib import MyClass, my_alias
      MyClass
      my_alias
      """
    );
    PythonType myClassType = ((ExpressionStatement) fileInput.statements().statements().get(1)).expressions().get(0).typeV2();
    PythonType myAliasType = ((ExpressionStatement) fileInput.statements().statements().get(2)).expressions().get(0).typeV2();
    assertThat(myClassType).isInstanceOfSatisfying(ClassType.class, ct -> {
      assertThat(ct.name()).isEqualTo("MyClass");
      assertThat(ct.fullyQualifiedName()).isEqualTo("lib.MyClass");
      }
    );
    // SONARPY-2352: my_alias should be resolved as an alias to MyClass
    assertThat(myAliasType).isInstanceOfSatisfying(ClassType.class, ct -> {
        assertThat(ct.name()).isEqualTo("my_alias");
        assertThat(ct.fullyQualifiedName()).isEqualTo("lib.my_alias");
      }
    );
    TypeChecker typeChecker = new TypeChecker(projectLevelTypeTable);
    assertThat(typeChecker.typeCheckBuilder().isInstanceOf("lib.MyClass").check(myAliasType)).isEqualTo(TriBool.UNKNOWN);
  }
}
