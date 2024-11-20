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
package org.sonar.plugins.python.api;

import com.sonar.sslr.api.RecognitionException;
import java.io.File;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.sonar.api.SonarProduct;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.PythonTestUtils;
import org.sonar.python.caching.CacheContextImpl;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.SymbolImpl;
import org.sonar.python.tree.FileInputImpl;
import org.sonar.python.tree.PythonTreeMaker;

import static org.assertj.core.api.AssertionsForInterfaceTypes.assertThat;
import static org.mockito.Mockito.mock;
import static org.sonar.python.PythonTestUtils.pythonFile;

class PythonVisitorContextTest {
  @Test
  void fullyQualifiedModuleName() {
    FileInput fileInput = PythonTestUtils.parse("def foo(): pass");

    PythonFile pythonFile = pythonFile("my_module.py");
    new PythonVisitorContext(fileInput, pythonFile, null, "my_package");
    FunctionDef functionDef = (FunctionDef) PythonTestUtils.getAllDescendant(fileInput, t -> t.is(Tree.Kind.FUNCDEF)).get(0);
    assertThat(functionDef.name().symbol().fullyQualifiedName()).isEqualTo("my_package.my_module.foo");

    // no package
    new PythonVisitorContext(fileInput, pythonFile, null, "");
    assertThat(functionDef.name().symbol().fullyQualifiedName()).isEqualTo("my_module.foo");

    // file without extension
    Mockito.when(pythonFile.fileName()).thenReturn("my_module");
    new PythonVisitorContext(fileInput, pythonFile, null, "my_package");
    functionDef = (FunctionDef) PythonTestUtils.getAllDescendant(fileInput, t -> t.is(Tree.Kind.FUNCDEF)).get(0);
    assertThat(functionDef.name().symbol().fullyQualifiedName()).isEqualTo("my_package.my_module.foo");
  }

  @Test
  void initModuleFullyQualifiedName() {
    FileInput fileInput = PythonTestUtils.parse("def fn(): pass");
    PythonFile pythonFile = pythonFile("__init__.py");
    new PythonVisitorContext(fileInput, pythonFile, null, "foo.bar");
    FunctionDef functionDef = (FunctionDef) PythonTestUtils.getAllDescendant(fileInput, t -> t.is(Tree.Kind.FUNCDEF)).get(0);
    assertThat(functionDef.name().symbol().fullyQualifiedName()).isEqualTo("foo.bar.fn");

    // no package
    new PythonVisitorContext(fileInput, pythonFile, null, "");
    assertThat(functionDef.name().symbol().fullyQualifiedName()).isEqualTo("fn");
  }

  @Test
  void globalSymbols() {
    String code = "from mod import a, b";
    FileInput fileInput = new PythonTreeMaker().fileInput(PythonParser.create().parse(code));
    PythonFile pythonFile = pythonFile("my_module.py");
    List<Symbol> modSymbols = Arrays.asList(new SymbolImpl("a", null), new SymbolImpl("b", null));
    Map<String, Set<Symbol>> globalSymbols = Collections.singletonMap("mod", new HashSet<>(modSymbols));
    new PythonVisitorContext(fileInput, pythonFile, null, "my_package", ProjectLevelSymbolTable.from(globalSymbols), null);
    assertThat(fileInput.globalVariables()).extracting(Symbol::name).containsExactlyInAnyOrder("a", "b");
  }

  @Test
  void sonar_product() {
    CacheContextImpl cacheContext = CacheContextImpl.dummyCache();
    ProjectLevelSymbolTable projectLevelSymbolTable = ProjectLevelSymbolTable.empty();
    String myPackage = "my_package";
    File workingDirectory = null;
    PythonFile pythonFile = pythonFile("my_module.py");
    FileInput fileInput = mock(FileInputImpl.class);

    PythonVisitorContext pythonVisitorContext = new PythonVisitorContext(fileInput, pythonFile, workingDirectory, myPackage, projectLevelSymbolTable, cacheContext, SonarProduct.SONARLINT);
    assertThat(pythonVisitorContext.sonarProduct()).isEqualTo(SonarProduct.SONARLINT);

    pythonVisitorContext = new PythonVisitorContext(fileInput, pythonFile, workingDirectory, myPackage, projectLevelSymbolTable, cacheContext, SonarProduct.SONARQUBE);
    assertThat(pythonVisitorContext.sonarProduct()).isEqualTo(SonarProduct.SONARQUBE);

    pythonVisitorContext = new PythonVisitorContext(fileInput, pythonFile, workingDirectory, myPackage, projectLevelSymbolTable, cacheContext);
    assertThat(pythonVisitorContext.sonarProduct()).isEqualTo(SonarProduct.SONARQUBE);

    RecognitionException parsingException = mock(RecognitionException.class);
    pythonVisitorContext = new PythonVisitorContext(pythonFile, parsingException);
    assertThat(pythonVisitorContext.sonarProduct()).isEqualTo(SonarProduct.SONARQUBE);

    pythonVisitorContext = new PythonVisitorContext(pythonFile, parsingException, SonarProduct.SONARLINT);
    assertThat(pythonVisitorContext.sonarProduct()).isEqualTo(SonarProduct.SONARLINT);
  }
}
