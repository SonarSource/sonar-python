/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
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
import java.util.Collections;
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
import org.sonar.python.index.Descriptor;
import org.sonar.python.index.VariableDescriptor;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.ProjectLevelTypeTable;
import org.sonar.python.semantic.v2.callgraph.CallGraph;
import org.sonar.python.semantic.v2.callgraph.CallGraphNode;
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
    var ctx = new PythonVisitorContext.Builder(fileInput, pythonFile).packageName("my_package").build();
    FunctionDef functionDef = (FunctionDef) PythonTestUtils.getAllDescendant(fileInput, t -> t.is(Tree.Kind.FUNCDEF)).get(0);
    assertThat(functionDef.name().symbol().fullyQualifiedName()).isEqualTo("my_package.my_module.foo");
    assertThat(ctx.moduleType()).isNotNull()
      .hasFieldOrPropertyWithValue("fullyQualifiedName", "my_package.my_module");

    // no package
    ctx = new PythonVisitorContext.Builder(fileInput, pythonFile).build();
    assertThat(functionDef.name().symbol().fullyQualifiedName()).isEqualTo("my_module.foo");
    assertThat(ctx.moduleType()).isNotNull()
      .hasFieldOrPropertyWithValue("fullyQualifiedName", "my_module");

    // file without extension
    Mockito.when(pythonFile.fileName()).thenReturn("my_module");
    ctx = new PythonVisitorContext.Builder(fileInput, pythonFile).packageName("my_package").build();
    functionDef = (FunctionDef) PythonTestUtils.getAllDescendant(fileInput, t -> t.is(Tree.Kind.FUNCDEF)).get(0);
    assertThat(functionDef.name().symbol().fullyQualifiedName()).isEqualTo("my_package.my_module.foo");
    assertThat(ctx.moduleType()).isNotNull()
      .hasFieldOrPropertyWithValue("fullyQualifiedName", "my_package.my_module");
  }

  @Test
  void initModuleFullyQualifiedName() {
    FileInput fileInput = PythonTestUtils.parse("def fn(): pass");
    PythonFile pythonFile = pythonFile("__init__.py");
    new PythonVisitorContext.Builder(fileInput, pythonFile).packageName("foo.bar").build();
    FunctionDef functionDef = (FunctionDef) PythonTestUtils.getAllDescendant(fileInput, t -> t.is(Tree.Kind.FUNCDEF)).get(0);
    assertThat(functionDef.name().symbol().fullyQualifiedName()).isEqualTo("foo.bar.fn");

    // no package
    new PythonVisitorContext.Builder(fileInput, pythonFile).build();
    assertThat(functionDef.name().symbol().fullyQualifiedName()).isEqualTo("fn");
  }

  @Test
  void globalSymbols() {
    String code = "from mod import a, b";
    FileInput fileInput = new PythonTreeMaker().fileInput(PythonParser.create().parse(code));
    PythonFile pythonFile = pythonFile("my_module.py");

    Set<Descriptor> descriptors = Set.of(new VariableDescriptor("a", "mod.a", null), new VariableDescriptor("b", "mod.b", null));
    Map<String, Set<Descriptor>> globalDescriptors = Collections.singletonMap("mod", descriptors);

    new PythonVisitorContext.Builder(fileInput, pythonFile)
      .packageName("my_package")
      .projectLevelSymbolTable(ProjectLevelSymbolTable.from(globalDescriptors))
      .cacheContext(CacheContextImpl.dummyCache())
      .build();
    assertThat(fileInput.globalVariables()).extracting(Symbol::name).containsExactlyInAnyOrder("a", "b");
  }

  @Test
  void callGraph() {
    String code = """
        def bar():
          pass

        def foo():
          bar()
        """;
    FileInput fileInput = new PythonTreeMaker().fileInput(PythonParser.create().parse(code));
    PythonFile pythonFile = pythonFile("my_module.py");

    var ctx = new PythonVisitorContext.Builder(fileInput, pythonFile)
      .packageName("my_package")
      .cacheContext(CacheContextImpl.dummyCache())
      .build();

    CallGraph callGraph = ctx.callGraph();

    assertThat(callGraph.getUsages("my_package.my_module.foo")).isEmpty();
    assertThat(callGraph.getUsages("my_package.my_module.bar")).extracting(CallGraphNode::fqn)
      .containsExactly("my_package.my_module.foo");

  }

  @Test
  void sonar_product() {
    CacheContextImpl cacheContext = CacheContextImpl.dummyCache();
    ProjectLevelSymbolTable projectLevelSymbolTable = ProjectLevelSymbolTable.empty();
    var projectLevelTypeTable = new ProjectLevelTypeTable(projectLevelSymbolTable);
    String myPackage = "my_package";
    PythonFile pythonFile = pythonFile("my_module.py");
    FileInput fileInput = mock(FileInputImpl.class);

    PythonVisitorContext pythonVisitorContext = new PythonVisitorContext.Builder(fileInput, pythonFile)
      .packageName(myPackage)
      .projectLevelSymbolTable(projectLevelSymbolTable)
      .typeTable(projectLevelTypeTable)
      .cacheContext(cacheContext)
      .sonarProduct(SonarProduct.SONARLINT)
      .build();
    assertThat(pythonVisitorContext.sonarProduct()).isEqualTo(SonarProduct.SONARLINT);

    pythonVisitorContext = new PythonVisitorContext.Builder(fileInput, pythonFile)
      .packageName(myPackage)
      .projectLevelSymbolTable(projectLevelSymbolTable)
      .typeTable(projectLevelTypeTable)
      .cacheContext(cacheContext)
      .sonarProduct(SonarProduct.SONARQUBE)
      .build();
    assertThat(pythonVisitorContext.sonarProduct()).isEqualTo(SonarProduct.SONARQUBE);

    pythonVisitorContext = new PythonVisitorContext.Builder(fileInput, pythonFile)
      .packageName(myPackage)
      .projectLevelSymbolTable(projectLevelSymbolTable)
      .typeTable(projectLevelTypeTable)
      .cacheContext(cacheContext)
      .build();
    assertThat(pythonVisitorContext.sonarProduct()).isEqualTo(SonarProduct.SONARQUBE);

    RecognitionException parsingException = mock(RecognitionException.class);
    pythonVisitorContext = new PythonVisitorContext(pythonFile, parsingException, SonarProduct.SONARQUBE);
    assertThat(pythonVisitorContext.sonarProduct()).isEqualTo(SonarProduct.SONARQUBE);

    pythonVisitorContext = new PythonVisitorContext(pythonFile, parsingException, SonarProduct.SONARLINT);
    assertThat(pythonVisitorContext.sonarProduct()).isEqualTo(SonarProduct.SONARLINT);

    pythonVisitorContext = new PythonVisitorContext.Builder(fileInput, pythonFile)
      .packageName(myPackage)
      .projectLevelSymbolTable(projectLevelSymbolTable)
      .cacheContext(cacheContext)
      .sonarProduct(SonarProduct.SONARLINT)
      .build();
    assertThat(pythonVisitorContext.sonarProduct()).isEqualTo(SonarProduct.SONARLINT);
  }
}
