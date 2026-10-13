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
package org.sonar.python;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.plugins.python.api.project.configuration.AwsLambdaHandlerInfo;
import org.sonar.plugins.python.api.project.configuration.ProjectConfiguration;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.ClassType;
import org.sonar.plugins.python.api.types.v2.FullyQualifiedNameHelper;
import org.sonar.plugins.python.api.types.v2.ModuleType;
import org.sonar.python.caching.CacheContextImpl;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.TypeInferenceV2;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockConstruction;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class TestPythonVisitorRunnerTest {

  @Test
  void file_and_mock_contexts_preserve_class_decorator_metadata() throws IOException {
    var file = new File("src/test/resources/semantic/decorator_context.py");
    var mockFile = new TestPythonVisitorRunner.MockPythonFile("", file.getName(), Files.readString(file.toPath()));
    var contexts = List.of(TestPythonVisitorRunner.createContext(file),
      TestPythonVisitorRunner.createContext(mockFile, null, "", ProjectLevelSymbolTable.empty(), CacheContextImpl.dummyCache()));

    for (var context : contexts) {
      List<ClassDef> definitions = PythonTestUtils.getAllDescendant(context.rootTree(), tree -> tree.is(Tree.Kind.CLASSDEF));
      assertThat(definitions).hasSize(2).allSatisfy(definition -> {
        var type = (ClassType) definition.name().typeV2();
        assertThat(type.decorators()).singleElement().satisfies(decorator ->
          assertThat(FullyQualifiedNameHelper.getFullyQualifiedName(decorator.expressionType().type())).contains("typing.final"));
      });
    }
  }

  @Test
  void both_context_overloads_build_v1_symbols_before_inference() throws IOException {
    var file = new File("src/test/resources/semantic/visitor_context.py");
    var mockFile = new TestPythonVisitorRunner.MockPythonFile("", file.getName(), Files.readString(file.toPath()));
    try (var inferences = mockConstruction(TypeInferenceV2.class, (inference, construction) ->
      when(inference.inferModuleType(any())).thenAnswer(invocation -> {
        FileInput root = invocation.getArgument(0);
        List<ClassDef> definitions = PythonTestUtils.getAllDescendant(root, tree -> tree.is(Tree.Kind.CLASSDEF));
        assertThat(definitions).hasSize(2).allSatisfy(definition -> assertThat(definition.name().symbol()).isNotNull());
        return mock(ModuleType.class);
      }))) {
      TestPythonVisitorRunner.createContext(file);
      TestPythonVisitorRunner.createContext(mockFile, null, "", ProjectLevelSymbolTable.empty(), CacheContextImpl.dummyCache());
      assertThat(inferences.constructed()).hasSize(2).allSatisfy(inference -> verify(inference).inferModuleType(any()));
    }
  }

  @Test
  void file_and_mock_contexts_preserve_cfgs_and_configuration() throws IOException {
    var file = new File("src/test/resources/semantic/visitor_context.py");
    var mockFile = new TestPythonVisitorRunner.MockPythonFile("", file.getName(), Files.readString(file.toPath()));
    var contexts = List.of(TestPythonVisitorRunner.createContext(file),
      TestPythonVisitorRunner.createContext(mockFile, null, "", ProjectLevelSymbolTable.empty(), CacheContextImpl.dummyCache()));

    for (var context : contexts) {
      assertThat(context.moduleType()).isNotNull();
      assertThat(context.cfg(context.rootTree())).isNotNull();
      List<ClassDef> definitions = PythonTestUtils.getAllDescendant(context.rootTree(), tree -> tree.is(Tree.Kind.CLASSDEF));
      assertThat(definitions).hasSize(2).allSatisfy(definition -> assertThat(definition.name().symbol()).isNotNull());
      FunctionDef handler = PythonTestUtils.getLastDescendant(context.rootTree(), tree -> tree.is(Tree.Kind.FUNCDEF));
      assertThat(context.cfg(handler)).isNotNull();
      assertThat(context.projectConfiguration().awsProjectConfiguration().awsLambdaHandlers())
        .containsExactly(new AwsLambdaHandlerInfo("visitor_context.lambda_handler"));
    }

    var suppliedConfiguration = new ProjectConfiguration();
    var context = TestPythonVisitorRunner.createContext(file, null, suppliedConfiguration);
    assertThat(context.projectConfiguration()).isSameAs(suppliedConfiguration);
    assertThat(suppliedConfiguration.awsProjectConfiguration().awsLambdaHandlers()).isEmpty();
  }

  @Test
  void unknownFile() {
    var file = new File("xxx");
    var check = (PythonCheck) visitorContext -> {};
    assertThatThrownBy(() -> TestPythonVisitorRunner.scanFile(file, check))
      .isInstanceOf(IllegalStateException.class);
  }

  @Test
  void fileUri() throws IOException {
    File tmpFile = Files.createTempFile("foo", ".py").toFile();
    PythonVisitorContext context = TestPythonVisitorRunner.createContext(tmpFile);
    assertThat(context.pythonFile().uri()).isEqualTo(tmpFile.toURI());
  }

  @Test
  void mockPythonFileUri() {
    var file = new TestPythonVisitorRunner.MockPythonFile("", "file.py", "print('hello')");
    PythonVisitorContext context = TestPythonVisitorRunner.createContext(file, null, "", ProjectLevelSymbolTable.empty(), CacheContextImpl.dummyCache());
    assertThat(context.pythonFile().uri()).isEqualTo(file.uri());
  }

  @Test
  void fileUriIPython() throws IOException {
    File tmpFile = Files.createTempFile("foo", ".ipynb").toFile();
    PythonVisitorContext context = TestPythonVisitorRunner.createContext(tmpFile);
    assertThat(context.pythonFile().uri()).isEqualTo(tmpFile.toURI());
  }

  @Test
  void globalSymbols() {
    File baseDir = new File("src/test/resources").getAbsoluteFile();
    List<File> files = List.of(new File(baseDir, "file.py"));
    ProjectLevelSymbolTable projectLevelSymbolTable = TestPythonVisitorRunner.globalSymbols(files, baseDir);
    assertThat(projectLevelSymbolTable.getSymbolsFromModule("file"))
      .extracting(Symbol::name)
      .containsExactlyInAnyOrder("hello", "A");
  }

  @Test
  void globalSymbols2() {
    ProjectLevelSymbolTable projectLevelSymbolTable = TestPythonVisitorRunner.globalSymbols(Map.of("src/test/resources/file.py", """
      def hello(): ...
      class A: ...
      """), "src/test/resources");
    assertThat(projectLevelSymbolTable.getSymbolsFromModule("file"))
      .extracting(Symbol::name)
      .containsExactlyInAnyOrder("hello", "A");
  }

  @Test
  void globalSymbolsIPython() {
    File baseDir = new File("src/test/resources").getAbsoluteFile();
    List<File> files = List.of(new File(baseDir, "file.py"), new File(baseDir, "file.ipynb"));
    ProjectLevelSymbolTable projectLevelSymbolTable = TestPythonVisitorRunner.globalSymbols(files, baseDir);
    assertThat(projectLevelSymbolTable.getSymbolsFromModule("file"))
      .extracting(Symbol::name)
      .containsExactlyInAnyOrder("hello", "A");
  }

  @Test
  void scanNotebookFile() throws IOException {
    File tmpFile = Files.createTempFile("foo", ".ipynb").toFile();
    var check = (PythonCheck) visitorContect -> {};
    PythonVisitorContext context = TestPythonVisitorRunner.scanNotebookFile(tmpFile, Map.of(), "", check);
    assertThat(context.pythonFile().uri()).isEqualTo(tmpFile.toURI());
  }

}
