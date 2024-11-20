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
package org.sonar.python;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.PythonCheck.PreciseIssue;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.PythonInputFileContext;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.PythonVisitorCheck;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.plugins.python.api.SubscriptionCheck;
import org.sonar.plugins.python.api.caching.CacheContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.caching.CacheContextImpl;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.types.TypeShed;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;

class PythonVisitorCheckTest {

  private static final File FILE = new File("src/test/resources/file.py");
  public static final String MESSAGE = "message";

  private static List<PreciseIssue> scanFileForIssues(File file, PythonCheck check) {
    PythonVisitorContext context = TestPythonVisitorRunner.createContext(file);
    check.scanFile(context);
    return context.getIssues();
  }

  @Test
  void test() {
    TestPythonCheck check = new TestPythonCheck (){
      @Override
      public void visitFunctionDef(FunctionDef pyFunctionDefTree) {
        super.visitFunctionDef(pyFunctionDefTree);
        Name name = pyFunctionDefTree.name();
        addIssue(name, name.firstToken().value());
      }
    };

    List<PreciseIssue> issues = scanFileForIssues(FILE, check);

    assertThat(issues).hasSize(2);
    PreciseIssue firstIssue = issues.get(0);

    assertThat(firstIssue.cost()).isNull();
    assertThat(firstIssue.secondaryLocations()).isEmpty();

    IssueLocation primaryLocation = firstIssue.primaryLocation();
    assertThat(primaryLocation.message()).isEqualTo("hello");

    assertThat(primaryLocation.startLine()).isEqualTo(1);
    assertThat(primaryLocation.endLine()).isEqualTo(1);
    assertThat(primaryLocation.startLineOffset()).isEqualTo(4);
    assertThat(primaryLocation.endLineOffset()).isEqualTo(9);
  }

  @Test
  void test_cost() {
    TestPythonCheck check = new TestPythonCheck (){
      @Override
      public void visitFunctionDef(FunctionDef pyFunctionDefTree) {
        super.visitFunctionDef(pyFunctionDefTree);
        Name name = pyFunctionDefTree.name();
        addIssue(name.firstToken(), MESSAGE).withCost(42);
      }
    };

    List<PreciseIssue> issues = scanFileForIssues(FILE, check);
    PreciseIssue firstIssue = issues.get(0);
    assertThat(firstIssue.cost()).isEqualTo(42);
  }

  @Test
  void working_directory() throws IOException {
    Path workDir = Files.createTempDirectory("workDir");
    PythonSubscriptionCheck check = new PythonSubscriptionCheck() {
      @Override
      public void initialize(SubscriptionCheck.Context context) {
        context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> assertThat(ctx.workingDirectory()).isEqualTo(workDir.toFile()));
      }
    };
    File tmpFile = Files.createTempFile("foo", "py").toFile();
    PythonVisitorContext context = TestPythonVisitorRunner.createContext(tmpFile, workDir.toFile());
    assertThat(context.workingDirectory()).isEqualTo(workDir.toFile());
    assertThat(context.pythonFile().uri()).isEqualTo(tmpFile.toURI());
    check.scanFile(context);
    SubscriptionVisitor.analyze(Collections.singletonList(check), context);
  }

  @Test
  void working_directory_null() throws IOException {
    PythonSubscriptionCheck check = new PythonSubscriptionCheck() {
      @Override
      public void initialize(SubscriptionCheck.Context context) {
        context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> assertThat(ctx.workingDirectory()).isNull());
      }
    };
    File tmpFile = Files.createTempFile("foo", "py").toFile();
    PythonVisitorContext context = TestPythonVisitorRunner.createContext(tmpFile, null);
    assertThat(context.workingDirectory()).isNull();
    assertThat(context.pythonFile().uri()).isEqualTo(tmpFile.toURI());
    check.scanFile(context);
    SubscriptionVisitor.analyze(Collections.singletonList(check), context);
  }

  @Test
  void cache_context() throws IOException {
    PythonSubscriptionCheck check = new PythonSubscriptionCheck() {
      @Override
      public void initialize(SubscriptionCheck.Context context) {
        context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> assertThat(ctx.workingDirectory()).isNull());
      }
    };
    File tmpFile = Files.createTempFile("foo", "py").toFile();

    var cache = mock(CacheContext.class);

    PythonVisitorContext context = TestPythonVisitorRunner.createContext(tmpFile, null, "", ProjectLevelSymbolTable.empty(), cache);
    assertThat(context.workingDirectory()).isNull();
    assertThat(context.pythonFile().uri()).isEqualTo(tmpFile.toURI());
    assertThat(context.pythonFile().key()).isEqualTo(tmpFile.getPath());
    assertThat(context.cacheContext()).isSameAs(cache);

    check.scanFile(context);
    SubscriptionVisitor.analyze(Collections.singletonList(check), context);
  }

  @Test
  void stubFilesSymbols() {
    PythonVisitorContext context = TestPythonVisitorRunner.createContext(FILE);

    SymbolsRecordingCheck check = new SymbolsRecordingCheck();
    check.scanFile(context);
    SubscriptionVisitor.analyze(Collections.singletonList(check), context);

    assertThat(check.symbols).isEqualTo(TypeShed.stubFilesSymbols());

    PythonInputFileContext inputFileContext = new PythonInputFileContext(mock(PythonFile.class), null, CacheContextImpl.dummyCache());
    assertThat(inputFileContext.stubFilesSymbols()).isEqualTo(TypeShed.stubFilesSymbols());
  }

  private static class TestPythonCheck extends PythonVisitorCheck {

  }

  private static class SymbolsRecordingCheck extends PythonSubscriptionCheck {
    public Collection<Symbol> symbols;
    @Override
    public void initialize(Context context) {
      context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> symbols = ctx.stubFilesSymbols());
    }
  }
}
