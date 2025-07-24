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
package org.sonar.python;

import java.util.Collections;
import java.util.concurrent.CountDownLatch;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.sonar.plugins.python.api.project.configuration.AwsLambdaHandlerInfo;
import org.sonar.plugins.python.api.project.configuration.ProjectConfiguration;
import org.sonar.plugins.python.api.ProjectPythonVersion;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.plugins.python.api.PythonVisitorContext.Builder;
import org.sonar.plugins.python.api.caching.CacheContext;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.regex.RegexContext;
import org.sonar.python.semantic.ProjectLevelSymbolTable;
import org.sonar.python.semantic.v2.callgraph.CallGraph;
import org.sonar.plugins.python.api.types.v2.TriBool;
import org.sonarsource.analyzer.commons.regex.RegexParseResult;
import org.sonarsource.analyzer.commons.regex.ast.FlagSet;

import static org.assertj.core.api.Assertions.assertThat;

class SubscriptionVisitorTest {

  @Test
  void test_regex_cache() {
    PythonSubscriptionCheck check = new PythonSubscriptionCheck() {
      @Override
      public void initialize(Context context) {
        context.registerSyntaxNodeConsumer(Tree.Kind.STRING_ELEMENT, ctx -> {
          StringElement stringElement = (StringElement)ctx.syntaxNode();
          RegexContext regexCtx = (RegexContext) ctx;
          RegexParseResult resultWithNoFlags = regexCtx.regexForStringElement(stringElement, new FlagSet());
          RegexParseResult resultWithFlags = regexCtx.regexForStringElement(stringElement, new FlagSet(Pattern.MULTILINE));

          assertThat(resultWithNoFlags).isNotSameAs(resultWithFlags);
          // When we retrieve them again, it will be the same instance retrieved from the cache.
          assertThat(resultWithNoFlags).isSameAs(regexCtx.regexForStringElement(stringElement, new FlagSet()));
          assertThat(resultWithFlags).isSameAs(regexCtx.regexForStringElement(stringElement, new FlagSet(Pattern.MULTILINE)));
        });
      }
    };

    FileInput fileInput = PythonTestUtils.parse("'.*'");
    PythonVisitorContext context = new PythonVisitorContext.Builder(fileInput, PythonTestUtils.pythonFile("file")).build();
    SubscriptionVisitor.analyze(Collections.singleton(check), context);
  }

  @Test
  void exposed_visitor_data() {
    FileInput fileInput = PythonTestUtils.parse("def foo(): ...");
    var cache = Mockito.mock(CacheContext.class);

    PythonFile pythonFile = PythonTestUtils.pythonFile("file");
    PythonVisitorContext context = new PythonVisitorContext.Builder(fileInput, pythonFile)
      .projectLevelSymbolTable(ProjectLevelSymbolTable.empty())
      .cacheContext(cache)
      .build();

    PythonSubscriptionCheck check = new PythonSubscriptionCheck() {
      @Override
      public void initialize(Context context) {
        context.registerSyntaxNodeConsumer(Tree.Kind.FILE_INPUT, ctx -> {
          assertThat(ctx.cacheContext()).isSameAs(cache);
          assertThat(ctx.pythonFile()).isEqualTo(pythonFile);
          assertThat(ctx.sourcePythonVersions()).isEqualTo(ProjectPythonVersion.currentVersions());
        });
      }
    };
    SubscriptionVisitor.analyze(Collections.singleton(check), context);
  }

  @Test
  void typeChecker() {
    PythonSubscriptionCheck check = new PythonSubscriptionCheck() {
      @Override
      public void initialize(Context context) {
        context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, ctx -> {
          ClassDef classDef = (ClassDef) ctx.syntaxNode();
          assertThat(ctx.typeChecker().typeCheckBuilder().instancesHaveMember("foo").check(classDef.name().typeV2())).isEqualTo(TriBool.TRUE);
        });
      }
    };

    FileInput fileInput = PythonTestUtils.parse("class A:\n  def foo(self): ...");
    PythonVisitorContext context = new PythonVisitorContext.Builder(fileInput, PythonTestUtils.pythonFile("file")).build();
    SubscriptionVisitor.analyze(Collections.singleton(check), context);
  }

  @Test
  void projectConfiguration() {
    var projectConfiguration = new ProjectConfiguration();
    projectConfiguration.awsProjectConfiguration().awsLambdaHandlers().add(new AwsLambdaHandlerInfo("a.b.c"));

    var latch = new CountDownLatch(1);
    var check = new PythonSubscriptionCheck() {
      @Override
      public void initialize(Context context) {
        context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, ctx -> {
          assertThat(ctx.projectConfiguration()).isSameAs(projectConfiguration);
          assertThat(ctx.projectConfiguration().awsProjectConfiguration().awsLambdaHandlers())
            .contains(new AwsLambdaHandlerInfo("a.b.c"));
          latch.countDown();
        });
      }
    };

    var fileInput = PythonTestUtils.parse("class my_handler: ...");

    var context = new Builder(fileInput, PythonTestUtils.pythonFile("file"))
      .projectConfiguration(projectConfiguration)
      .workingDirectory(null)
      .packageName("")
      .build();

    SubscriptionVisitor.analyze(Collections.singleton(check), context);
    assertThat(latch.getCount())
      .withFailMessage("CallGraph was not accessed")
      .isZero();
  }

  @Test
  void callGraph() {
    var callGraph = Mockito.mock(CallGraph.class);
    var latch = new CountDownLatch(1);
    var check = new PythonSubscriptionCheck() {
      @Override
      public void initialize(Context context) {
        context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, ctx -> {
          assertThat(ctx.callGraph()).isSameAs(callGraph);
          latch.countDown();
        });
      }
    };

    var fileInput = PythonTestUtils.parse("class my_handler: ...");
    var context = new Builder(fileInput, PythonTestUtils.pythonFile("file"))
      .callGraph(callGraph)
      .build();

    SubscriptionVisitor.analyze(Collections.singleton(check), context);
    assertThat(latch.getCount())
      .withFailMessage("CallGraph was not accessed")
      .isZero();
  }
}
