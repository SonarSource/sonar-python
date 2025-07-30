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
package org.sonar.python.checks.utils;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.project.configuration.AwsLambdaHandlerInfo;
import org.sonar.plugins.python.api.project.configuration.ProjectConfiguration;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.python.semantic.v2.callgraph.CallGraph;

class AwsLambdaChecksUtilsTest {
  @Test
  void isLambdaHandlerTest_direct() {
    var subscriptionContext = subscriptionContext(CallGraph.EMPTY);

    var functionDef = functionDef("a.b.c");
    assertThat(AwsLambdaChecksUtils.isLambdaHandler(subscriptionContext, functionDef)).isFalse();

    subscriptionContext.projectConfiguration().awsProjectConfiguration().awsLambdaHandlers()
        .add(new AwsLambdaHandlerInfo("a.b.c"));
    assertThat(AwsLambdaChecksUtils.isLambdaHandler(subscriptionContext, functionDef)).isTrue();

    var functionDefWithUnknownType = functionDef(PythonType.UNKNOWN);
    assertThat(AwsLambdaChecksUtils.isLambdaHandler(subscriptionContext, functionDefWithUnknownType)).isFalse();
  }

  @Test
  void isLambdaHandlerTest_callGraph() {
    var callGraph = new CallGraph.Builder()
        .addUsage("lambda.handler", "a.b.c")
        .addUsage("a.b.c", "e.f.g")
        .build();

    var subscriptionContext = subscriptionContext(callGraph);

    var functionDef = functionDef("e.f.g");

    assertThat(AwsLambdaChecksUtils.isLambdaHandler(subscriptionContext, functionDef)).isFalse();

    subscriptionContext.projectConfiguration().awsProjectConfiguration().awsLambdaHandlers()
        .add(new AwsLambdaHandlerInfo("lambda.handler"));
    assertThat(AwsLambdaChecksUtils.isLambdaHandler(subscriptionContext, functionDef)).isTrue();
  }

  @Test
  void isOnlyLambdaHandlerTest() {
    var callGraph = new CallGraph.Builder()
        .addUsage("lambda.handler", "a.b.c")
        .build();

    var subscriptionContext = subscriptionContext(callGraph);
    subscriptionContext.projectConfiguration().awsProjectConfiguration().awsLambdaHandlers()
        .add(new AwsLambdaHandlerInfo("lambda.handler"));

    var handlerFunction = functionDef("lambda.handler");
    assertThat(AwsLambdaChecksUtils.isOnlyLambdaHandler(subscriptionContext, handlerFunction)).isTrue();

    var calledFunction = functionDef("a.b.c");
    assertThat(AwsLambdaChecksUtils.isOnlyLambdaHandler(subscriptionContext, calledFunction)).isFalse();
    assertThat(AwsLambdaChecksUtils.isLambdaHandler(subscriptionContext, calledFunction)).isTrue();

    var unknownTypeFunction = functionDef(PythonType.UNKNOWN);
    assertThat(AwsLambdaChecksUtils.isOnlyLambdaHandler(subscriptionContext, unknownTypeFunction)).isFalse();
  }

  private static SubscriptionContext subscriptionContext(CallGraph callGraph) {
    var subscriptionContext = Mockito.mock(org.sonar.plugins.python.api.SubscriptionContext.class);
    Mockito.when(subscriptionContext.projectConfiguration()).thenReturn(new ProjectConfiguration());
    Mockito.when(subscriptionContext.callGraph()).thenReturn(callGraph);

    return subscriptionContext;
  }

  private static FunctionDef functionDef(String name) {
    FunctionType functionNameType = Mockito.mock(FunctionType.class);
    Mockito.when(functionNameType.fullyQualifiedName()).thenReturn(name);
    return functionDef(functionNameType);
  }

  private static FunctionDef functionDef(PythonType type) {
    Name functionName = Mockito.mock(Name.class);
    FunctionDef functionDef = Mockito.mock(FunctionDef.class);

    Mockito.when(functionName.typeV2()).thenReturn(type);
    Mockito.when(functionDef.name()).thenReturn(functionName);
    return functionDef;
  }
}
