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
import static org.mockito.Mockito.mock;

import java.io.IOException;
import java.nio.file.Files;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.sonar.plugins.python.api.PythonFile;
import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.plugins.python.api.project.configuration.AwsLambdaHandlerInfo;
import org.sonar.plugins.python.api.project.configuration.ProjectConfiguration;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.python.semantic.v2.callgraph.CallGraph;
import org.sonar.python.tree.FileInputImpl;

class AwsLambdaChecksUtilsTest {
  @Test
  void isLambdaHandlerTest_direct() {
    var pythonVisitorContext = pythonVisitorContext(CallGraph.EMPTY);

    var functionDef = functionDef("a.b.c");
    assertThat(AwsLambdaChecksUtils.isLambdaHandler(pythonVisitorContext, functionDef)).isFalse();

    pythonVisitorContext.projectConfiguration().awsProjectConfiguration().awsLambdaHandlers().add(new AwsLambdaHandlerInfo("a.b.c"));
    assertThat(AwsLambdaChecksUtils.isLambdaHandler(pythonVisitorContext, functionDef)).isTrue();

    var functionDefWithUnknownType = functionDef(PythonType.UNKNOWN);
    assertThat(AwsLambdaChecksUtils.isLambdaHandler(pythonVisitorContext, functionDefWithUnknownType)).isFalse();
  }

  @Test
  void isLambdaHandlerTest_callGraph() {
    var callGraph = new CallGraph.Builder()
      .addUsage("lambda.handler", "a.b.c")
      .addUsage("a.b.c", "e.f.g")
      .build();

    var pythonVisitorContext = pythonVisitorContext(callGraph);

    var functionDef = functionDef("e.f.g");

    assertThat(AwsLambdaChecksUtils.isLambdaHandler(pythonVisitorContext, functionDef)).isFalse();

    pythonVisitorContext.projectConfiguration().awsProjectConfiguration().awsLambdaHandlers().add(new AwsLambdaHandlerInfo("lambda.handler"));
    assertThat(AwsLambdaChecksUtils.isLambdaHandler(pythonVisitorContext, functionDef)).isTrue();
  }

  private static PythonVisitorContext pythonVisitorContext(CallGraph callGraph) {
    PythonFile pythonFile = pythonFile("test.py");
    FileInput fileInput = mock(FileInputImpl.class);
    
    return new PythonVisitorContext.Builder(fileInput, pythonFile)
      .projectConfiguration(new ProjectConfiguration())
      .callGraph(callGraph)
      .build();
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

  private static PythonFile pythonFile(String fileName) {
    PythonFile pythonFile = Mockito.mock(PythonFile.class);
    Mockito.when(pythonFile.fileName()).thenReturn(fileName);
    try {
      Mockito.when(pythonFile.uri()).thenReturn(Files.createTempFile(fileName, "py").toUri());
    } catch (IOException e) {
      throw new IllegalStateException("Cannot create temporary file");
    }
    return pythonFile;
  }

}
