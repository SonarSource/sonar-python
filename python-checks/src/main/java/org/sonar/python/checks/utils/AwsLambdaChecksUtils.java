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

import org.sonar.plugins.python.api.PythonVisitorContext;
import org.sonar.plugins.python.api.project.configuration.ProjectConfiguration;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.TriBool;
import org.sonar.python.semantic.v2.callgraph.CallGraph;
import org.sonar.python.semantic.v2.callgraph.CallGraphWalker;

public class AwsLambdaChecksUtils {

  private AwsLambdaChecksUtils() {
  }

  public static boolean isLambdaHandler(PythonVisitorContext ctx, FunctionDef functionDef) {
    if (functionDef.name().typeV2() instanceof FunctionType functionType) {
      String fqn = functionType.fullyQualifiedName();
      return isLambdaHandlerFqn(ctx.projectConfiguration(), fqn) 
        || isFqnCalledFromLambdaHandler(ctx.callGraph(), ctx.projectConfiguration(), fqn);
    }
    return false;
  }

  private static boolean isLambdaHandlerFqn(ProjectConfiguration projectConfiguration, String fqn) {
    return projectConfiguration.awsProjectConfiguration()
             .awsLambdaHandlers()
             .stream()
             .anyMatch(handler -> handler.fullyQualifiedName().equals(fqn));
  }

  private static boolean isFqnCalledFromLambdaHandler(CallGraph callGraph, ProjectConfiguration projectConfiguration, String fqn) {
    return new CallGraphWalker(callGraph)
      .isUsedFrom(fqn, node -> isLambdaHandlerFqn(projectConfiguration, node.fqn()))
      .isTrue();
  }
}
