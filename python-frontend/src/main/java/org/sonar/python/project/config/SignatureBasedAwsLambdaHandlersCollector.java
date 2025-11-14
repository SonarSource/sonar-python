/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2025 SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.python.project.config;

import java.util.Objects;
import java.util.Set;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.python.tree.TreeUtils;

public class SignatureBasedAwsLambdaHandlersCollector {

  public static final String AWS_LAMBDA_HANDLER_NAME_PATTERN = ".*(?:(_handler)|(Handler))$";
  public static final Set<String> EVENT_PARAM_ACCEPTABLE_NAMES = Set.of("event");
  public static final Set<String> CONTEXT_PARAM_ACCEPTABLE_NAMES = Set.of("ctx", "context");

  public void collect(ProjectConfigurationBuilder configBuilder, Tree rootTree, String packageName) {
    var visitor = new CollectorVisitor(configBuilder, packageName);
    rootTree.accept(visitor);
  }

  private static class CollectorVisitor extends BaseTreeVisitor {

    private final ProjectConfigurationBuilder configBuilder;
    private final String packageName;

    private CollectorVisitor(ProjectConfigurationBuilder configBuilder, String packageName) {
      this.configBuilder = configBuilder;
      this.packageName = packageName;
    }

    @Override
    public void visitFunctionDef(FunctionDef functionDef) {
      var name = functionDef.name().name();
      if (!name.matches(AWS_LAMBDA_HANDLER_NAME_PATTERN)) {
        return;
      }
      var parameters = functionDef.parameters();
      if (parameters == null) {
        return;
      }

      var parameterNames = parameters.all().stream()
        .flatMap(TreeUtils.toStreamInstanceOfMapper(Parameter.class))
        .map(Parameter::name)
        .filter(Objects::nonNull)
        .map(Name::name)
        .toList();

      if (functionDef.name().typeV2() instanceof FunctionType functionType
          && parameterNames.size() == 2
          && EVENT_PARAM_ACCEPTABLE_NAMES.contains(parameterNames.get(0))
          && CONTEXT_PARAM_ACCEPTABLE_NAMES.contains(parameterNames.get(1))
      ) {
        var fullyQualifiedName = functionType.fullyQualifiedName();
        configBuilder.addAwsLambdaHandler(packageName, fullyQualifiedName);
      }
    }
  }


}
