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
package org.sonar.python.project.config;

import java.util.Collection;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;
import org.sonar.api.scanner.ScannerSide;
import org.sonar.plugins.python.api.project.configuration.AwsLambdaHandlerInfo;
import org.sonar.plugins.python.api.project.configuration.AwsProjectConfiguration;
import org.sonar.plugins.python.api.project.configuration.ProjectConfiguration;
import org.sonarsource.api.sonarlint.SonarLintSide;

@ScannerSide
@SonarLintSide(lifespan = SonarLintSide.MODULE)
public class ProjectConfigurationBuilder {
  private final Map<String, Set<String>> awsLambdaHandlersByPackage;

  public ProjectConfigurationBuilder() {
    awsLambdaHandlersByPackage = new ConcurrentHashMap<>();
  }

  public ProjectConfigurationBuilder addAwsLambdaHandler(String packageName, String fullyQualifiedName) {
    awsLambdaHandlersByPackage.computeIfAbsent(packageName, k -> new HashSet<>()).add(fullyQualifiedName);
    return this;
  }

  public ProjectConfigurationBuilder removePackageAwsLambdaHandlers(String packageName) {
    awsLambdaHandlersByPackage.remove(packageName);
    return this;
  }

  public ProjectConfiguration build() {
    return new ProjectConfiguration(
      new AwsProjectConfiguration(
        awsLambdaHandlersByPackage.values()
          .stream()
          .flatMap(Collection::stream)
          .map(AwsLambdaHandlerInfo::new)
          .collect(Collectors.toSet())
      )
    );
  }

}
