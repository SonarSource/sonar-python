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

import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.sonar.plugins.python.api.project.configuration.AwsLambdaHandlerInfo;
import org.sonar.plugins.python.api.project.configuration.ProjectConfiguration;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.PythonType;

import static org.assertj.core.api.Assertions.assertThat;

class AwsLambdaCheckUtilsTest {


  @Test
  void isLambdaHandlerTest() {
    var projectConfiguration = new ProjectConfiguration();

    var functionNameType = Mockito.mock(FunctionType.class);
    var functionName = Mockito.mock(Name.class);
    var functionDef = Mockito.mock(FunctionDef.class);

    Mockito.when(functionNameType.fullyQualifiedName()).thenReturn("a.b.c");
    Mockito.when(functionName.typeV2()).thenReturn(functionNameType);
    Mockito.when(functionDef.name()).thenReturn(functionName);

    assertThat(AwsLambdaChecksUtils.isLambdaHandler(projectConfiguration, functionDef)).isFalse();

    projectConfiguration.awsProjectConfiguration().awsLambdaHandlers().add(new AwsLambdaHandlerInfo("a.b.c"));
    assertThat(AwsLambdaChecksUtils.isLambdaHandler(projectConfiguration, functionDef)).isTrue();

    Mockito.when(functionName.typeV2()).thenReturn(PythonType.UNKNOWN);
    assertThat(AwsLambdaChecksUtils.isLambdaHandler(projectConfiguration, functionDef)).isFalse();
  }

}
