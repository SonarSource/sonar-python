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

import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.project.configuration.AwsLambdaHandlerInfo;
import org.sonar.python.types.v2.TypesTestUtils;

class SignatureBasedAwsLambdaHandlersCollectorTest {


  @Test
  void test() {
    var fileInput = TypesTestUtils.parseAndInferTypes("""
      def my_handler(event, context):
        ...
      
      def myHandler(event, context):
        ...
      
      def my2Handler(event, ctx):
        ...
      
      def foo(event, context):
        ...
      
      def h1_handler(x, context):
        ...
      
      def h2_handler(event, x):
        ...
      
      def h3_handler(event, context, x):
        ...
      
      def h4_handler():
        ...
      """);

    var collector = new SignatureBasedAwsLambdaHandlersCollector();
    var configBuilder = new ProjectConfigurationBuilder();
    collector.collect(configBuilder, fileInput, "my_package.mod");

    var config = configBuilder.build();
    Assertions.assertThat(config.awsProjectConfiguration().awsLambdaHandlers())
      .extracting(AwsLambdaHandlerInfo::fullyQualifiedName)
      .containsOnly(
        "my_package.mod.my_handler",
        "my_package.mod.myHandler",
        "my_package.mod.my2Handler"
      );
  }
}
