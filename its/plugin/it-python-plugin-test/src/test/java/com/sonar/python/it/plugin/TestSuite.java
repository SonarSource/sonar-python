/*
 * SonarQube Python Plugin
 * Copyright (C) 2012-2024 SonarSource SA
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
package com.sonar.python.it.plugin;

import org.junit.platform.suite.api.SelectClasses;
import org.junit.platform.suite.api.Suite;

@Suite
@SelectClasses({
  BanditReportTest.class,
  CoverageTest.class,
  CPDTest.class,
  CustomRulesTest.class,
  Flake8ReportTest.class,
  SonarLintIPythonTest.class,
  MetricsTest.class,
  MypyReportTest.class,
  NoSonarTest.class,
  PylintReportTest.class,
  RuffReportTest.class,
  TestReportTest.class,
  TestRulesTest.class,
  SonarLintTest.class
})
public class TestSuite {
}
