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
package com.sonarsource.plugins.dbd.api;

import java.util.HashSet;
import java.util.Set;

/**
 * Class required to test SonarWay for DBD rules
 */
public class PythonRules {
  public static Set<String> ruleKeys = new HashSet<>();

  public static boolean throwOnCall = false;

  public static Set<String> getDataflowBugDetectionRuleKeys() {
    if (throwOnCall) {
      throw new RuntimeException("Boom!");
    }
    return ruleKeys;
  }

  public static String getRepositoryKey() {
    return "dbd-repo-key";
  }

  public static Set<String> methodThrowingException() throws Exception {
    throw new RuntimeException("testing");
  }
}
