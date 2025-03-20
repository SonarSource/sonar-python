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
package com.sonarsource.architecture.python;

import java.util.HashSet;
import java.util.Set;

public class PythonRules {
  public static Set<String> ruleKeys = new HashSet<>();

  public static boolean throwOnCall = false;

  public static Set<String> getRuleKeys() {
    if (throwOnCall) {
      throw new RuntimeException("Boom!");
    }
    return ruleKeys;
  }

  public static String getRepositoryKey() {
    return "pythonarchitecture";
  }

}
