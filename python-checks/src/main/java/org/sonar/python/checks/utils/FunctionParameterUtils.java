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
package org.sonar.python.checks.utils;

import java.util.HashSet;
import java.util.Optional;
import java.util.Set;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.types.v2.FunctionType;
import org.sonar.plugins.python.api.types.v2.ParameterV2;
import org.sonar.plugins.python.api.types.v2.PythonType;

/**
 * Utility class for extracting function parameter information.
 */
public final class FunctionParameterUtils {

  private FunctionParameterUtils() {
  }

  /**
   * Information about a function's parameters for path/URL parameter checks.
   */
  public record FunctionParameterInfo(Set<String> allParams, Set<String> positionalOnlyParams, boolean hasVariadicKeyword) {
    public static FunctionParameterInfo empty() {
      return new FunctionParameterInfo(Set.of(), Set.of(), false);
    }

    public boolean isMissingFromSignature(String param) {
      return !allParams.contains(param) && !hasVariadicKeyword;
    }
  }

  public static FunctionParameterInfo extractFunctionParameters(FunctionDef functionDef) {
    return getFunctionType(functionDef)
      .map(FunctionParameterUtils::buildParameterInfo)
      .orElse(FunctionParameterInfo.empty());
  }

  public static Optional<FunctionType> getFunctionType(FunctionDef functionDef) {
    PythonType functionType = functionDef.name().typeV2();
    if (functionType instanceof FunctionType funcType) {
      return Optional.of(funcType);
    }
    return Optional.empty();
  }

  public static FunctionParameterInfo buildParameterInfo(FunctionType functionType) {
    Set<String> allParams = new HashSet<>();
    Set<String> positionalOnlyParams = new HashSet<>();
    boolean hasVariadicKeyword = functionType.parameters().stream()
      .anyMatch(param -> param.isVariadic() && param.isKeywordVariadic());

    functionType.parameters().stream()
      .filter(param -> !param.isVariadic())
      .forEach(param -> addParameter(param, allParams, positionalOnlyParams));

    return new FunctionParameterInfo(allParams, positionalOnlyParams, hasVariadicKeyword);
  }

  private static void addParameter(ParameterV2 param, Set<String> allParams, Set<String> positionalOnlyParams) {
    String paramName = param.name();
    if (paramName != null) {
      allParams.add(paramName);
      if (param.isPositionalOnly()) {
        positionalOnlyParams.add(paramName);
      }
    }
  }
}
