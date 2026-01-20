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
package org.sonar.python.semantic.v2.types.typecalculator;

import java.util.List;
import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.python.semantic.v2.types.TypeInferenceMatcher;
import org.sonar.python.semantic.v2.types.TypeInferenceMatchers;
import org.sonar.python.semantic.v2.typetable.TypeTable;
import org.sonar.python.types.v2.matchers.TypePredicateContext;

/**
 * Calculates the result type of awaiting a coroutine or awaitable type.
 * When awaiting a Coroutine[T, ...], the result type is T (the first type parameter).
 */
public class AwaitedTypeCalculator {

  private static final TypeInferenceMatcher IS_COROUTINE_TYPE = TypeInferenceMatcher.of(
    TypeInferenceMatchers.isObjectOfType("typing.Coroutine"));

  private final TypePredicateContext typePredicateContext;

  public AwaitedTypeCalculator(TypeTable typeTable) {
    this.typePredicateContext = TypePredicateContext.of(typeTable);
  }

  public PythonType calculate(PythonType awaitedType) {
    if (!(awaitedType instanceof ObjectType objectType)) {
      return PythonType.UNKNOWN;
    }
    if (IS_COROUTINE_TYPE.evaluate(awaitedType, typePredicateContext) != TriBool.TRUE) {
      return PythonType.UNKNOWN;
    }
    List<PythonType> attributes = objectType.attributes();
    if (attributes.isEmpty()) {
      return PythonType.UNKNOWN;
    }
    return attributes.get(0);
  }
}

