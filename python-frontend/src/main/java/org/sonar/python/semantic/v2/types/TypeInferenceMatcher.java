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
package org.sonar.python.semantic.v2.types;

import org.sonar.plugins.python.api.TriBool;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.python.types.v2.matchers.TypePredicate;
import org.sonar.python.types.v2.matchers.TypePredicateContext;
import org.sonar.python.types.v2.matchers.TypePredicateUtils;

public class TypeInferenceMatcher {
  private final TypePredicate predicate;

  private TypeInferenceMatcher(TypePredicate predicate) {
    this.predicate = predicate;
  }

  public TriBool evaluate(PythonType type, TypePredicateContext ctx) {
    return TypePredicateUtils.evaluate(predicate, type, ctx);
  }

  public static TypeInferenceMatcher of(TypePredicate predicate) {
    return new TypeInferenceMatcher(predicate);
  }
}
