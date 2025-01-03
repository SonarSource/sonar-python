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
package org.sonar.plugins.python.api.tree;

import java.util.List;
import javax.annotation.CheckForNull;
import org.sonar.plugins.python.api.symbols.Symbol;

/**
 * <pre>
 * {@link #callee()} ( {@link #argumentList()} )
 * </pre>
 *
 * See https://docs.python.org/3/reference/expressions.html#calls
 */
public interface CallExpression extends Expression {
  Expression callee();

  Token leftPar();

  @CheckForNull
  ArgList argumentList();

  /**
   * Utility method to return {@code argumentList().arguments()} or an empty list when argumentList is null.
   */
  List<Argument> arguments();

  Token rightPar();

  @CheckForNull
  default Symbol calleeSymbol() {
    if (callee() instanceof HasSymbol hasSymbol) {
      return hasSymbol.symbol();
    }
    return null;
  }
}
