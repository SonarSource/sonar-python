/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2022 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
package org.sonar.python.checks.cdk;

import java.util.Collection;
import java.util.Locale;
import java.util.Optional;
import java.util.function.Predicate;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

public class CdkPredicate {

  private CdkPredicate() {

  }

  /**
   * @return Predicate which tests if expression is boolean literal and is set to `false`
   */
  public static Predicate<Expression> isFalse() {
    return expression -> Optional.ofNullable(expression.firstToken()).map(Token::value).filter("False"::equals).isPresent();
  }

  /**
   * @return Predicate which tests if expression is `none`
   */
  public static Predicate<Expression> isNone() {
    return expression -> expression.is(Tree.Kind.NONE);
  }

  /**
   * @return Predicate which tests if expression is a fully qualified name (FQN) and is equal the expected FQN
   */
  public static Predicate<Expression> isFqn(String fqnValue) {
    return expression ->  Optional.ofNullable(TreeUtils.fullyQualifiedNameFromExpression(expression))
      .filter(fqnValue::equals)
      .isPresent();
  }

  /**
   * @return Predicate which tests if expression is a fully qualified name (FQN) and part of the FQN list
   */
  public static Predicate<Expression> isFqnOf(Collection<String> fqnValues) {
    return expression ->  Optional.ofNullable(TreeUtils.fullyQualifiedNameFromExpression(expression))
      .filter(fqnValues::contains)
      .isPresent();
  }

  /**
   * @return Predicate which tests if expression is a string and is equal the expected value
   */
  public static Predicate<Expression> isString(String expectedValue) {
    return expression -> CdkUtils.getString(expression).filter(expectedValue::equals).isPresent();
  }

  /**
   * @return Predicate which tests if expression is a string and starts with the expected value
   */
  public static Predicate<Expression> startsWith(String expected) {
    return expression -> CdkUtils.getString(expression).filter(str -> str.toLowerCase(Locale.ROOT).startsWith(expected)).isPresent();
  }

}
