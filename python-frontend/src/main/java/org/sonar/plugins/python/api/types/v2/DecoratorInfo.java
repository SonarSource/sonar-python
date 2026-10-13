/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the Sonar Source-Available License for more details.
 *
 * You should have received a copy of the Sonar Source-Available License
 * along with this program; if not, see https://sonarsource.com/license/ssal/
 */
package org.sonar.plugins.python.api.types.v2;

import java.util.Optional;
import javax.annotation.Nullable;
import org.sonar.api.Beta;

/**
 * Semantic information about one decorator expression, without retaining its AST.
 * The expression type describes the value applied to the class. For a top-level call,
 * the callee type separately describes the callable producing that value.
 * Unknown types are retained, including unknown callees of known call expressions.
 */
@Beta
public final class DecoratorInfo {
  private final TypeWrapper expressionType;
  private final Optional<TypeWrapper> calleeType;

  public DecoratorInfo(TypeWrapper expressionType, @Nullable TypeWrapper calleeType) {
    this.expressionType = expressionType;
    this.calleeType = Optional.ofNullable(calleeType);
  }

  public DecoratorInfo(TypeWrapper expressionType) {
    this(expressionType, null);
  }

  public TypeWrapper expressionType() {
    return expressionType;
  }

  public Optional<TypeWrapper> calleeType() {
    return calleeType;
  }
}
