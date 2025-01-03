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
package org.sonar.python.types;

import org.junit.jupiter.api.Test;
import org.sonar.python.semantic.SymbolImpl;

import static org.assertj.core.api.Assertions.assertThat;

class UnknownClassTypeTest {

  @Test
  void test() {
    var symbol = new SymbolImpl("Unknown", "in.the.middle.of.nowhere");
    var type = new UnknownClassType(symbol);

    assertThat(type.isIdentityComparableWith(new RuntimeType("SonethingElse"))).isTrue();
    assertThat(type.isIdentityComparableWith(type)).isTrue();
    assertThat(type.canHaveMember("a")).isTrue();
    assertThat(type.declaresMember("a")).isTrue();
    assertThat(type.resolveMember("a")).isPresent()
      .hasValueSatisfying(s -> "in.the.middle.of.nowhere.Unknown.a".equals(s.fullyQualifiedName()));
    assertThat(type.resolveDeclaredMember("a"))
      .hasValueSatisfying(s -> "in.the.middle.of.nowhere.Unknown.a".equals(s.fullyQualifiedName()));
    assertThat(type.canOnlyBe("a")).isFalse();
    assertThat(type.canBeOrExtend("a")).isTrue();
    assertThat(type.isCompatibleWith(type)).isTrue();
    assertThat(type.mustBeOrExtend("a")).isFalse();
    assertThat(type.typeSymbol()).isEqualTo(symbol);
  }

}
