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
package org.sonar.python.types;

import org.junit.jupiter.api.Test;
import org.sonar.python.types.protobuf.SymbolsProtos;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class TypeShedTypeTableTest {

  private static final TypeShedTypeTable TABLE = TypeShedTypeTable.from(SymbolsProtos.ModuleSymbol.newBuilder()
    .addTypeTable(SymbolsProtos.Type.newBuilder().setFullyQualifiedName("builtins.str"))
    .build());

  @Test
  void rejectsOutOfRangeIds() {
    assertThatThrownBy(() -> TABLE.resolve(2))
      .isInstanceOf(IllegalArgumentException.class)
      .hasMessage("Invalid typeshed type ID: 2");
    assertThatThrownBy(() -> TABLE.resolve(-1))
      .isInstanceOf(IllegalArgumentException.class)
      .hasMessage("Invalid typeshed type ID: -1");
  }

  @Test
  void emptyTableDegradesUnknownTypeIdsToNoType() {
    assertThat(TypeShedTypeTable.EMPTY.resolve(1)).isNull();
    assertThat(TypeShedTypeTable.EMPTY.resolve(-1)).isNull();
  }
}
