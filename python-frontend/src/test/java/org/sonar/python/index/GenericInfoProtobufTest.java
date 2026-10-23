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
package org.sonar.python.index;

import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.sonar.plugins.python.api.types.v2.GenericInfo.BareTypeArgumentStatus;
import org.sonar.plugins.python.api.types.v2.GenericInfo;
import org.sonar.python.semantic.v2.ClassTypeBuilder;
import org.sonar.python.types.protobuf.DescriptorsProtos;
import org.sonar.python.types.protobuf.SymbolsProtos;

import static org.assertj.core.api.Assertions.assertThat;

class GenericInfoProtobufTest {
  static Stream<GenericInfo> states() {
    return Stream.of(GenericInfo.UNKNOWN, GenericInfo.NON_GENERIC,
      new GenericInfo.Generic(BareTypeArgumentStatus.UNKNOWN),
      new GenericInfo.Generic(BareTypeArgumentStatus.CONTAINS_ANY),
      new GenericInfo.Generic(BareTypeArgumentStatus.NO_ANY));
  }

  @ParameterizedTest
  @MethodSource("states")
  void structured_metadata_is_authoritative_and_round_trips(GenericInfo state) throws Exception {
    var proto = GenericInfoProtobuf.toProtobuf(state);
    // Deliberately contradict the legacy flag: only absent structured metadata should use it.
    var symbol = SymbolsProtos.ClassSymbol.newBuilder().setIsGeneric(!state.isGeneric()).setGenericInfo(proto).build();
    var descriptor = DescriptorsProtos.ClassDescriptor.newBuilder()
      .setSupportsGenerics(!state.isGeneric()).setGenericInfo(proto).build();
    assertThat(GenericInfoProtobuf.fromProtobuf(SymbolsProtos.ClassSymbol.parseFrom(symbol.toByteArray()))).isEqualTo(state);
    assertThat(GenericInfoProtobuf.fromProtobuf(DescriptorsProtos.ClassDescriptor.parseFrom(descriptor.toByteArray()))).isEqualTo(state);
  }

  @Test
  void legacy_metadata_preserves_uncertainty() {
    for (boolean generic : new boolean[]{false, true}) {
      var expected = generic ? new GenericInfo.Generic(BareTypeArgumentStatus.UNKNOWN) : GenericInfo.UNKNOWN;
      assertThat(GenericInfoProtobuf.fromProtobuf(SymbolsProtos.ClassSymbol.newBuilder().setIsGeneric(generic).build())).isEqualTo(expected);
      assertThat(GenericInfoProtobuf.fromProtobuf(DescriptorsProtos.ClassDescriptor.newBuilder().setSupportsGenerics(generic).build())).isEqualTo(expected);
      assertThat(new ClassTypeBuilder("C", "C").withIsGeneric(generic).build().genericInfo()).isEqualTo(expected);
    }
  }

  @Test
  void future_argument_status_is_unknown_but_still_generic() {
    var symbol = SymbolsProtos.ClassSymbol.newBuilder()
      .setGenericInfo(SymbolsProtos.GenericInfo.newBuilder().setGenericValue(100)).build();
    assertThat(GenericInfoProtobuf.fromProtobuf(symbol)).isEqualTo(new GenericInfo.Generic(BareTypeArgumentStatus.UNKNOWN));
  }
}
