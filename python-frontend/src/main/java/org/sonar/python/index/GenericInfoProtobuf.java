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

import org.sonar.plugins.python.api.types.v2.GenericInfo.BareTypeArgumentStatus;
import org.sonar.plugins.python.api.types.v2.GenericInfo;
import org.sonar.python.types.protobuf.DescriptorsProtos;
import org.sonar.python.types.protobuf.SymbolsProtos;

/** Shared encoding for cached descriptors and bundled Typeshed symbols. */
public final class GenericInfoProtobuf {
  private GenericInfoProtobuf() {
  }

  public static SymbolsProtos.GenericInfo toProtobuf(GenericInfo info) {
    var builder = SymbolsProtos.GenericInfo.newBuilder();
    if (info instanceof GenericInfo.NonGeneric) {
      builder.setNonGeneric(SymbolsProtos.GenericInfo.NonGeneric.getDefaultInstance());
    } else if (info instanceof GenericInfo.Generic(var bareTypeArguments)) {
      builder.setGeneric(switch (bareTypeArguments) {
        case UNKNOWN -> SymbolsProtos.GenericInfo.BareTypeArgumentStatus.UNKNOWN;
        case CONTAINS_ANY -> SymbolsProtos.GenericInfo.BareTypeArgumentStatus.CONTAINS_ANY;
        case NO_ANY -> SymbolsProtos.GenericInfo.BareTypeArgumentStatus.NO_ANY;
      });
    }
    return builder.build();
  }

  public static GenericInfo fromProtobuf(SymbolsProtos.ClassSymbol symbol) {
    return symbol.hasGenericInfo() ? decode(symbol.getGenericInfo()) : GenericInfo.fromLegacy(symbol.getIsGeneric());
  }

  public static GenericInfo fromProtobuf(DescriptorsProtos.ClassDescriptor descriptor) {
    return descriptor.hasGenericInfo() ? decode(descriptor.getGenericInfo()) : GenericInfo.fromLegacy(descriptor.getSupportsGenerics());
  }

  private static GenericInfo decode(SymbolsProtos.GenericInfo info) {
    return switch (info.getClassificationCase()) {
      case NON_GENERIC -> GenericInfo.NON_GENERIC;
      case GENERIC -> new GenericInfo.Generic(switch (info.getGeneric()) {
        case CONTAINS_ANY -> BareTypeArgumentStatus.CONTAINS_ANY;
        case NO_ANY -> BareTypeArgumentStatus.NO_ANY;
        case UNKNOWN, UNRECOGNIZED -> BareTypeArgumentStatus.UNKNOWN;
      });
      case CLASSIFICATION_NOT_SET -> GenericInfo.UNKNOWN;
    };
  }
}
