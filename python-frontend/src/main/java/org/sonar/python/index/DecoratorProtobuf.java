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

import java.util.List;
import java.util.Optional;
import org.sonar.python.index.DecoratorDescriptor.TypeReference;
import org.sonar.python.types.protobuf.DescriptorsProtos;
import org.sonar.python.types.protobuf.SymbolsProtos;

/** Shared encoding for cached descriptors and bundled Typeshed symbols. */
public final class DecoratorProtobuf {
  private DecoratorProtobuf() {}

  public static SymbolsProtos.DecoratorInfo toProtobuf(DecoratorDescriptor decorator) {
    var builder = SymbolsProtos.DecoratorInfo.newBuilder()
      .addAllExpressionTypes(decorator.expressionTypes().stream().map(DecoratorProtobuf::toProtobuf).toList());
    decorator.calleeTypes().ifPresent(types -> builder.setHasCallee(true)
      .addAllCalleeTypes(types.stream().map(DecoratorProtobuf::toProtobuf).toList()));
    return builder.build();
  }

  private static SymbolsProtos.DecoratorTypeReference toProtobuf(TypeReference reference) {
    var builder = SymbolsProtos.DecoratorTypeReference.newBuilder().setIsInstance(reference.isInstance());
    if (reference.fullyQualifiedName() != null) {
      builder.setFullyQualifiedName(reference.fullyQualifiedName());
    }
    return builder.build();
  }

  private static DecoratorDescriptor fromProtobuf(SymbolsProtos.DecoratorInfo decorator) {
    return new DecoratorDescriptor(decorator.getExpressionTypesList().stream().map(DecoratorProtobuf::fromProtobuf).toList(),
      decorator.getHasCallee() ? Optional.of(decorator.getCalleeTypesList().stream().map(DecoratorProtobuf::fromProtobuf).toList()) : Optional.empty());
  }

  private static TypeReference fromProtobuf(SymbolsProtos.DecoratorTypeReference reference) {
    return new TypeReference(reference.hasFullyQualifiedName() ? reference.getFullyQualifiedName() : null,
      reference.getIsInstance());
  }

  public static List<DecoratorDescriptor> fromProtobuf(SymbolsProtos.ClassSymbol symbol) {
    return decode(symbol.getDecoratorsList());
  }

  public static List<DecoratorDescriptor> fromProtobuf(DescriptorsProtos.ClassDescriptor descriptor) {
    return decode(descriptor.getDecoratorsList());
  }

  private static List<DecoratorDescriptor> decode(List<SymbolsProtos.DecoratorInfo> decorators) {
    return decorators.stream().map(DecoratorProtobuf::fromProtobuf).toList();
  }
}
