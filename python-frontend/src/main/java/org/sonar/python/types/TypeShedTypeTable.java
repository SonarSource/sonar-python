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

import java.util.Collections;
import java.util.List;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.python.types.protobuf.SymbolsProtos;

/** Resolves 1-based type IDs from one lazily loaded typeshed module. */
public final class TypeShedTypeTable {

  public static final TypeShedTypeTable EMPTY = new TypeShedTypeTable(Collections.emptyList());

  private final List<SymbolsProtos.Type> types;

  private TypeShedTypeTable(List<SymbolsProtos.Type> types) {
    this.types = types;
  }

  public static TypeShedTypeTable from(SymbolsProtos.ModuleSymbol module) {
    return module.getTypeTableCount() == 0 ? EMPTY : new TypeShedTypeTable(module.getTypeTableList());
  }

  @CheckForNull
  public SymbolsProtos.Type resolve(int typeId) {
    if (typeId == 0 || this == EMPTY) {
      return null;
    }
    if (typeId < 0 || typeId > types.size()) {
      throw new IllegalArgumentException("Invalid typeshed type ID: " + typeId);
    }
    return types.get(typeId - 1);
  }

  @CheckForNull
  // This flag centralizes compatibility with legacy protobufs that embed types.
  @SuppressWarnings("java:S2301")
  public SymbolsProtos.Type resolve(int typeId, boolean hasEmbeddedType, @Nullable SymbolsProtos.Type embeddedType) {
    if (typeId != 0) {
      return resolve(typeId);
    }
    return hasEmbeddedType ? embeddedType : null;
  }

  public List<SymbolsProtos.Type> arguments(SymbolsProtos.Type type) {
    if (type.getArgTypeIdsCount() == 0) {
      return type.getArgsList();
    }
    return type.getArgTypeIdsList().stream().map(this::resolve).toList();
  }

  public SymbolsProtos.Type argument(SymbolsProtos.Type type, int index) {
    if (type.getArgTypeIdsCount() == 0) {
      return type.getArgs(index);
    }
    return resolve(type.getArgTypeIds(index));
  }
}
