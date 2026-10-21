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
package org.sonar.python.semantic.v2.types;

import org.sonar.plugins.python.api.symbols.v2.SymbolV2;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.types.v2.ObjectType;
import org.sonar.plugins.python.api.types.v2.PythonType;
import org.sonar.python.types.v2.TypeUtils;
import org.sonar.python.types.v2.matchers.TypePredicateContext;

public class LoopAssignment extends Assignment {
  private static final TypeInferenceMatcher IS_TUPLE = TypeInferenceMatcher.of(TypeInferenceMatchers.isObjectOfType("tuple"));
  // Match exact types: subclasses may override iteration or use a different generic parameter order.
  private static final TypeInferenceMatcher HAS_ELEMENT_ARGUMENT = TypeInferenceMatcher.of(TypeInferenceMatchers.isObjectSatisfying(
    TypeInferenceMatchers.any(
      TypeInferenceMatchers.isType("list"),
      TypeInferenceMatchers.isType("set"),
      TypeInferenceMatchers.isType("frozenset"),
      TypeInferenceMatchers.isType("dict"),
      TypeInferenceMatchers.isType("filter"),
      TypeInferenceMatchers.isType("map"),
      TypeInferenceMatchers.isType("reversed"),
      TypeInferenceMatchers.isType("collections.deque"),
      TypeInferenceMatchers.isType("collections.defaultdict"),
      TypeInferenceMatchers.isType("collections.OrderedDict"),
      TypeInferenceMatchers.isType("collections.Counter"),
      TypeInferenceMatchers.isType("collections.ChainMap"),
      TypeInferenceMatchers.isType("collections.UserList"),
      TypeInferenceMatchers.isType("collections.UserDict"),
      TypeInferenceMatchers.isType("typing.Iterable"),
      TypeInferenceMatchers.isType("typing.Reversible"),
      TypeInferenceMatchers.isType("typing.Iterator"),
      TypeInferenceMatchers.isType("typing.IO"),
      TypeInferenceMatchers.isType("typing.Generator"),
      TypeInferenceMatchers.isType("typing.Collection"),
      TypeInferenceMatchers.isType("typing.Sequence"),
      TypeInferenceMatchers.isType("typing.MutableSequence"),
      TypeInferenceMatchers.isType("typing.AbstractSet"),
      TypeInferenceMatchers.isType("typing.MutableSet"),
      TypeInferenceMatchers.isType("typing.Mapping"),
      TypeInferenceMatchers.isType("typing.MutableMapping"),
      TypeInferenceMatchers.isType("typing.KeysView"),
      TypeInferenceMatchers.isType("typing.ValuesView"))));
  private final TypePredicateContext typePredicateContext;

  public LoopAssignment(SymbolV2 lhsSymbol, Name lhsName, Expression rhs, TypePredicateContext typePredicateContext) {
    super(lhsSymbol, lhsName, rhs);
    this.typePredicateContext = typePredicateContext;
  }

  @Override
  public PythonType rhsType() {
    return TypeUtils.map(super.rhsType(), this::elementType);
  }

  private PythonType elementType(PythonType rhsType) {
    // Without element arguments there is nothing to infer; avoid resolving unrelated types from the allowlist.
    if (rhsType instanceof ObjectType objectType && !objectType.attributes().isEmpty() && objectType.hasMember("__iter__").isTrue()) {
      if (IS_TUPLE.evaluate(objectType, typePredicateContext).isTrue()) {
        // Tuple arguments describe positional elements, unlike e.g. dict[K, V], which iterates only over K.
        return objectType.attributes().stream().collect(TypeUtils.toUnionType());
      }
      if (HAS_ELEMENT_ARGUMENT.evaluate(objectType, typePredicateContext).isTrue()) {
        // Attributes describe values, including class objects stored in literal collections.
        return objectType.attributes().stream().findFirst().orElse(PythonType.UNKNOWN);
      }
    }
    return PythonType.UNKNOWN;
  }
}
