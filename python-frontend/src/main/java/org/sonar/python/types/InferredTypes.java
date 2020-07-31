/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2020 SonarSource SA
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
package org.sonar.python.types;

import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.symbols.AmbiguousSymbol;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.plugins.python.api.tree.TypeAnnotation;
import org.sonar.plugins.python.api.types.BuiltinTypes;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.tree.TreeUtils;

public class InferredTypes {

  private static final Map<String, String> ALIASED_ANNOTATIONS = new HashMap<>();
  static {
    ALIASED_ANNOTATIONS.put("typing.List", BuiltinTypes.LIST);
    ALIASED_ANNOTATIONS.put("typing.Tuple", BuiltinTypes.TUPLE);
    ALIASED_ANNOTATIONS.put("typing.Dict", BuiltinTypes.DICT);
    ALIASED_ANNOTATIONS.put("typing.Set", BuiltinTypes.SET);
    ALIASED_ANNOTATIONS.put("typing.FrozenSet", "frozenset");
    ALIASED_ANNOTATIONS.put("typing.Deque", "deque");
    ALIASED_ANNOTATIONS.put("typing.DefaultDict", "defaultdict");
    ALIASED_ANNOTATIONS.put("typing.Type", "type");
  }

  public static final InferredType INT = runtimeBuiltinType(BuiltinTypes.INT);
  public static final InferredType FLOAT = runtimeBuiltinType(BuiltinTypes.FLOAT);
  public static final InferredType COMPLEX = runtimeBuiltinType(BuiltinTypes.COMPLEX);

  public static final InferredType STR = runtimeBuiltinType(BuiltinTypes.STR);

  public static final InferredType SET = runtimeBuiltinType(BuiltinTypes.SET);
  public static final InferredType DICT = runtimeBuiltinType(BuiltinTypes.DICT);
  public static final InferredType LIST = runtimeBuiltinType(BuiltinTypes.LIST);
  public static final InferredType TUPLE = runtimeBuiltinType(BuiltinTypes.TUPLE);

  public static final InferredType NONE = runtimeBuiltinType(BuiltinTypes.NONE_TYPE);

  public static final InferredType BOOL = runtimeBuiltinType(BuiltinTypes.BOOL);

  private static Map<String, Symbol> builtinSymbols;

  private InferredTypes() {
  }

  public static boolean isInitialized() {
    return  builtinSymbols != null;
  }

  public static InferredType anyType() {
    return AnyType.ANY;
  }

  private static InferredType runtimeBuiltinType(String fullyQualifiedName) {
    return new RuntimeType(TypeShed.typeShedClass(fullyQualifiedName));
  }

  public static InferredType runtimeType(@Nullable Symbol typeClass) {
    if (typeClass instanceof ClassSymbol) {
      return new RuntimeType((ClassSymbol) typeClass);
    }
    if (typeClass instanceof AmbiguousSymbol) {
      return union(((AmbiguousSymbol) typeClass).alternatives().stream().map(InferredTypes::runtimeType));
    }
    return anyType();
  }

  public static InferredType declaredType(@Nullable Symbol typeClass) {
    if (typeClass instanceof ClassSymbol) {
      return new DeclaredType((ClassSymbol) typeClass);
    }
    if (typeClass instanceof AmbiguousSymbol) {
      return union(((AmbiguousSymbol) typeClass).alternatives().stream().map(InferredTypes::declaredType));
    }
    return anyType();
  }



  static void setBuiltinSymbols(Map<String, Symbol> builtinSymbols) {
    InferredTypes.builtinSymbols = Collections.unmodifiableMap(builtinSymbols);
  }

  public static InferredType or(InferredType t1, InferredType t2) {
    return UnionType.or(t1, t2);
  }

  public static InferredType union(Stream<InferredType> types) {
    return types.reduce(InferredTypes::or).orElse(anyType());
  }

  public static InferredType fromTypeAnnotation(TypeAnnotation typeAnnotation, boolean isFromTypeshed) {
    if (builtinSymbols != null) {
      return fromTypeAnnotation(typeAnnotation.expression(), builtinSymbols, isFromTypeshed);
    } else {
      return fromTypeAnnotation(typeAnnotation.expression(), Collections.emptyMap(), isFromTypeshed);
    }
  }

  private static InferredType fromTypeAnnotation(Expression expression, Map<String, Symbol> builtinSymbols, boolean isFromTypeshed) {
    if (expression.is(Kind.NAME) && !((Name) expression).name().equals("Any")) {
      Symbol symbol = ((Name) expression).symbol();
      if (symbol != null && "typing.Text".equals(symbol.fullyQualifiedName())) {
        return InferredTypes.fromTypeAnnotation(builtinSymbols.get("str"), isFromTypeshed);
      }
      return InferredTypes.fromTypeAnnotation(symbol, isFromTypeshed);
    }
    if (expression.is(Kind.SUBSCRIPTION)) {
      SubscriptionExpression subscription = (SubscriptionExpression) expression;
      return TreeUtils.getSymbolFromTree(subscription.object())
        .map(symbol -> InferredTypes.genericType(symbol, subscription.subscripts().expressions(), builtinSymbols, isFromTypeshed))
        .orElse(InferredTypes.anyType());
    }
    return InferredTypes.anyType();
  }

  private static InferredType fromTypeAnnotation(@Nullable Symbol typeClass, boolean isFromTypeshed) {
    return isFromTypeshed ? InferredTypes.runtimeType(typeClass) : InferredTypes.declaredType(typeClass);
  }

  private static InferredType genericType(Symbol symbol, List<Expression> subscripts, Map<String, Symbol> builtinSymbols, boolean isFromTypeshed) {
    String builtinFqn = ALIASED_ANNOTATIONS.get(symbol.fullyQualifiedName());
    if (builtinFqn == null) {
      if ("typing.Optional".equals(symbol.fullyQualifiedName()) && subscripts.size() == 1) {
        InferredType noneType = InferredTypes.fromTypeAnnotation(builtinSymbols.get(BuiltinTypes.NONE_TYPE), isFromTypeshed);
        return InferredTypes.or(fromTypeAnnotation(subscripts.get(0), builtinSymbols, isFromTypeshed), noneType);
      }
      if ("typing.Union".equals(symbol.fullyQualifiedName())) {
        return union(subscripts.stream().map(s -> fromTypeAnnotation(s, builtinSymbols, isFromTypeshed)));
      }
      return InferredTypes.fromTypeAnnotation(symbol, isFromTypeshed);
    }
    return InferredTypes.fromTypeAnnotation(builtinSymbols.get(builtinFqn), isFromTypeshed);
  }

  public static Collection<ClassSymbol> typeSymbols(InferredType inferredType) {
    if (inferredType instanceof RuntimeType) {
      return Collections.singleton(((RuntimeType) inferredType).getTypeClass());
    }
    if (inferredType instanceof DeclaredType) {
      return Collections.singleton(((DeclaredType) inferredType).getTypeClass());
    }
    if (inferredType instanceof UnionType) {
      Set<ClassSymbol> typeClasses = new HashSet<>();
      ((UnionType) inferredType).types().forEach(type -> typeClasses.addAll(typeSymbols(type)));
      return typeClasses;
    }
    return Collections.emptySet();
  }

  @CheckForNull
  public static String typeName(InferredType inferredType) {
    Collection<ClassSymbol> typeClasses = typeSymbols(inferredType);
    if (typeClasses.size() == 1) {
      return typeClasses.iterator().next().name();
    }
    return null;
  }

  @CheckForNull
  public static LocationInFile typeClassLocation(InferredType inferredType) {
    Collection<ClassSymbol> typeClasses = typeSymbols(inferredType);
    if (typeClasses.size() == 1) {
      return typeClasses.iterator().next().definitionLocation();
    }
    return null;
  }
}
