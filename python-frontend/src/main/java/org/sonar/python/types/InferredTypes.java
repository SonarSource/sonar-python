/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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

import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.LocationInFile;
import org.sonar.plugins.python.api.symbols.AmbiguousSymbol;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.BinaryExpression;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.plugins.python.api.tree.TypeAnnotation;
import org.sonar.plugins.python.api.types.BuiltinTypes;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.semantic.SymbolImpl;
import org.sonar.python.tree.TreeUtils;
import org.sonar.python.types.protobuf.SymbolsProtos;

import static org.sonar.plugins.python.api.symbols.Symbol.Kind.CLASS;

public class InferredTypes {

  private static final Map<String, String> ALIASED_ANNOTATIONS = new HashMap<>();

  static {
    ALIASED_ANNOTATIONS.put("typing.List", BuiltinTypes.LIST);
    ALIASED_ANNOTATIONS.put("typing.Tuple", BuiltinTypes.TUPLE);
    ALIASED_ANNOTATIONS.put("typing.Dict", BuiltinTypes.DICT);
    ALIASED_ANNOTATIONS.put("typing.Set", BuiltinTypes.SET);
    ALIASED_ANNOTATIONS.put("typing.FrozenSet", "frozenset");
    ALIASED_ANNOTATIONS.put("typing.Type", "type");
  }

  public static final InferredType INT = runtimeBuiltinType(BuiltinTypes.INT);
  public static final InferredType DECL_INT = declaredBuiltinType(BuiltinTypes.INT);
  public static final InferredType FLOAT = runtimeBuiltinType(BuiltinTypes.FLOAT);
  public static final InferredType DECL_FLOAT = declaredBuiltinType(BuiltinTypes.FLOAT);
  public static final InferredType COMPLEX = runtimeBuiltinType(BuiltinTypes.COMPLEX);
  public static final InferredType DECL_COMPLEX = declaredBuiltinType(BuiltinTypes.COMPLEX);

  public static final InferredType STR = runtimeBuiltinType(BuiltinTypes.STR);
  public static final InferredType DECL_STR = declaredBuiltinType(BuiltinTypes.STR);

  public static final InferredType SET = runtimeBuiltinType(BuiltinTypes.SET);
  public static final InferredType DECL_SET = declaredBuiltinType(BuiltinTypes.SET);
  public static final InferredType DICT = runtimeBuiltinType(BuiltinTypes.DICT);
  public static final InferredType DECL_DICT = declaredBuiltinType(BuiltinTypes.DICT);
  public static final InferredType LIST = runtimeBuiltinType(BuiltinTypes.LIST);
  public static final InferredType DECL_LIST = declaredBuiltinType(BuiltinTypes.LIST);
  public static final InferredType TUPLE = runtimeBuiltinType(BuiltinTypes.TUPLE);
  public static final InferredType DECL_TUPLE = declaredBuiltinType(BuiltinTypes.TUPLE);

  public static final InferredType NONE = runtimeBuiltinType(BuiltinTypes.NONE_TYPE);
  public static final InferredType DECL_NONE = declaredBuiltinType(BuiltinTypes.NONE_TYPE);

  public static final InferredType BOOL = runtimeBuiltinType(BuiltinTypes.BOOL);
  public static final InferredType DECL_BOOL = declaredBuiltinType(BuiltinTypes.BOOL);

  public static final InferredType TYPE = runtimeBuiltinType("type");
  public static final InferredType DECL_TYPE = declaredBuiltinType("type");

  private static final String UNICODE = "unicode";
  private static final String BYTES = "bytes";
  // https://github.com/python/mypy/blob/e97377c454a1d5c019e9c56871d5f229db6b47b2/mypy/semanal_classprop.py#L16-L46
  private static final Map<String, Set<String>> HARDCODED_COMPATIBLE_TYPES = new HashMap<>();

  private static final Set<Set<String>> HARDCODED_INCOMPATIBLE_TYPES = Set.of(Set.of(BuiltinTypes.TUPLE, BuiltinTypes.LIST));

  static {
    HARDCODED_COMPATIBLE_TYPES.put(BuiltinTypes.INT, new HashSet<>(Arrays.asList(BuiltinTypes.FLOAT, BuiltinTypes.COMPLEX)));
    HARDCODED_COMPATIBLE_TYPES.put(BuiltinTypes.FLOAT, new HashSet<>(List.of(BuiltinTypes.COMPLEX)));
    HARDCODED_COMPATIBLE_TYPES.put(BuiltinTypes.BOOL, new HashSet<>(List.of(BuiltinTypes.FLOAT)));
    HARDCODED_COMPATIBLE_TYPES.put("bytearray", new HashSet<>(Arrays.asList(BYTES, BuiltinTypes.STR, UNICODE)));
    HARDCODED_COMPATIBLE_TYPES.put("memoryview", new HashSet<>(Arrays.asList(BYTES, BuiltinTypes.STR, UNICODE)));
    // str <=> bytes equivalence only for Python2
    HARDCODED_COMPATIBLE_TYPES.put(BuiltinTypes.STR, new HashSet<>(Arrays.asList(UNICODE, BYTES)));
    HARDCODED_COMPATIBLE_TYPES.put(BYTES, new HashSet<>(Collections.singletonList(BuiltinTypes.STR)));
    // TODO SONARPY-1340: This is a workaround to avoid FPs with TypedDict, however, this produces false-negatives. We should have a more
    //  fine-grained solution to check dictionaries against TypedDict.
    HARDCODED_COMPATIBLE_TYPES.put(BuiltinTypes.DICT, Set.of("typing.TypedDict"));
  }

  protected static final Map<String, String> BUILTINS_TYPE_CATEGORY = new HashMap<>();
  private static final String NUMBER = "number";

  static {
    BUILTINS_TYPE_CATEGORY.put(BuiltinTypes.STR, BuiltinTypes.STR);
    BUILTINS_TYPE_CATEGORY.put(BuiltinTypes.INT, NUMBER);
    BUILTINS_TYPE_CATEGORY.put(BuiltinTypes.FLOAT, NUMBER);
    BUILTINS_TYPE_CATEGORY.put(BuiltinTypes.COMPLEX, NUMBER);
    BUILTINS_TYPE_CATEGORY.put(BuiltinTypes.BOOL, NUMBER);
    BUILTINS_TYPE_CATEGORY.put(BuiltinTypes.LIST, BuiltinTypes.LIST);
    BUILTINS_TYPE_CATEGORY.put(BuiltinTypes.SET, BuiltinTypes.SET);
    BUILTINS_TYPE_CATEGORY.put("frozenset", BuiltinTypes.SET);
    BUILTINS_TYPE_CATEGORY.put(BuiltinTypes.DICT, BuiltinTypes.DICT);
    BUILTINS_TYPE_CATEGORY.put(BuiltinTypes.TUPLE, BuiltinTypes.TUPLE);
  }

  private static final SymbolImpl OPTIONAL_SYMBOL = new SymbolImpl("Optional", "typing.Optional");

  private InferredTypes() {
  }

  public static InferredType anyType() {
    return AnyType.ANY;
  }

  static InferredType runtimeBuiltinType(String fullyQualifiedName) {
    return new RuntimeType(fullyQualifiedName);
  }

  private static InferredType declaredBuiltinType(String fullyQualifiedName) {
    return new DeclaredType(fullyQualifiedName);
  }

  public static InferredType runtimeType(@Nullable Symbol typeClass) {
    if (typeClass instanceof ClassSymbol classSymbol) {
      return new RuntimeType(classSymbol);
    }
    if (typeClass instanceof AmbiguousSymbol ambiguousSymbol) {
      return union(ambiguousSymbol.alternatives().stream().map(InferredTypes::runtimeType));
    }
    return anyType();
  }

  /**
   * Assumes unknown imported symbols whose name start with a capital letter to be classes.
   * @param symbol
   * @return {@link UnknownClassType} or {@link AnyType}
   */
  public static InferredType anyOrUnknownClassType(Symbol symbol) {
    return Optional.of(symbol)
      .filter(s -> Character.isUpperCase(s.name().charAt(0)))
      .filter(s -> {
        var usageKinds = s.usages().stream().map(Usage::kind).collect(Collectors.toSet());
        return usageKinds.contains(Usage.Kind.IMPORT) && !usageKinds.contains(Usage.Kind.ASSIGNMENT_LHS);
      })
      .filter(SymbolImpl.class::isInstance)
      .map(SymbolImpl.class::cast)
      .map(unknownClassSymbol -> {
        if (anyType().equals(unknownClassSymbol.inferredType())) {
          unknownClassSymbol.setInferredType(new UnknownClassType(unknownClassSymbol));
        }
        return unknownClassSymbol.inferredType();
      }).orElseGet(InferredTypes::anyType);
  }

  public static InferredType or(InferredType t1, InferredType t2) {
    return UnionType.or(t1, t2);
  }

  public static InferredType union(Stream<InferredType> types) {
    return types.reduce(InferredTypes::or).orElse(anyType());
  }

  public static InferredType fromTypeAnnotation(TypeAnnotation typeAnnotation) {
    Map<String, Symbol> builtins = TypeShed.builtinSymbols();
    DeclaredType declaredType = declaredTypeFromTypeAnnotation(typeAnnotation.expression(), builtins);
    if (declaredType == null) {
      return InferredTypes.anyType();
    }
    return declaredType;
  }

  public static InferredType fromTypeshedTypeAnnotation(TypeAnnotation typeAnnotation) {
    Map<String, Symbol> builtins = TypeShed.builtinSymbols();
    return runtimeTypefromTypeAnnotation(typeAnnotation.expression(), builtins);
  }

  public static InferredType fromTypeshedProtobuf(SymbolsProtos.Type type) {
    switch (type.getKind()) {
      case INSTANCE:
        String typeName = type.getFullyQualifiedName();
        // _SpecialForm is the type used for some special types, like Callable, Union, TypeVar, ...
        // It comes from CPython impl: https://github.com/python/cpython/blob/e39ae6bef2c357a88e232dcab2e4b4c0f367544b/Lib/typing.py#L439
        // This doesn't seem to be very precisely specified in typeshed, because it has special semantic.
        // To avoid FPs, we treat it as ANY
        if ("typing._SpecialForm".equals(typeName)) {
          return anyType();
        }
        return typeName.isEmpty() ? anyType() : runtimeType(TypeShed.symbolWithFQN(typeName));
      case TYPE:
        return InferredTypes.TYPE;
      case TYPE_ALIAS:
        return fromTypeshedProtobuf(type.getArgs(0));
      case CALLABLE:
        // this should be handled as a function type - see SONARPY-953
        return anyType();
      case UNION:
        return union(type.getArgsList().stream().map(InferredTypes::fromTypeshedProtobuf));
      case TUPLE:
        return InferredTypes.TUPLE;
      case NONE:
        return InferredTypes.NONE;
      case TYPED_DICT:
        return InferredTypes.DICT;
      case TYPE_VAR:
        return Optional.of(type)
          .filter(InferredTypes::filterTypeVar)
          .map(SymbolsProtos.Type::getFullyQualifiedName)
          .map(TypeShed::symbolWithFQN)
          .map(InferredTypes::runtimeType)
          .orElseGet(InferredTypes::anyType);
      default:
        return anyType();
    }
  }

  private static final Set<String> EXCLUDING_TYPE_VAR_FQN_PATTERNS = Set.of(
    "^builtins\\.object$",
    "^_ctypes\\._CanCastTo$"); // ref: SONARPY-1477

  public static boolean filterTypeVar(SymbolsProtos.Type type) {
    return Optional.of(type)
      // Filtering self returning methods until the SONARPY-1472 will be solved
      .filter(Predicate.not(t -> t.getPrettyPrintedName().endsWith(".Self")))
      .map(SymbolsProtos.Type::getFullyQualifiedName)
      .filter(Predicate.not(String::isEmpty))
      .filter(fqn -> EXCLUDING_TYPE_VAR_FQN_PATTERNS.stream().noneMatch(fqn::matches))
      .isPresent();
  }

  @CheckForNull
  private static DeclaredType declaredTypeFromTypeAnnotation(Expression expression, Map<String, Symbol> builtinSymbols) {
    if (expression.is(Kind.NAME) && !((Name) expression).name().equals("Any")) {
      Symbol symbol = ((Name) expression).symbol();
      if (symbol != null) {
        String builtinFqn = ALIASED_ANNOTATIONS.get(symbol.fullyQualifiedName());
        return builtinFqn != null ? new DeclaredType(builtinSymbols.get(builtinFqn)) : new DeclaredType(symbol);
      }
    }
    if (expression.is(Kind.SUBSCRIPTION)) {
      return declaredTypeFromTypeAnnotationSubscription((SubscriptionExpression) expression, builtinSymbols);
    }
    if (expression.is(Kind.NONE)) {
      return new DeclaredType(builtinSymbols.get(BuiltinTypes.NONE_TYPE));
    }
    if (expression.is(Kind.BITWISE_OR)) {
      var binaryExpression = (BinaryExpression) expression;
      return declaredUnionType(binaryExpression.leftOperand(), binaryExpression.rightOperand(), builtinSymbols);
    }
    return null;
  }

  @CheckForNull
  public static DeclaredType declaredUnionType(Expression leftOperand, Expression rightOperand, Map<String, Symbol> builtinSymbols) {
    DeclaredType leftType = declaredTypeFromTypeAnnotation(leftOperand, builtinSymbols);
    DeclaredType rightType = declaredTypeFromTypeAnnotation(rightOperand, builtinSymbols);
    if (leftType == null || rightType == null) {
      return null;
    }
    if (leftType.mustBeOrExtend(BuiltinTypes.NONE_TYPE)) {
      return new DeclaredType(OPTIONAL_SYMBOL, Arrays.asList(rightType));
    }
    if (rightType.mustBeOrExtend(BuiltinTypes.NONE_TYPE)) {
      return new DeclaredType(OPTIONAL_SYMBOL, Arrays.asList(leftType));
    }
    return new DeclaredType(new SymbolImpl("Union", "typing.Union"), Arrays.asList(leftType, rightType));
  }

  @CheckForNull
  private static DeclaredType declaredTypeFromTypeAnnotationSubscription(SubscriptionExpression subscription, Map<String, Symbol> builtinSymbols) {
    if (isAnnotatedSubscription(subscription)) {
      return declaredTypeFromTypeAnnotation(subscription.subscripts().expressions().get(0), builtinSymbols);
    }
    return TreeUtils.getSymbolFromTree(subscription.object())
      .map(symbol -> {
        List<DeclaredType> args = subscription.subscripts().expressions().stream()
          .map(exp -> declaredTypeFromTypeAnnotation(exp, builtinSymbols))
          .toList();
        if (args.stream().anyMatch(Objects::isNull)) {
          // null args indicate something was wrong in the resolution of some of the alternatives
          // returning null here will ensure the resulting type will be AnyType, which will avoid potential FPs
          return null;
        }
        String builtinFqn = ALIASED_ANNOTATIONS.get(symbol.fullyQualifiedName());
        return builtinFqn != null ? new DeclaredType(builtinSymbols.get(builtinFqn), args) : new DeclaredType(symbol, args);
      })
      .orElse(null);
  }

  private static InferredType runtimeTypefromTypeAnnotation(Expression expression, Map<String, Symbol> builtinSymbols) {
    if (expression.is(Kind.NAME) && !((Name) expression).name().equals("Any")) {
      Symbol symbol = ((Name) expression).symbol();
      if (symbol != null) {
        if ("typing.Text".equals(symbol.fullyQualifiedName())) {
          return InferredTypes.runtimeType(builtinSymbols.get("str"));
        }
        return InferredTypes.genericType(symbol, Collections.emptyList(), builtinSymbols);
      }
      return InferredTypes.anyType();
    }
    if (expression.is(Kind.SUBSCRIPTION)) {
      SubscriptionExpression subscription = (SubscriptionExpression) expression;
      if (isAnnotatedSubscription(subscription)) {
        return runtimeTypefromTypeAnnotation(subscription.subscripts().expressions().get(0), builtinSymbols);
      }
      return TreeUtils.getSymbolFromTree(subscription.object())
        .map(symbol -> InferredTypes.genericType(symbol, subscription.subscripts().expressions(), builtinSymbols))
        .orElse(InferredTypes.anyType());
    }
    if (expression.is(Kind.NONE)) {
      return InferredTypes.runtimeType(builtinSymbols.get(BuiltinTypes.NONE_TYPE));
    }
    return InferredTypes.anyType();
  }

  private static boolean isAnnotatedSubscription(SubscriptionExpression subscription) {
    Optional<Symbol> objectSymbol = TreeUtils.getSymbolFromTree(subscription.object());
    return objectSymbol.isPresent() && "typing.Annotated".equals(objectSymbol.get().fullyQualifiedName());
  }

  private static InferredType genericType(Symbol symbol, List<Expression> subscripts, Map<String, Symbol> builtinSymbols) {
    String builtinFqn = ALIASED_ANNOTATIONS.get(symbol.fullyQualifiedName());
    if (builtinFqn == null) {
      if ("typing.Optional".equals(symbol.fullyQualifiedName()) && subscripts.size() == 1) {
        InferredType noneType = InferredTypes.runtimeType(builtinSymbols.get(BuiltinTypes.NONE_TYPE));
        return InferredTypes.or(runtimeTypefromTypeAnnotation(subscripts.get(0), builtinSymbols), noneType);
      }
      if ("typing.Union".equals(symbol.fullyQualifiedName())) {
        return union(subscripts.stream().map(s -> runtimeTypefromTypeAnnotation(s, builtinSymbols)));
      }
      return InferredTypes.runtimeType(symbol);
    }
    return InferredTypes.runtimeType(builtinSymbols.get(builtinFqn));
  }

  public static Collection<ClassSymbol> typeSymbols(InferredType inferredType) {
    if (inferredType instanceof RuntimeType) {
      return Collections.singleton(inferredType.runtimeTypeSymbol());
    }
    if (inferredType instanceof DeclaredType declaredType) {
      Symbol typeClass = declaredType.getTypeClass();
      return typeClass.is(CLASS) ? Collections.singleton(((ClassSymbol) typeClass)) : Collections.emptySet();
    }
    if (inferredType instanceof UnionType unionType) {
      Set<ClassSymbol> typeClasses = new HashSet<>();
      unionType.types().forEach(type -> typeClasses.addAll(typeSymbols(type)));
      return typeClasses;
    }
    return Collections.emptySet();
  }

  @CheckForNull
  public static String typeName(InferredType inferredType) {
    if (inferredType instanceof DeclaredType declaredType) {
      return declaredType.typeName();
    }
    Collection<ClassSymbol> typeClasses = typeSymbols(inferredType);
    if (typeClasses.size() == 1) {
      return typeClasses.iterator().next().name();
    }
    return null;
  }

  @CheckForNull
  public static String fullyQualifiedTypeName(InferredType inferredType) {
    if (inferredType instanceof DeclaredType declaredType) {
      return declaredType.getTypeClass().fullyQualifiedName();
    }
    Collection<ClassSymbol> typeClasses = typeSymbols(inferredType);
    if (typeClasses.size() == 1) {
      return typeClasses.iterator().next().fullyQualifiedName();
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

  public static boolean isDeclaredTypeWithTypeClass(InferredType type, String typeName) {
    if (type instanceof DeclaredType declaredType) {
      Symbol typeClass = declaredType.getTypeClass();
      return typeName.equals(typeClass.fullyQualifiedName());
    }
    return false;
  }

  static boolean isTypeClassCompatibleWith(Symbol typeClass, InferredType other) {
    if (other instanceof RuntimeType runtimeType) {
      return InferredTypes.areSymbolsCompatible(typeClass, runtimeType.getTypeClass());
    }
    if (other instanceof DeclaredType declaredType) {
      if (declaredType.alternativeTypeSymbols().isEmpty()) {
        return true;
      }
      return (declaredType).alternativeTypeSymbols().stream().anyMatch(a -> InferredTypes.areSymbolsCompatible(typeClass, a));
    }
    if (other instanceof UnionType unionType) {
      return unionType.types().stream().anyMatch(t -> InferredTypes.isTypeClassCompatibleWith(typeClass, t));
    }
    // other is AnyType
    return true;
  }

  static boolean areSymbolsCompatible(Symbol actual, Symbol expected) {
    if (!expected.is(CLASS) || !actual.is(CLASS)) {
      return true;
    }
    ClassSymbol actualTypeClass = (ClassSymbol) actual;
    ClassSymbol expectedTypeClass = (ClassSymbol) expected;
    String otherFullyQualifiedName = expectedTypeClass.fullyQualifiedName();
    boolean areHardcodedCompatible = areHardcodedCompatible(actualTypeClass, expectedTypeClass);
    boolean isDuckTypeCompatible = !"NoneType".equals(otherFullyQualifiedName) &&
      expectedTypeClass.declaredMembers().stream().allMatch(m -> actualTypeClass.resolveMember(m.name()).isPresent());
    boolean canBeOrExtend = otherFullyQualifiedName == null || actualTypeClass.canBeOrExtend(otherFullyQualifiedName);
    boolean areHardcodedIncompatible = areHardCodedIncompatible(actualTypeClass, expectedTypeClass);
    boolean areTupleClasses = actualTypeClass.isOrExtends(BuiltinTypes.TUPLE) && expectedTypeClass.isOrExtends(BuiltinTypes.TUPLE);
    return (areHardcodedCompatible || isDuckTypeCompatible || canBeOrExtend || areTupleClasses) && !areHardcodedIncompatible;
  }

  private static boolean areHardcodedCompatible(ClassSymbol actual, ClassSymbol expected) {
    Set<String> compatibleTypes = HARDCODED_COMPATIBLE_TYPES.getOrDefault(actual.fullyQualifiedName(), Collections.emptySet());
    return compatibleTypes.stream().anyMatch(expected::canBeOrExtend);
  }

  private static boolean areHardCodedIncompatible(ClassSymbol actual, ClassSymbol expected) {
    return HARDCODED_INCOMPATIBLE_TYPES.contains(
      Stream.of(actual, expected)
        .map(ClassSymbol::fullyQualifiedName)
        .filter(Objects::nonNull)
        .collect(Collectors.toSet()));
  }

  public static boolean containsDeclaredType(InferredType type) {
    if (type instanceof DeclaredType) {
      return true;
    }
    if (type instanceof UnionType unionType) {
      return unionType.types().stream().anyMatch(InferredTypes::containsDeclaredType);
    }
    return false;
  }

  public static String getBuiltinCategory(InferredType inferredType) {
    List<String> list = BUILTINS_TYPE_CATEGORY.keySet().stream()
      .filter(inferredType::canOnlyBe)
      .map(BUILTINS_TYPE_CATEGORY::get).toList();
    return list.size() == 1 ? list.get(0) : null;
  }

  public static Map<String, String> getBuiltinsTypeCategory() {
    return Collections.unmodifiableMap(BUILTINS_TYPE_CATEGORY);
  }
}
