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
package org.sonar.python.semantic;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TypeAnnotation;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.tree.NameImpl;
import org.sonar.python.types.InferredTypes;
import org.sonar.python.types.TypeShed;
import org.sonar.python.types.protobuf.SymbolsProtos;

public class SymbolImpl implements Symbol {

  private final String name;
  @Nullable
  String fullyQualifiedName;
  private final List<Usage> usages = new ArrayList<>();
  private Map<String, Symbol> childrenSymbolByName = new HashMap<>();
  private Kind kind;
  private InferredType inferredType = InferredTypes.anyType();
  private String annotatedTypeName = null;

  /**
   * Represents type deserialized from typeshed or from a custom stub.
   * This will be lazily converted to an InferredType
   */
  private SymbolsProtos.Type deserializedType = null;

  private boolean hasReadDeserializedType = false;

  protected Set<String> validForPythonVersions = Collections.emptySet();

  public SymbolImpl(String name, @Nullable String fullyQualifiedName) {
    this(name, fullyQualifiedName, null);
  }

  public SymbolImpl(String name, @Nullable String fullyQualifiedName, @Nullable String annotatedTypeName) {
    this.name = name;
    this.fullyQualifiedName = fullyQualifiedName;
    this.annotatedTypeName = annotatedTypeName;
    this.kind = Kind.OTHER;
  }

  public SymbolImpl(SymbolsProtos.VarSymbol varSymbol, String moduleName, boolean isFromClass) {
    this.name = varSymbol.getName();
    this.fullyQualifiedName = TypeShed.normalizedFqn(varSymbol.getFullyQualifiedName(), moduleName, name);
    String fqn = varSymbol.getTypeAnnotation().getFullyQualifiedName();
    if (!fqn.isEmpty()) {
      this.annotatedTypeName = TypeShed.normalizedFqn(fqn);
    }
    this.deserializedType = isFromClass ? varSymbol.getTypeAnnotation() : null;
    this.validForPythonVersions = new HashSet<>(varSymbol.getValidForList());
    this.kind = Kind.OTHER;
  }

  @Override
  public String name() {
    return name;
  }

  @Override
  public List<Usage> usages() {
    return Collections.unmodifiableList(usages);
  }

  @CheckForNull
  @Override
  public String fullyQualifiedName() {
    return fullyQualifiedName;
  }

  @Override
  public boolean is(Kind... kinds) {
    Kind symbolKind = kind();
    for (Kind kindIter : kinds) {
      if (symbolKind == kindIter) {
        return true;
      }
    }
    return false;
  }

  @Override
  public Kind kind() {
    return this.kind;
  }

  public void setKind(Kind kind) {
    this.kind = kind;
  }

  void addUsage(Tree tree, Usage.Kind kind) {
    UsageImpl usage = new UsageImpl(tree, kind);
    usages.add(usage);
    if (tree.is(Tree.Kind.NAME)) {
      ((NameImpl) tree).setSymbol(this);
      ((NameImpl) tree).setUsage(usage);
    }
  }

  void addOrCreateChildUsage(Name name, Usage.Kind kind) {
    String childSymbolName = name.name();
    if (!childrenSymbolByName.containsKey(childSymbolName)) {
      String childFullyQualifiedName = fullyQualifiedName != null
        ? (fullyQualifiedName + "." + childSymbolName)
        : null;
      SymbolImpl symbol = new SymbolImpl(childSymbolName, childFullyQualifiedName);
      childrenSymbolByName.put(childSymbolName, symbol);
    }
    Symbol symbol = childrenSymbolByName.get(childSymbolName);
    ((SymbolImpl) symbol).addUsage(name, kind);
  }

  public void addChildSymbol(Symbol symbol) {
    childrenSymbolByName.put(symbol.name(), symbol);
  }


  /**
   * Note that, for symbols that have been deserialized from protobuf, we compute their type lazily.
   *
   * <pre>
   *   a_var : Foo
   *   ...
   *   class Foo: ...
   * </pre>
   *
   * Here, {@code a_var} has type {@code Foo}, which is defined later.
   * Hence, by resolving types lazily, we avoid having to topologically sort dependencies between types declaration and their usages.
   */
  public InferredType inferredType() {
    if (!hasReadDeserializedType && deserializedType != null) {
      inferredType = InferredTypes.fromTypeshedProtobuf(deserializedType);
      hasReadDeserializedType = true;
    }
    return inferredType;
  }

  public void setInferredType(InferredType inferredType) {
    this.inferredType = inferredType;
  }

  @Override
  public String annotatedTypeName() {
    return annotatedTypeName;
  }

  public void setAnnotatedTypeName(TypeAnnotation typeAnnotation) {
    this.annotatedTypeName = Optional.ofNullable(getTypeSymbolFromExpression(typeAnnotation.expression())).map(Symbol::fullyQualifiedName).orElse(null);
  }

  public SymbolImpl copyWithoutUsages() {
    return copyWithoutUsages(name());
  }

  public SymbolImpl copyWithoutUsages(String name) {
    SymbolImpl copiedSymbol = new SymbolImpl(name, fullyQualifiedName());
    copiedSymbol.annotatedTypeName = annotatedTypeName;
    copiedSymbol.deserializedType = deserializedType;
    return copiedSymbol;
  }

  public void removeUsages() {
    usages.clear();
    childrenSymbolByName.values().forEach(symbol -> ((SymbolImpl) symbol).removeUsages());
  }

  public Map<String, Symbol> getChildrenSymbolByName() {
    return Collections.unmodifiableMap(childrenSymbolByName);
  }

  @Nullable
  static Symbol getTypeSymbolFromExpression(Expression expression) {
    if (expression.is(Tree.Kind.SUBSCRIPTION)) {
      SubscriptionExpression subscriptionExpression = (SubscriptionExpression) expression;
      return getTypeSymbolFromExpression(subscriptionExpression.object());
    }
    if (expression instanceof HasSymbol hasSymbol) {
      return hasSymbol.symbol();
    }
    return null;
  }

  public Set<String> validForPythonVersions() {
    return validForPythonVersions;
  }
}
