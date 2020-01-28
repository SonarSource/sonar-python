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
package org.sonar.python.semantic;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.NameImpl;

public class SymbolImpl implements Symbol {

  private final String name;
  @Nullable
  String fullyQualifiedName;
  private final List<Usage> usages = new ArrayList<>();
  private Map<String, Symbol> childrenSymbolByName = new HashMap<>();
  private Kind kind;
  private Type type;

  public SymbolImpl(String name, @Nullable String fullyQualifiedName) {
    this.name = name;
    this.fullyQualifiedName = fullyQualifiedName;
    this.kind = Kind.OTHER;
    this.type = null;
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

  void updateChildrenFQNBasedOnType() {
    if (type != null) {
      String typeFQN = type.symbol().fullyQualifiedName();
      childrenSymbolByName.values().forEach(childSymbol ->
        ((SymbolImpl) childSymbol).fullyQualifiedName = typeFQN + "." + childSymbol.name());
    }
  }

  void addChildSymbol(Symbol symbol) {
    childrenSymbolByName.put(symbol.name(), symbol);
  }

  void setType(@Nullable Type type) {
    this.type = type;
  }

  @CheckForNull
  public Type type() {
    return type;
  }
}
