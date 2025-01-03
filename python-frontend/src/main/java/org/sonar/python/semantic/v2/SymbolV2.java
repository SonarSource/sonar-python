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
package org.sonar.python.semantic.v2;

import java.util.ArrayList;
import java.util.List;
import org.sonar.api.Beta;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.python.tree.NameImpl;

@Beta
public class SymbolV2 {

  private final String name;
  private final List<UsageV2> usages;

  public SymbolV2(String name, List<UsageV2> usages) {
    this.name = name;
    this.usages = usages;
  }

  public SymbolV2(String name) {
    this(name, new ArrayList<>());
  }

  void addUsage(Name name, UsageV2.Kind kind) {
    UsageV2 usage = new UsageV2(name, kind);
    usages.add(usage);
    if (name instanceof NameImpl ni) {
      ni.symbolV2(this);
    }
  }

  @Beta
  public boolean hasSingleBindingUsage() {
    return usages.stream().filter(UsageV2::isBindingUsage).toList().size() == 1;
  }

  public String name() {
    return name;
  }

  public List<UsageV2> usages() {
    return usages;
  }

}
