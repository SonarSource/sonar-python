/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.python.semantic.v2;

import java.util.ArrayList;
import java.util.List;
import javax.annotation.Nullable;
import org.sonar.api.Beta;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.python.tree.NameImpl;

@Beta
public record SymbolV2(String name, @Nullable String fullyQualifiedName, List<UsageV2> usages) {

  public SymbolV2(String name) {
    this(name, null, new ArrayList<>());
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
}
