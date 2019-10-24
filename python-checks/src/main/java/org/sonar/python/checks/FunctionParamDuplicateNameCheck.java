/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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
package org.sonar.python.checks;

import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Collectors;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.ParameterList;
import org.sonar.plugins.python.api.tree.Tree;

@Rule(key = "S1536")
public class FunctionParamDuplicateNameCheck extends PythonSubscriptionCheck {

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
      FunctionDef tree = (FunctionDef) ctx.syntaxNode();
      ParameterList parameters = tree.parameters();
      if (parameters == null) {
        return;
      }
      Set<String> paramNames = new HashSet<>();
      List<Name> duplications = parameters.nonTuple().stream()
        .map(Parameter::name)
        .filter(Objects::nonNull)
        .filter(paramName -> !paramNames.add(paramName.name()))
        .collect(Collectors.toList());
      if (!duplications.isEmpty()) {
        String plural = duplications.size() > 1 ? "s" : "";
        String list = duplications.stream().map(Name::name).collect(Collectors.joining(", "));
        ctx.addIssue(tree.name(), String.format("Rename the duplicated function parameter%s \"%s\".", plural, list));
      }
    });
  }
}
