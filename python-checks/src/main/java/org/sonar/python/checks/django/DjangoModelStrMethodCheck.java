/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2023 SonarSource SA
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
package org.sonar.python.checks.django;

import java.util.Objects;
import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S6554")
public class DjangoModelStrMethodCheck extends PythonSubscriptionCheck {

  public static final String MESSAGE = "Define a \"__str__\" method for this Django model.";
  private static final String DJANGO_MODEL_FQN = "django.db.models.Model";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, ctx -> {
      var classDef = (ClassDef) ctx.syntaxNode();
      if (TreeUtils.getParentClassesFQN(classDef).contains(DJANGO_MODEL_FQN)) {
        var strMethodFqn = getStrMethodFqn(classDef);
        var hasStrMethod = TreeUtils.topLevelFunctionDefs(classDef)
          .stream()
          .map(TreeUtils::getFunctionSymbolFromDef)
          .filter(Objects::nonNull)
          .map(Symbol::fullyQualifiedName)
          .filter(Objects::nonNull)
          .anyMatch(strMethodFqn::equals);

        if (!hasStrMethod) {
          ctx.addIssue(classDef.name(), MESSAGE);
        }
      }

    });
  }

  private static String getStrMethodFqn(ClassDef classDef) {
    var classFqn = Optional.of(classDef)
      .map(TreeUtils::getClassSymbolFromDef)
      .map(Symbol::fullyQualifiedName)
      .orElse("");
    return classFqn + ".__str__";
  }
}
