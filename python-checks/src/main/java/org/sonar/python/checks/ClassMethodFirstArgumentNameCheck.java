/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2021 SonarSource SA
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

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.ParameterList;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S2710")
public class ClassMethodFirstArgumentNameCheck extends PythonSubscriptionCheck {

  private static final String DEFAULT_CLASS_PARAMETER_NAMES = "cls,mcs,metacls";
  private static final List<String> DEFAULT_CLASS_METHODS = Arrays.asList("__init_subclass__", "__class_getitem__", "__new__");
  List<String> classParameterNamesList;

  @RuleProperty(
    key = "classParameterNames",
    description = "Comma separated list of valid class parameter names",
    defaultValue = DEFAULT_CLASS_PARAMETER_NAMES)
  public String classParameterNames = DEFAULT_CLASS_PARAMETER_NAMES;

  private List<String> classParameterNames() {
    if (classParameterNamesList == null) {
      classParameterNamesList = Stream.of(classParameterNames.split(","))
        .map(String::trim).collect(Collectors.toList());
    }
    return classParameterNamesList;
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
      FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
      if (!functionDef.isMethodDefinition()) {
        return;
      }
      if (DEFAULT_CLASS_METHODS.contains(functionDef.name().name())) {
        checkFirstParameterName(functionDef, ctx);
        return;
      }
      for (Decorator decorator : functionDef.decorators()) {
        String decoratorName = TreeUtils.decoratorNameFromExpression(decorator.expression());
        if ("classmethod".equals(decoratorName)) {
          checkFirstParameterName(functionDef, ctx);
        }
      }
    });
  }

  private void checkFirstParameterName(FunctionDef functionDef, SubscriptionContext ctx) {
    ParameterList parameterList = functionDef.parameters();
    if (parameterList == null || !parameterList.all().get(0).is(Tree.Kind.PARAMETER)) {
      // Those cases belong to S5719 scope
      return;
    }
    Parameter parameter = (Parameter) parameterList.all().get(0);
    Name parameterName = parameter.name();
    if (parameterName == null || parameter.starToken() != null) {
      // S5719 scope
      return;
    }
    if (!classParameterNames().contains(parameterName.name())) {
      ctx.addIssue(parameterName, String.format("Rename \"%s\" to a valid class parameter name or add the missing class parameter.", parameterName.name()));
    }
  }
}
