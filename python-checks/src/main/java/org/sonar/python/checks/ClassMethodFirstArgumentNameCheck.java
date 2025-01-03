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
package org.sonar.python.checks;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.quickfix.PythonTextEdit;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.ParameterList;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.quickfix.TextEditUtils;
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
        .map(String::trim)
        .toList();
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
      PreciseIssue issue = ctx.addIssue(parameterName,
        String.format("Rename \"%s\" to a valid class parameter name or add the missing class parameter.", parameterName.name()));
      
      issue.addQuickFix(addClsAsTheFirstArgument(parameterName));
      issue.addQuickFix(renameTheFirstArgument(parameterName));
    }
  }
  
  private PythonQuickFix addClsAsTheFirstArgument(Name parameterName) {
    String newName = newName();
    PythonTextEdit text;
    if (isSecondArgInNewLine(parameterName)) {
      int indent = secondArgIndent(parameterName);
      text = TextEditUtils.insertBefore(parameterName, newName + ",\n" + " ".repeat(indent));
    } else {
      text = TextEditUtils.insertBefore(parameterName, newName + ", ");
    }
    return PythonQuickFix.newQuickFix(String.format("Add '%s' as the first argument.", newName))
      .addTextEdit(text)
      .build();
  }

  private static boolean isSecondArgInNewLine(Name parameterName) {
    List<Tree> parameterListChildren = parameterName.parent().parent().children();
    if(parameterListChildren.size() >= 2) {
      Tree secondArg = parameterListChildren.get(2);
      return parameterName.firstToken().line() != secondArg.firstToken().line();
    }
    return false;
  }
  private static int secondArgIndent(Name parameterName) {
    List<Tree> parameterListChildren = parameterName.parent().parent().children();
    Tree secondArg = parameterListChildren.get(2);
    return secondArg.firstToken().column();
  }

  private PythonQuickFix renameTheFirstArgument(Name parameterName) {
    String newName = newName();

    return PythonQuickFix.newQuickFix(String.format("Rename '%s' to '%s'", parameterName.name(), newName))
      .addTextEdit(TextEditUtils.renameAllUsages(parameterName, newName))
      .build();
  }

  private String newName() {
    return classParameterNames().get(0).isEmpty() ? "cls" : classParameterNames().get(0);
  }
}
