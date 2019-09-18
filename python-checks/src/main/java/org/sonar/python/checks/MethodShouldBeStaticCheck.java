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

import java.util.List;
import org.sonar.check.Rule;
import org.sonar.python.PythonSubscriptionCheck;
import org.sonar.python.api.tree.PyExpressionStatementTree;
import org.sonar.python.api.tree.PyFunctionDefTree;
import org.sonar.python.api.tree.PyNameTree;
import org.sonar.python.api.tree.PyParameterListTree;
import org.sonar.python.api.tree.PyParameterTree;
import org.sonar.python.api.tree.PyRaiseStatementTree;
import org.sonar.python.api.tree.PyStatementTree;
import org.sonar.python.api.tree.Tree;
import org.sonar.python.tree.BaseTreeVisitor;

import static org.sonar.python.checks.CheckUtils.classHasInheritance;
import static org.sonar.python.checks.CheckUtils.getParentClassDef;

@Rule(key = "S2325")
public class MethodShouldBeStaticCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Make this method static.";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, ctx -> {
      PyFunctionDefTree funcDef = (PyFunctionDefTree) ctx.syntaxNode();
      if (funcDef.isMethodDefinition()
        && !classHasInheritance(getParentClassDef(funcDef))
        && !isBuiltInMethod(funcDef)
        && !isStatic(funcDef)
        && hasValuableCode(funcDef)
        && !mayRaiseNotImplementedError(funcDef)
        && !isUsingSelfArg(funcDef)
      ) {
        ctx.addIssue(funcDef.name(), MESSAGE);
      }
    });
  }

  private static boolean mayRaiseNotImplementedError(PyFunctionDefTree funcDef) {
    RaiseStatementVisitor visitor = new RaiseStatementVisitor();
    funcDef.accept(visitor);
    return visitor.hasNotImplementedError;

  }

  private static boolean hasValuableCode(PyFunctionDefTree funcDef) {
    List<PyStatementTree> statements = funcDef.body().statements();
    return !statements.stream().allMatch(st -> isStringLiteral(st) || st.is(Tree.Kind.PASS_STMT));
  }

  private static boolean isStringLiteral(PyStatementTree st) {
    return st.is(Tree.Kind.EXPRESSION_STMT) && ((PyExpressionStatementTree) st).expressions().stream().allMatch(e -> e.is(Tree.Kind.STRING_LITERAL));
  }

  private static boolean isUsingSelfArg(PyFunctionDefTree funcDef) {
    PyParameterListTree parameters = funcDef.parameters();
    if (parameters == null) {
      // if a method has no parameters then it can't be a instance method.
      return true;
    }
    List<PyParameterTree> params = parameters.nonTuple();
    if (params.isEmpty()) {
      return false;
    }
    PyParameterTree first = params.get(0);
    SelfVisitor visitor = new SelfVisitor(first.name().name());
    funcDef.body().accept(visitor);
    return visitor.isUsingSelfArg;
  }

  private static boolean isStatic(PyFunctionDefTree funcDef) {
    return funcDef.decorators().stream()
      .map(d -> d.name().names().get(d.name().names().size() - 1))
      .anyMatch(n -> n.name().equals("staticmethod") || n.name().equals("classmethod"));
  }

  private static boolean isBuiltInMethod(PyFunctionDefTree funcDef) {
    String name = funcDef.name().name();
    String doubleUnderscore = "__";
    return name.startsWith(doubleUnderscore) && name.endsWith(doubleUnderscore);
  }

  private static class RaiseStatementVisitor extends BaseTreeVisitor {
    private int withinRaise = 0;
    boolean hasNotImplementedError = false;

    @Override
    public void visitRaiseStatement(PyRaiseStatementTree pyRaiseStatementTree) {
      withinRaise++;
      scan(pyRaiseStatementTree.expressions());
      withinRaise--;
    }

    @Override
    public void visitName(PyNameTree pyNameTree) {
      if (withinRaise > 0) {
        hasNotImplementedError |= pyNameTree.name().equals("NotImplementedError");
      }
    }
  }

  private static class SelfVisitor extends BaseTreeVisitor {
    private final String selfName;
    boolean isUsingSelfArg = false;

    SelfVisitor(String selfName) {
      this.selfName = selfName;
    }

    @Override
    public void visitName(PyNameTree pyNameTree) {
      isUsingSelfArg |= selfName.equals(pyNameTree.name());
    }
  }
}
