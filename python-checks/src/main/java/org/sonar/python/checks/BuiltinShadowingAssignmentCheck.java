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
package org.sonar.python.checks;

import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.AliasedName;
import org.sonar.plugins.python.api.tree.AnnotatedAssignment;
import org.sonar.plugins.python.api.tree.AssignmentExpression;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.ParameterList;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

import static org.sonar.plugins.python.api.symbols.Symbol.Kind.FUNCTION;

@Rule(key = BuiltinShadowingAssignmentCheck.CHECK_KEY)
public class BuiltinShadowingAssignmentCheck extends PythonSubscriptionCheck {

  private static final boolean DEFAULT_REPORT_ON_PARAMETERS = false;

  @RuleProperty(
    key = "reportOnParameters",
    description = "Enable issues on functions', methods' and lambdas' parameters which have the same name as a builtin.",
    defaultValue = "" + DEFAULT_REPORT_ON_PARAMETERS)
  public boolean reportOnParameters = DEFAULT_REPORT_ON_PARAMETERS;

  public static final String CHECK_KEY = "S5806";

  public static final String MESSAGE = "Rename this %s; it shadows a builtin.";

  private final Set<String> reservedNames = new HashSet<>(Arrays.asList(
    "ArithmeticError", "AssertionError", "AttributeError", "BaseException", "BufferError", "BytesWarning", "DeprecationWarning",
    "EOFError", "Ellipsis", "EnvironmentError", "Exception", "False", "FloatingPointError", "FutureWarning", "GeneratorExit",
    "IOError", "ImportError", "ImportWarning", "IndentationError", "IndexError", "KeyError", "KeyboardInterrupt", "LookupError",
    "MemoryError", "NameError", "None", "NotImplemented", "NotImplementedError", "OSError", "OverflowError", "PendingDeprecationWarning",
    "ReferenceError", "RuntimeError", "RuntimeWarning", "StopIteration", "SyntaxError", "SyntaxWarning", "SystemError", "SystemExit",
    "TabError", "True", "TypeError", "UnboundLocalError", "UnicodeDecodeError", "UnicodeEncodeError", "UnicodeError", "UnicodeTranslateError",
    "UnicodeWarning", "UserWarning", "ValueError", "Warning", "ZeroDivisionError", "__IPYTHON__", "__debug__", "__doc__", "__import__",
    "__name__", "__package__", "abs", "all", "any", "bin", "bool", "bytearray", "bytes", "callable", "chr", "classmethod", "compile",
    "complex", "copyright", "credits", "delattr", "dict", "dir", "display", "divmod", "enumerate", "eval", "filter", "float", "format",
    "frozenset", "get_ipython", "getattr", "globals", "hasattr", "hash", "help", "hex", "id", "input", "int", "isinstance", "issubclass",
    "iter", "len", "license", "list", "locals", "map", "max", "memoryview", "min", "next", "object", "oct", "open", "ord", "pow", "print",
    "property", "range", "repr", "reversed", "round", "set", "setattr", "slice", "sorted", "staticmethod", "str", "sum", "super", "tuple",
    "type", "vars", "zip"
  ));

  private static final String VRBL_ISSUE_TYPE = "variable";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.ASSIGNMENT_STMT, this::checkAssignment);
    context.registerSyntaxNodeConsumer(Tree.Kind.ANNOTATED_ASSIGNMENT, this::checkAnnotatedAssignment);
    context.registerSyntaxNodeConsumer(Tree.Kind.ASSIGNMENT_EXPRESSION, this::checkAssignmentExpression);
    context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, this::checkClassDefinition);
    context.registerSyntaxNodeConsumer(Tree.Kind.FUNCDEF, this::checkFunctionDeclaration);
    context.registerSyntaxNodeConsumer(Tree.Kind.IMPORT_FROM, ctx ->
      ((ImportFrom) ctx.syntaxNode()).importedNames().forEach(m -> checkImportedNameAlias(ctx, m)));
    context.registerSyntaxNodeConsumer(Tree.Kind.PARAMETER_LIST, this::checkParameterList);
  }

  private void checkAssignmentExpression(SubscriptionContext ctx) {
    AssignmentExpression assignmentExpression = (AssignmentExpression) ctx.syntaxNode();
    Name lhsName = assignmentExpression.lhsName();
    if (shouldReportIssue(lhsName)) {
      ctx.addIssue(lhsName, String.format(MESSAGE, VRBL_ISSUE_TYPE));
    }
  }

  private void checkAssignment(SubscriptionContext ctx) {
    AssignmentStatement assignment = (AssignmentStatement) ctx.syntaxNode();
    Tree ancestor = TreeUtils.firstAncestorOfKind(assignment, Tree.Kind.FUNCDEF, Tree.Kind.CLASSDEF);
    if (ancestor == null || ancestor.is(Tree.Kind.FUNCDEF)) {
      for (int i = 0; i < assignment.lhsExpressions().size(); i++) {
        for (Expression expression : assignment.lhsExpressions().get(i).expressions()) {
          if (shouldReportIssue(expression)) {
            ctx.addIssue(expression, String.format(MESSAGE, VRBL_ISSUE_TYPE));
          }
        }
      }
    }
  }

  private void checkAnnotatedAssignment(SubscriptionContext ctx) {
    AnnotatedAssignment assignment = (AnnotatedAssignment) ctx.syntaxNode();
    Tree ancestor = TreeUtils.firstAncestorOfKind(assignment, Tree.Kind.FUNCDEF, Tree.Kind.CLASSDEF);
    if (ancestor == null || ancestor.is(Tree.Kind.FUNCDEF)) {
      Expression variable = assignment.variable();
      Token equalToken = assignment.equalToken();
      if (equalToken != null && shouldReportIssue(variable)) {
        ctx.addIssue(variable, String.format(MESSAGE, VRBL_ISSUE_TYPE));
      }
    }
  }

  private void checkFunctionDeclaration(SubscriptionContext ctx) {
    FunctionDef functionDef = (FunctionDef) ctx.syntaxNode();
    if (!functionDef.isMethodDefinition() && shouldReportIssue(functionDef.name())) {
      ctx.addIssue(functionDef.name(), String.format(MESSAGE, "function"));
    }
  }

  private void checkParameterList(SubscriptionContext ctx) {
    ParameterList parameterList = (ParameterList) ctx.syntaxNode();
    if (reportOnParameters && !checkIfMethodIsOverride(parameterList)) {
      for (Parameter parameter : parameterList.nonTuple()) {
        Name parameterName = parameter.name();
        if (parameterName != null && isBuiltInName(parameterName)) {
          ctx.addIssue(parameterName, String.format(MESSAGE, "parameter"));
        }
      }
    }
  }

  private static boolean checkIfMethodIsOverride(ParameterList parameterList) {
    Tree parent = parameterList.parent();
    if (parent.is(Tree.Kind.FUNCDEF) && ((FunctionDef) parent).isMethodDefinition()) {
      FunctionDef method = (FunctionDef) parent;
      Symbol symbol = method.name().symbol();
      if (symbol == null || symbol.kind() != FUNCTION) {
        return false;
      }
      FunctionSymbol functionSymbol = (FunctionSymbol) symbol;
      return CheckUtils.getOverriddenMethod(functionSymbol).isPresent();
    }
    return false;
  }

  private void checkClassDefinition(SubscriptionContext ctx) {
    ClassDef classDef = (ClassDef) ctx.syntaxNode();
    if (shouldReportIssue(classDef.name())) {
      ctx.addIssue(classDef.name(), String.format(MESSAGE, "class"));
    }
  }

  private void checkImportedNameAlias(SubscriptionContext ctx, AliasedName aliasedName) {
    Name alias = aliasedName.alias();
    if (alias != null && shouldReportIssue(alias)) {
      ctx.addIssue(alias, String.format(MESSAGE, "alias"));
    }
  }

  private boolean shouldReportIssue(Tree tree) {
    return tree.is(Tree.Kind.NAME) && isBuiltInName((Name) tree) && !shouldSkipIssueCondition(tree.parent());
  }

  private boolean isBuiltInName(Name name) {
    return reservedNames.contains(name.name());
  }

  private static boolean shouldSkipIssueCondition(Tree name) {
    // No issue will be raised when the defined name is in an except clause,
    // the else clause of a try or an if block at module level.
    if (TreeUtils.firstAncestorOfKind(name, Tree.Kind.FUNCDEF, Tree.Kind.CLASSDEF) == null) {
      return TreeUtils.firstAncestorOfKind(name, Tree.Kind.EXCEPT_CLAUSE) != null ||
        TreeUtils.firstAncestorOfKind(name, Tree.Kind.IF_STMT) != null ||
        TreeUtils.firstAncestorOfKind(name, Tree.Kind.ELSE_CLAUSE) != null;
    }
    return false;
  }
}
