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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.FunctionSymbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.Parameter;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key="S5720")
public class InstanceMethodSelfAsFirstCheck extends PythonSubscriptionCheck {

  // We allow "_" as sometimes it is conventionally used to signal that self won't be used.
  private static final List<String> ALLOWED_NAMES = Arrays.asList("self", "_");
  private static final List<String> EXCEPTIONS = Arrays.asList("__init_subclass__", "__class_getitem__", "__new__");

  private static final String DEFAULT_IGNORED_DECORATORS = "classproperty";
  private List<String> decoratorsToExclude;

  @RuleProperty(
    key = "ignoredDecorators",
    description = "Comma-separated list of decorators which will disable this rule.",
    defaultValue = DEFAULT_IGNORED_DECORATORS)
  public String ignoredDecorators = DEFAULT_IGNORED_DECORATORS;

  private List<String> getExcludedDecorators() {
    if (decoratorsToExclude == null) {
      decoratorsToExclude = new ArrayList<>();
      decoratorsToExclude.add("staticmethod");
      decoratorsToExclude.add("classmethod");
      decoratorsToExclude.addAll(Arrays.asList(this.ignoredDecorators.split(",")));
    }

    return decoratorsToExclude;
  }

  private boolean isNonInstanceMethodDecorator(Decorator decorator) {
    String fqn = decorator.name().names().stream().map(Name::name).collect(Collectors.joining("."));
    return this.getExcludedDecorators().stream().anyMatch(fqn::contains);
  }

  private static boolean isExceptionalUsage(Usage usage, ClassDef parentClass) {
    Tree tree = usage.tree();
    Tree parentTree = tree.parent();

    // Check if the function is called inside a class body
    if (!parentTree.is(Tree.Kind.CALL_EXPR)) {
      return false;
    }

    CallExpression call = (CallExpression) parentTree;
    return call.callee().equals(tree)
      && TreeUtils.firstAncestorOfKind(call, Tree.Kind.FUNCDEF) == null
      && parentClass.equals(TreeUtils.firstAncestorOfKind(call, Tree.Kind.CLASSDEF));
  }

  private boolean isRelevantMethod(ClassDef classDef, ClassSymbol classSymbol, FunctionDef functionDef) {
    // Skip some known special methods
    if (EXCEPTIONS.contains(functionDef.name().name())) {
      return false;
    }

    // Skip if the class has a ignored decorator
    if (functionDef.decorators().stream().anyMatch(this::isNonInstanceMethodDecorator)) {
      return false;
    }

    FunctionSymbol functionSymbol = TreeUtils.getFunctionSymbolFromDef(functionDef);
    if (functionSymbol == null) {
      return false;
    }

    if (classSymbol.isOrExtends("zope.interface.Interface")) {
      return false;
    }

    return functionSymbol.usages().stream().noneMatch(usage -> isExceptionalUsage(usage, classDef));
  }

  private void handleFunctionDef(SubscriptionContext ctx, ClassDef classDef, ClassSymbol classSymbol, FunctionDef functionDef) {
    List<Parameter> parameters = TreeUtils.positionalParameters(functionDef);
    if (parameters.isEmpty()) {
      return;
    }

    Parameter first = parameters.get(0);
    if (first.starToken() != null) {
      return;
    }

    Optional<Name> paramName = Optional.ofNullable(first.name());
    paramName.ifPresent(name -> {
      if (!ALLOWED_NAMES.contains(name.name()) && isRelevantMethod(classDef, classSymbol, functionDef)) {
        ctx.addIssue(first, String.format("Rename \"%s\" to \"self\" or add the missing \"self\" parameter.", name.name()));
      }
    });
  }

  private static boolean isRelevantClass(ClassDef classDef, ClassSymbol classSymbol) {
    // Do not raise on nested classes - they might use a different name for "self" in order to avoid confusion.
    Tree possibleParent = TreeUtils.firstAncestorOfKind(classDef, Tree.Kind.CLASSDEF);
    if (possibleParent != null) {
      return false;
    }
    
    // Do not raise on meta-classes
    return !classSymbol.isOrExtends("type");
  }

  /**
   * A visitor class used to collect all function definitions within a class def.
   * It is used to discover methods defined within "strange" constructs, such as
   * <code>
   *   class A:
   *       if p:
   *           def f(self): ...
   * </code>
   */
  private static class CollectFunctionDefsVisitor extends BaseTreeVisitor {
    private List<FunctionDef> functionDefs = new ArrayList<>();

    @Override
    public void visitClassDef(ClassDef pyClassDefTree) {
      // Do not descend into nested classes
    }

    @Override
    public void visitFunctionDef(FunctionDef pyFunctionDefTree) {
      this.functionDefs.add(pyFunctionDefTree);
      // Do not descend into nested functions
    }
  }

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, ctx -> {
      ClassDef classDef = (ClassDef) ctx.syntaxNode();
      ClassSymbol classSymbol = TreeUtils.getClassSymbolFromDef(classDef);
      if (classSymbol == null || !isRelevantClass(classDef, classSymbol)) {
        return;
      }

      CollectFunctionDefsVisitor visitor = new CollectFunctionDefsVisitor();
      classDef.body().accept(visitor);

      visitor.functionDefs.forEach(functionDef -> handleFunctionDef(ctx, classDef, classSymbol, functionDef));
    });
  }
}
