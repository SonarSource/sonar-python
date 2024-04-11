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

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Optional;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ImportName;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.tree.ListLiteralImpl;
import org.sonar.python.tree.NameImpl;
import org.sonar.python.tree.NumericLiteralImpl;
import org.sonar.python.tree.StringLiteralImpl;
import org.sonar.python.types.RuntimeType;
import org.sonar.python.types.v2.ClassType;
import org.sonar.python.types.v2.FunctionType;
import org.sonar.python.types.v2.Member;
import org.sonar.python.types.v2.ModuleType;
import org.sonar.python.types.v2.ObjectType;
import org.sonar.python.types.v2.ObjectTypeBuilder;
import org.sonar.python.types.v2.PythonType;

public class TypeInferenceV2 extends BaseTreeVisitor {

  private final ProjectLevelTypeTable projectLevelTypeTable;

  private final Deque<PythonType> typeStack = new ArrayDeque<>();

  public TypeInferenceV2(ProjectLevelTypeTable projectLevelTypeTable) {
    this.projectLevelTypeTable = projectLevelTypeTable;
  }

  @Override
  public void visitFileInput(FileInput fileInput) {
    var type = new ModuleType("somehow get its name", new HashMap<>());
    inTypeScope(type, () -> super.visitFileInput(fileInput));
  }

  @Override
  public void visitStringLiteral(StringLiteral stringLiteral) {
    ModuleType builtins = this.projectLevelTypeTable.getModule("builtins");
    // TODO: multiple object types to represent str instance?
    ((StringLiteralImpl) stringLiteral).typeV2(new ObjectType(builtins.resolveMember("str"), List.of(), List.of()));
  }

  @Override
  public void visitNumericLiteral(NumericLiteral numericLiteral) {
    ModuleType builtins = this.projectLevelTypeTable.getModule("builtins");
    InferredType type = numericLiteral.type();
    String memberName = ((RuntimeType) type).getTypeClass().fullyQualifiedName();
    if (memberName != null) {
      ((NumericLiteralImpl) numericLiteral).typeV2(new ObjectType(builtins.resolveMember(memberName), List.of(), List.of()));
    }
  }

  @Override
  public void visitListLiteral(ListLiteral listLiteral) {
    ModuleType builtins = this.projectLevelTypeTable.getModule("builtins");
    scan(listLiteral.elements());
    List<PythonType> pythonTypes = listLiteral.elements().expressions().stream().map(Expression::typeV2).distinct().toList();
    // TODO: cleanly reduce attributes
    ((ListLiteralImpl) listLiteral).typeV2(new ObjectType(builtins.resolveMember("list"), pythonTypes, List.of()));
  }

  @Override
  public void visitClassDef(ClassDef classDef) {
    scan(classDef.args());
    Name name = classDef.name();
    ClassType type = new ClassType(name.name());
    ((NameImpl) name).typeV2(type);
    
    inTypeScope(type, () -> scan(classDef.body()));
  }

  @Override
  public void visitFunctionDef(FunctionDef functionDef) {
    scan(functionDef.decorators());
    FunctionType functionType = new FunctionType(functionDef.name().name(), new ArrayList<>(), new ArrayList<>(), PythonType.UNKNOWN);
    if (currentType() instanceof ClassType classType) {
      if (functionDef.name().symbolV2().hasSingleBindingUsage()) {
        classType.members().add(new Member(functionType.name(), functionType));
      } else {
        // TODO: properly infer type in case of multiple assignments
        classType.members().add(new Member(functionType.name(), PythonType.UNKNOWN));
      }
    }
    ((NameImpl) functionDef.name()).typeV2(functionType);
    inTypeScope(functionType, () -> {
      // TODO: check scope accuracy
      scan(functionDef.typeParams());
      scan(functionDef.parameters());
      scan(functionDef.returnTypeAnnotation());
      scan(functionDef.body());
    });
  }

  @Override
  public void visitImportName(ImportName importName) {
    //createImportedNames(importName.modules(), null, Collections.emptyList());
    super.visitImportName(importName);
  }

  @Override
  public void visitAssignmentStatement(AssignmentStatement assignmentStatement) {
    scan(assignmentStatement.assignedValue());
    Optional.of(assignmentStatement)
      .map(AssignmentStatement::lhsExpressions)
      .filter(lhs -> lhs.size() == 1)
      .map(lhs -> lhs.get(0))
      .map(ExpressionList::expressions)
      .filter(lhs -> lhs.size() == 1)
      .map(lhs -> lhs.get(0))
      .filter(NameImpl.class::isInstance)
      .map(NameImpl.class::cast)
      .ifPresent(lhsName -> {
        var assignedValueType = assignmentStatement.assignedValue().typeV2();
        lhsName.typeV2(assignedValueType);
      });
  }

  @Override
  public void visitQualifiedExpression(QualifiedExpression qualifiedExpression) {
    scan(qualifiedExpression.qualifier());
    if (qualifiedExpression.name() instanceof NameImpl name) {
      var nameType = qualifiedExpression.qualifier().typeV2().resolveMember(qualifiedExpression.name().name());
      name.typeV2(nameType);
    }
  }

  @Override
  public void visitName(Name name) {
    var type = Optional.of(name)
      .map(Name::symbolV2)
      .map(SymbolV2::usages)
      .stream()
      .flatMap(Collection::stream)
      .filter(UsageV2::isBindingUsage)
      .map(UsageV2::tree)
      .filter(Expression.class::isInstance)
      .map(Expression.class::cast)
      .map(Expression::typeV2)
      .toList();

    if (type.size() == 1 && name instanceof NameImpl nameImpl) {
      nameImpl.typeV2(type.get(0));
    }
  }

  private PythonType currentType() {
    return typeStack.peek();
  }

  private void inTypeScope(PythonType pythonType, Runnable runnable) {
    this.typeStack.push(pythonType);
    runnable.run();
    this.typeStack.poll();
  }

}
