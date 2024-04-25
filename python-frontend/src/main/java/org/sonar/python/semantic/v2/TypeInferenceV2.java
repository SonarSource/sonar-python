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
import java.util.List;
import java.util.Optional;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.tree.ArgList;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.DictionaryLiteral;
import org.sonar.plugins.python.api.tree.DottedName;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionList;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.ImportFrom;
import org.sonar.plugins.python.api.tree.ImportName;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.NoneExpression;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.QualifiedExpression;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.SetLiteral;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tuple;
import org.sonar.plugins.python.api.types.InferredType;
import org.sonar.python.tree.DictionaryLiteralImpl;
import org.sonar.python.tree.ListLiteralImpl;
import org.sonar.python.tree.NameImpl;
import org.sonar.python.tree.NoneExpressionImpl;
import org.sonar.python.tree.NumericLiteralImpl;
import org.sonar.python.tree.SetLiteralImpl;
import org.sonar.python.tree.StringLiteralImpl;
import org.sonar.python.tree.TupleImpl;
import org.sonar.python.types.RuntimeType;
import org.sonar.python.types.v2.ClassType;
import org.sonar.python.types.v2.FunctionType;
import org.sonar.python.types.v2.Member;
import org.sonar.python.types.v2.ModuleType;
import org.sonar.python.types.v2.ObjectType;
import org.sonar.python.types.v2.PythonType;

public class TypeInferenceV2 extends BaseTreeVisitor {

  private final ProjectLevelTypeTable projectLevelTypeTable;

  private final Deque<PythonType> typeStack = new ArrayDeque<>();

  public TypeInferenceV2(ProjectLevelTypeTable projectLevelTypeTable) {
    this.projectLevelTypeTable = projectLevelTypeTable;
  }

  private static final String BUILTINS = "builtins";

  @Override
  public void visitFileInput(FileInput fileInput) {
    var type = new ModuleType("somehow get its name");
    inTypeScope(type, () -> super.visitFileInput(fileInput));
  }

  @Override
  public void visitStringLiteral(StringLiteral stringLiteral) {
    ModuleType builtins = this.projectLevelTypeTable.getModule(BUILTINS);
    // TODO: multiple object types to represent str instance?
    PythonType strType = builtins.resolveMember("str").orElse(PythonType.UNKNOWN);
    ((StringLiteralImpl) stringLiteral).typeV2(new ObjectType(strType, List.of(), List.of()));
  }

  @Override
  public void visitTuple(Tuple tuple) {
    super.visitTuple(tuple);
    List<PythonType> contentTypes = tuple.elements().stream().map(Expression::typeV2).distinct().toList();
    List<PythonType> attributes = List.of();
    if (contentTypes.size() == 1 && !contentTypes.get(0).equals(PythonType.UNKNOWN)) {
      attributes = contentTypes;
    }
    ModuleType builtins = this.projectLevelTypeTable.getModule(BUILTINS);
    PythonType tupleType = builtins.resolveMember("tuple").orElse(PythonType.UNKNOWN);
    ((TupleImpl) tuple).typeV2(new ObjectType(tupleType,  attributes, List.of()));
  }

  @Override
  public void visitDictionaryLiteral(DictionaryLiteral dictionaryLiteral) {
    super.visitDictionaryLiteral(dictionaryLiteral);
    ModuleType builtins = this.projectLevelTypeTable.getModule(BUILTINS);
    PythonType dictType = builtins.resolveMember("dict").orElse(PythonType.UNKNOWN);
    ((DictionaryLiteralImpl) dictionaryLiteral).typeV2(new ObjectType(dictType,  List.of(), List.of()));
  }

  @Override
  public void visitSetLiteral(SetLiteral setLiteral) {
    super.visitSetLiteral(setLiteral);
    ModuleType builtins = this.projectLevelTypeTable.getModule(BUILTINS);
    PythonType setType = builtins.resolveMember("set").orElse(PythonType.UNKNOWN);
    ((SetLiteralImpl) setLiteral).typeV2(new ObjectType(setType,  List.of(), List.of()));
  }

  @Override
  public void visitNumericLiteral(NumericLiteral numericLiteral) {
    ModuleType builtins = this.projectLevelTypeTable.getModule(BUILTINS);
    InferredType type = numericLiteral.type();
    String memberName = ((RuntimeType) type).getTypeClass().fullyQualifiedName();
    if (memberName != null) {
      PythonType pythonType = builtins.resolveMember(memberName).orElse(PythonType.UNKNOWN);
      ((NumericLiteralImpl) numericLiteral).typeV2(new ObjectType(pythonType, List.of(), List.of()));
    }
  }

  @Override
  public void visitNone(NoneExpression noneExpression) {
    ModuleType builtins = this.projectLevelTypeTable.getModule(BUILTINS);
    // TODO: multiple object types to represent str instance?
    PythonType noneType = builtins.resolveMember("NoneType").orElse(PythonType.UNKNOWN);
    ((NoneExpressionImpl) noneExpression).typeV2(new ObjectType(noneType, List.of(), List.of()));
  }

  @Override
  public void visitListLiteral(ListLiteral listLiteral) {
    ModuleType builtins = this.projectLevelTypeTable.getModule(BUILTINS);
    scan(listLiteral.elements());
    List<PythonType> pythonTypes = listLiteral.elements().expressions().stream().map(Expression::typeV2).distinct().toList();
    // TODO: cleanly reduce attributes
    PythonType listType = builtins.resolveMember("list").orElse(PythonType.UNKNOWN);
    ((ListLiteralImpl) listLiteral).typeV2(new ObjectType(listType, pythonTypes, List.of()));
  }

  @Override
  public void visitClassDef(ClassDef classDef) {
    scan(classDef.args());
    Name name = classDef.name();
    ClassTypeBuilder classTypeBuilder = new ClassTypeBuilder().setName(name.name());
    resolveTypeHierarchy(classDef, classTypeBuilder);
    ClassType type = classTypeBuilder.build();
    ((NameImpl) name).typeV2(type);
    
    inTypeScope(type, () -> scan(classDef.body()));
  }

  static void resolveTypeHierarchy(ClassDef classDef, ClassTypeBuilder classTypeBuilder) {
    Optional.of(classDef)
      .map(ClassDef::args)
      .map(ArgList::arguments)
      .stream()
      .flatMap(Collection::stream)
      .forEach(argument -> {
        if (argument instanceof RegularArgument regularArgument) {
          addParentClass(classTypeBuilder, regularArgument);
        } else {
          classTypeBuilder.superClasses().add(PythonType.UNKNOWN);
        }
      });
  }

  private static void addParentClass(ClassTypeBuilder classTypeBuilder, RegularArgument regularArgument) {
    Name keyword = regularArgument.keywordArgument();
    // TODO: store names if not resolved properly
    if (keyword != null) {
      if ("metaclass".equals(keyword.name())) {
        PythonType argumentType = getTypeV2FromArgument(regularArgument);
        classTypeBuilder.metaClasses().add(argumentType);
      }
      return;
    }
    PythonType argumentType = getTypeV2FromArgument(regularArgument);
    classTypeBuilder.superClasses().add(argumentType);
    // TODO: handle generics
  }

  private static PythonType getTypeV2FromArgument(RegularArgument regularArgument) {
    Expression expression = regularArgument.expression();
    // Ensure we support correctly typing symbols like "List[str] / list[str]"
    return expression.typeV2();
  }

  @Override
  public void visitFunctionDef(FunctionDef functionDef) {
    scan(functionDef.decorators());
    scan(functionDef.typeParams());
    scan(functionDef.parameters());
    FunctionType functionType = buildFunctionType(functionDef);
    ((NameImpl) functionDef.name()).typeV2(functionType);
    inTypeScope(functionType, () -> {
      // TODO: check scope accuracy
      scan(functionDef.typeParams());
      scan(functionDef.parameters());
      scan(functionDef.returnTypeAnnotation());
      scan(functionDef.body());
    });
  }

  private FunctionType buildFunctionType(FunctionDef functionDef) {
    FunctionTypeBuilder functionTypeBuilder = new FunctionTypeBuilder().fromFunctionDef(functionDef);
    ClassType owner = null;
    if (currentType() instanceof ClassType classType) {
      owner = classType;
    }
    if (owner != null) {
      functionTypeBuilder.withOwner(owner);
    }
    FunctionType functionType = functionTypeBuilder.build();
    if (owner != null) {
      if (functionDef.name().symbolV2().hasSingleBindingUsage()) {
        owner.members().add(new Member(functionType.name(), functionType));
      } else {
        owner.members().add(new Member(functionType.name(), PythonType.UNKNOWN));
      }
    }
    return functionType;
  }

  @Override
  public void visitImportName(ImportName importName) {
    importName.modules()
      .forEach(aliasedName -> {
        var names = aliasedName.dottedName().names();
        var fqn = names
              .stream().map(Name::name)
              .toList();
        var module = projectLevelTypeTable.getModule(fqn);

        if (aliasedName.alias() != null) {
          setTypeToName(aliasedName.alias(), module);
        } else {
          for (int i = names.size() - 1; i >= 0; i--) {
            setTypeToName(names.get(i), module);
            module = Optional.ofNullable(module)
              .map(ModuleType::parent)
              .orElse(null);
          }
        }
      });
  }

  @Override
  public void visitImportFrom(ImportFrom importFrom) {
    Optional.of(importFrom)
      .map(ImportFrom::module)
      .map(DottedName::names)
      .ifPresent(names -> {
        var fqn = names
          .stream().map(Name::name)
          .toList();

        var module = projectLevelTypeTable.getModule(fqn);
        importFrom.importedNames().forEach(aliasedName -> aliasedName
          .dottedName()
          .names()
          .stream()
          .findFirst()
          .ifPresent(name -> {
            var type = module.resolveMember(name.name()).orElse(PythonType.UNKNOWN);

            var boundName = Optional.ofNullable(aliasedName.alias())
              .orElse(name);

            setTypeToName(boundName, type);
          }));
      });
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
      var nameType = qualifiedExpression.qualifier().typeV2()
        .resolveMember(qualifiedExpression.name().name())
        .orElse(PythonType.UNKNOWN);
      name.typeV2(nameType);
    }
  }

  @Override
  public void visitName(Name name) {
    SymbolV2 symbolV2 = name.symbolV2();
    if (symbolV2 == null) {
      return;
    }
    List<PythonType> types = new ArrayList<>();
    for (UsageV2 usage : symbolV2.usages()) {
      if (usage.kind().equals(UsageV2.Kind.GLOBAL_DECLARATION)) {
        // Don't infer type for global variables
        return;
      }
      Optional.of(usage)
        .filter(UsageV2::isBindingUsage)
        .map(UsageV2::tree)
        .filter(Expression.class::isInstance)
        .map(Expression.class::cast)
        .map(Expression::typeV2)
        .ifPresent(types::add);
    }
    if (types.size() == 1) {
      setTypeToName(name, types.get(0));
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

  private static void setTypeToName(@Nullable Tree tree, @Nullable PythonType type) {
    if (tree instanceof NameImpl name && type != null) {
      name.typeV2(type);
    }
  }

}
