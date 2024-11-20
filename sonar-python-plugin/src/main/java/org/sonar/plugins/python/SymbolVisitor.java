/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2024 SonarSource SA
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
package org.sonar.plugins.python;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import org.sonar.api.batch.sensor.symbol.NewSymbol;
import org.sonar.api.batch.sensor.symbol.NewSymbolTable;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.symbols.Usage;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.ComprehensionExpression;
import org.sonar.plugins.python.api.tree.DictCompExpression;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.LambdaExpression;
import org.sonar.plugins.python.api.tree.Tree;

public class SymbolVisitor extends BaseTreeVisitor {

  private final NewSymbolTable newSymbolTable;

  public SymbolVisitor(NewSymbolTable newSymbolTable) {
    this.newSymbolTable = newSymbolTable;
  }

  @Override
  public void visitClassDef(ClassDef classDef) {
    classDef.classFields().forEach(this::handleSymbol);
    classDef.instanceFields().forEach(this::handleSymbol);
    super.visitClassDef(classDef);
  }

  @Override
  public void visitFunctionDef(FunctionDef functionDef) {
    functionDef.localVariables().forEach(this::handleSymbol);
    super.visitFunctionDef(functionDef);
  }

  @Override
  public void visitLambda(LambdaExpression lambdaExpression) {
    lambdaExpression.localVariables().forEach(this::handleSymbol);
    super.visitLambda(lambdaExpression);
  }

  @Override
  public void visitPyListOrSetCompExpression(ComprehensionExpression tree) {
    tree.localVariables().forEach(this::handleSymbol);
    super.visitPyListOrSetCompExpression(tree);
  }

  @Override
  public void visitDictCompExpression(DictCompExpression tree) {
    tree.localVariables().forEach(this::handleSymbol);
    super.visitDictCompExpression(tree);
  }

  @Override
  public void visitFileInput(FileInput fileInput) {
    fileInput.globalVariables().forEach(this::handleSymbol);
    super.visitFileInput(fileInput);
    newSymbolTable.save();
  }

  private void handleSymbol(Symbol symbol) {
    if (symbol.usages().isEmpty()) {
      // global symbols might not have any usages
      return;
    }
    List<Usage> usages = new ArrayList<>(symbol.usages());
    usages.sort(Comparator.comparingInt(u -> u.tree().firstToken().line()));
    Tree firstUsageTree = usages.get(0).tree();
    NewSymbol newSymbol = newSymbolTable.newSymbol(firstUsageTree.firstToken().line(), firstUsageTree.firstToken().column(),
      firstUsageTree.lastToken().line(), firstUsageTree.lastToken().column() + firstUsageTree.lastToken().value().length());
    for (int i = 1; i < usages.size(); i++) {
      Tree usageTree = usages.get(i).tree();
      newSymbol.newReference(usageTree.firstToken().line(), usageTree.firstToken().column(),
        usageTree.lastToken().line(), usageTree.lastToken().column() + usageTree.lastToken().value().length());
    }
  }
}
