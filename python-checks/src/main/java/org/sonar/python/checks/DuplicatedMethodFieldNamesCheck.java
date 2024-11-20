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
package org.sonar.python.checks;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;
import java.util.Optional;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.ClassSymbol;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.BaseTreeVisitor;
import org.sonar.plugins.python.api.tree.ClassDef;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S1845")
public class DuplicatedMethodFieldNamesCheck extends PythonSubscriptionCheck {

  private static final String MESSAGE = "Rename %s \"%s\" to prevent any misunderstanding/clash with %s \"%s\" defined on line %s";

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Tree.Kind.CLASSDEF, ctx -> {
      ClassDef classDef = (ClassDef) ctx.syntaxNode();
      ClassSymbol classSymbol = TreeUtils.getClassSymbolFromDef(classDef);
      if (classSymbol == null) {
        return;
      }
      MethodVisitor methodVisitor = new MethodVisitor();
      classDef.body().accept(methodVisitor);
      List<Tree> fieldNames = classSymbol.declaredMembers().stream()
        .filter(s -> s.kind() == Symbol.Kind.OTHER)
        .map(s -> s.usages().stream().findFirst())
        .filter(Optional::isPresent)
        .map(usage -> usage.get().tree())
        .toList();
      lookForDuplications(ctx, fieldNames, methodVisitor.methodNames);
    });
  }

  private static class MethodVisitor extends BaseTreeVisitor {
    private List<Tree> methodNames = new ArrayList<>();

    @Override
    public void visitFunctionDef(FunctionDef pyFunctionDefTree) {
      methodNames.add(pyFunctionDefTree.name());
    }

    @Override
    public void visitClassDef(ClassDef pyClassDefTree) {
      // skip nested class definition
    }
  }

  private static class TokenWithTypeInfo {
    private final Tree tree;
    private final String type;

    TokenWithTypeInfo(Tree tree, String type) {
      this.tree = tree;
      this.type = type;
    }

    String getValue() {
      return tree.firstToken().value();
    }

    int getLine() {
      return tree.firstToken().line();
    }

    String getType() {
      return type;
    }
  }

  private static void lookForDuplications(SubscriptionContext ctx, List<Tree> fieldNames, List<Tree> methodNames) {
    List<TokenWithTypeInfo> allTokensWithInfo = mergeLists(fieldNames, methodNames);
    allTokensWithInfo.sort(Comparator.comparingInt(TokenWithTypeInfo::getLine));
    for (int i = 1; i < allTokensWithInfo.size(); i++) {
      for (int j = i - 1; j >= 0; j--) {
        TokenWithTypeInfo token1 = allTokensWithInfo.get(j);
        TokenWithTypeInfo token2 = allTokensWithInfo.get(i);
        if (differOnlyByCapitalization(token1.getValue(), token2.getValue())) {
          ctx.addIssue(token2.tree, getMessage(token1, token2))
            .secondary(token1.tree, "Original");
          break;
        }
      }
    }
  }

  private static boolean differOnlyByCapitalization(String name1, String name2) {
    return name1.equalsIgnoreCase(name2) && !name1.equals(name2);
  }

  private static List<TokenWithTypeInfo> mergeLists(List<Tree> fieldNames, List<Tree> methodNames) {
    List<TokenWithTypeInfo> allTokensWithInfo = new LinkedList<>();
    for (Tree tree : fieldNames) {
      allTokensWithInfo.add(new TokenWithTypeInfo(tree, "field"));
    }
    for (Tree tree : methodNames) {
      allTokensWithInfo.add(new TokenWithTypeInfo(tree, "method"));
    }
    return allTokensWithInfo;
  }

  private static String getMessage(TokenWithTypeInfo token1, TokenWithTypeInfo token2) {
    return String.format(MESSAGE, token2.getType(), token2.getValue(), token1.getType(), token1.getValue(), token1.getLine());
  }

}
