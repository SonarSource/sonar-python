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

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.plugins.python.api.PythonSubscriptionCheck;
import org.sonar.plugins.python.api.SubscriptionContext;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.AssignmentStatement;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.HasSymbol;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.SliceExpression;
import org.sonar.plugins.python.api.tree.SliceItem;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.SubscriptionExpression;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.Tree.Kind;
import org.sonar.plugins.python.api.tree.UnaryExpression;
import org.sonar.python.checks.utils.CheckUtils;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S4143")
public class OverwrittenCollectionEntryCheck extends PythonSubscriptionCheck {

  @Override
  public void initialize(Context context) {
    context.registerSyntaxNodeConsumer(Kind.STATEMENT_LIST, ctx -> check(ctx, (StatementList) ctx.syntaxNode()));
  }

  private static void check(SubscriptionContext ctx, StatementList statementList) {
    Map<CollectionKey, List<CollectionWrite>> collectionWrites = new HashMap<>();
    for (Statement statement : statementList.statements()) {
      CollectionWrite write = null;
      if (statement.is(Kind.ASSIGNMENT_STMT)) {
        AssignmentStatement assignment = (AssignmentStatement) statement;
        Expression expression = lhs(assignment);
        write = collectionWrite(assignment, expression);
      }
      if (write != null) {
        collectionWrites.computeIfAbsent(write.collectionKey, k -> new ArrayList<>()).add(write);
      } else {
        reportOverwrites(ctx, collectionWrites);
        collectionWrites.clear();
      }
    }
    reportOverwrites(ctx, collectionWrites);
  }

  private static Expression lhs(AssignmentStatement assignment) {
    return assignment.lhsExpressions().get(0).expressions().get(0);
  }

  @CheckForNull
  private static CollectionWrite collectionWrite(AssignmentStatement assignment, Expression expression) {
    if (expression.is(Kind.SLICE_EXPR)) {
      SliceExpression sliceExpression = (SliceExpression) expression;
      String key = key(sliceExpression.sliceList().children());
      return collectionWrite(assignment, sliceExpression.object(), key, sliceExpression.leftBracket(), sliceExpression.rightBracket());
    } else if (expression.is(Kind.SUBSCRIPTION)) {
      SubscriptionExpression subscription = (SubscriptionExpression) expression;
      String key = key(subscription.subscripts().children());
      return collectionWrite(assignment, subscription.object(), key, subscription.leftBracket(), subscription.rightBracket());
    } else {
      return null;
    }
  }

  private static CollectionWrite collectionWrite(AssignmentStatement assignment, Expression collection, @Nullable String key, Token lBracket, Token rBracket) {
    if (key == null) {
      return null;
    }

    if (collection.is(Kind.SLICE_EXPR, Kind.SUBSCRIPTION)) {
      CollectionWrite nested = collectionWrite(assignment, collection);
      if (nested != null) {
        return new CollectionWrite(nested.collectionKey.nest(key), nested.leftBracket, rBracket, assignment, collection);
      }
    }


    var collectionSymbols = getCollectionSymbol(collection);
    if (!collectionSymbols.isEmpty()) {
      var collectionKey = new CollectionKey(collectionSymbols, key);
      return new CollectionWrite(collectionKey, lBracket, rBracket, assignment, collection);
    } else {
      return null;
    }
  }

  private static List<Symbol> getCollectionSymbol(Expression collection) {
    if (collection.is(Kind.CALL_EXPR)
      || TreeUtils.hasDescendant(collection, t -> t.is(Kind.CALL_EXPR, Kind.SUBSCRIPTION, Kind.SLICE_EXPR))) {
      return List.of();
    }
    var names = findNames(collection);
    return names.stream()
      .map(HasSymbol::symbol)
      .filter(Objects::nonNull)
      .toList();
  }

  private static List<Name> findNames(Tree tree) {
    if (tree.is(Kind.NAME)) {
      return List.of((Name) tree);
    } else {
      return tree.children()
        .stream()
        .map(OverwrittenCollectionEntryCheck::findNames)
        .flatMap(Collection::stream)
        .toList();
    }
  }

  @CheckForNull
  private static String key(List<Tree> trees) {
    StringBuilder key = new StringBuilder();
    for (Tree tree : trees) {
      String keyElement = key(tree);
      if (keyElement == null) {
        return null;
      }
      key.append(keyElement);
    }
    return key.toString();
  }

  @CheckForNull
  private static String key(Tree tree) {
    if (tree.is(Kind.TOKEN)) {
      return ((Token) tree).value();
    } else if (tree.is(Kind.NUMERIC_LITERAL)) {
      return ((NumericLiteral) tree).valueAsString();
    } else if (tree.is(Kind.STRING_LITERAL)) {
      return Expressions.unescape((StringLiteral) tree);
    } else if (tree.is(Kind.NAME)) {
      return ((Name) tree).name();
    } else if (tree.is(Kind.SLICE_ITEM)) {
      SliceItem sliceItem = (SliceItem) tree;
      List<String> keyParts = Stream.of(sliceItem.lowerBound(), sliceItem.upperBound(), sliceItem.stride())
        .map(e -> e == null ? "" : key(e))
        .toList();
      return keyParts.contains(null) ? null : String.join(":", keyParts);
    } else if (tree.is(Kind.UNARY_MINUS)) {
      String nested = key(((UnaryExpression) tree).expression());
      return nested == null ? null : ("-" + nested);
    }
    return null;
  }

  private static void reportOverwrites(SubscriptionContext ctx, Map<CollectionKey, List<CollectionWrite>> collectionWrites) {
    collectionWrites.forEach((key, writes) -> {
      if (writes.size() > 1) {
        CollectionWrite firstWrite = writes.get(0);
        CollectionWrite secondWrite = writes.get(1);
        AssignmentStatement assignment = secondWrite.assignment;
        if (TreeUtils.hasDescendant(assignment.assignedValue(), t -> CheckUtils.areEquivalent(firstWrite.collection, t))) {
          return;
        }
        String message = String.format(
          "Verify this is the key that was intended; a value has already been saved for it on line %s.",
          firstWrite.leftBracket.line());
        ctx.addIssue(secondWrite.leftBracket, secondWrite.rightBracket, message)
          .secondary(IssueLocation.preciseLocation(firstWrite.leftBracket, firstWrite.rightBracket, "Original value."));
      }
    });
  }

  private static class CollectionKey extends AbstractMap.SimpleImmutableEntry<List<Symbol>, String> {

    private CollectionKey(List<Symbol> collection, String key) {
      super(collection, key);
    }

    private CollectionKey nest(String parentTreeKey) {
      return new CollectionKey(getKey(), getValue() + "/" + parentTreeKey);
    }
  }

  private static class CollectionWrite {
    private final CollectionKey collectionKey;
    private final Token leftBracket;
    private final Token rightBracket;
    private final AssignmentStatement assignment;
    private final Expression collection;

    private CollectionWrite(CollectionKey collectionKey, Token leftBracket, Token rightBracket, AssignmentStatement assignment, Expression collection) {
      this.collectionKey = collectionKey;
      this.leftBracket = leftBracket;
      this.rightBracket = rightBracket;
      this.assignment = assignment;
      this.collection = collection;
    }
  }
}
