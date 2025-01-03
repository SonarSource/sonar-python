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
package org.sonar.plugins.python.api.cfg;

import com.google.common.collect.BiMap;
import com.google.common.collect.HashBiMap;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.CheckForNull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.api.tree.Argument;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonar.plugins.python.api.tree.ExceptClause;
import org.sonar.plugins.python.api.tree.Expression;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.ForStatement;
import org.sonar.plugins.python.api.tree.ListLiteral;
import org.sonar.plugins.python.api.tree.Name;
import org.sonar.plugins.python.api.tree.NumericLiteral;
import org.sonar.plugins.python.api.tree.RegularArgument;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.python.cfg.PythonCfgEndBlock;

class ExpectedCfgStructure {

  static final String EMPTY = "_empty";
  // The string value is the CfgBlock test id
  private final BiMap<CfgBlock, String> testIds;

  // The key is CfgBlock test id
  private final Map<String, BlockExpectation> expectations;
  final List<BlockExpectation> emptyBlockExpectations = new ArrayList<>();

  private ExpectedCfgStructure() {
    testIds = HashBiMap.create();
    expectations = new HashMap<>();
  }

  static ExpectedCfgStructure parse(Set<CfgBlock> blocks) {
    return parse(blocks, Function.identity());
  }

  static ExpectedCfgStructure parse(Set<CfgBlock> blocks, Function<ExpectedCfgStructure, ExpectedCfgStructure> process) {
    return process.apply(Parser.parse(blocks));
  }

  int size() {
    return expectations.size() + emptyBlockExpectations.size();
  }

  boolean hasNonEmptyPredecessors() {
    return expectations.values().stream()
      .anyMatch(expectation -> !expectation.expectedPredecessorIds.isEmpty());
  }

  boolean hasNonEmptyElementNumbers() {
    return expectations.values().stream()
      .anyMatch(expectation -> expectation.expectedNumberOfElements != -1);
  }

  String testId(CfgBlock block) {
    if (block instanceof PythonCfgEndBlock) {
      return "END";
    }
    if (block.elements().isEmpty()) {
      return EMPTY;
    }
    return testIds.get(block);
  }

  List<String> blockIds(Collection<? extends CfgBlock> blocks) {
    return blocks.stream().map(this::testId).toList();
  }

  CfgBlock cfgBlock(String testId) {
    return testIds.inverse().get(testId);
  }

  List<String> expectedSucc(CfgBlock block) {
    return getExpectation(block).expectedSuccessorIds;
  }

  @CheckForNull
  String expectedSyntSucc(CfgBlock block) {
    return getExpectation(block).expectedSyntacticSuccessor;
  }

  List<String> expectedPred(CfgBlock block) {
    return getExpectation(block).expectedPredecessorIds;
  }

  Set<String> expectedLiveInVariables(CfgBlock block) {
    return getExpectation(block).expectedLiveInVariables;
  }

  Set<String> expectedLiveOutVariables(CfgBlock block) {
    return getExpectation(block).expectedLiveOutVariables;
  }

  Set<String> expecteInDefVariables(CfgBlock block) {
    return getExpectation(block).expectedInDefVariables;
  }

  Set<String> expecteOutDefVariables(CfgBlock block) {
    return getExpectation(block).expectedOutDefVariables;
  }

  Set<String> expectedGenVariables(CfgBlock block) {
    return getExpectation(block).expectedGenVariables;
  }

  Set<String> expectedKilledVariables(CfgBlock block) {
    return getExpectation(block).expectedKilledVariables;
  }

  int expectedNumberOfElements(CfgBlock block) {
    return getExpectation(block).expectedNumberOfElements;
  }

  private BlockExpectation getExpectation(CfgBlock block) {
    return expectations.get(testId(block));
  }

  private BlockExpectation createExpectation(CfgBlock cfgBlock, String testId) {
    testIds.put(cfgBlock, testId);
    BlockExpectation expectation = new BlockExpectation();
    expectations.put(testId, expectation);
    return expectation;
  }

  BlockExpectation createEmptyBlockExpectation() {
    BlockExpectation blockExpectation = new BlockExpectation();
    emptyBlockExpectations.add(blockExpectation);
    return blockExpectation;
  }

  class BlockExpectation {
    final List<String> expectedSuccessorIds = new ArrayList<>();
    private String expectedSyntacticSuccessor = null;
    final List<String> expectedPredecessorIds = new ArrayList<>();
    private int expectedNumberOfElements = -1;
    private final Set<String> expectedInDefVariables = new HashSet<>();
    private final Set<String> expectedOutDefVariables = new HashSet<>();
    private final Set<String> expectedLiveInVariables = new HashSet<>();
    private final Set<String> expectedLiveOutVariables = new HashSet<>();
    private final Set<String> expectedGenVariables = new HashSet<>();
    private final Set<String> expectedKilledVariables = new HashSet<>();


    BlockExpectation withSuccessorsIds(String... ids) {
      Collections.addAll(expectedSuccessorIds, ids);
      return this;
    }

    BlockExpectation withPredecessorIds(String... ids) {
      Collections.addAll(expectedPredecessorIds, ids);
      return this;
    }

    BlockExpectation withElementNumber(int elementNumber) {
      expectedNumberOfElements = elementNumber;
      return this;
    }

    BlockExpectation withSyntacticSuccessor(@Nullable String syntacticSuccessor) {
      expectedSyntacticSuccessor = syntacticSuccessor;
      return this;
    }

    BlockExpectation withDefInVariables(String... ids) {
      Collections.addAll(expectedInDefVariables, ids);
      return this;
    }

    BlockExpectation withDefOutVariables(String... ids) {
      Collections.addAll(expectedOutDefVariables, ids);
      return this;
    }

    BlockExpectation withLiveInVariables(String... ids) {
      Collections.addAll(expectedLiveInVariables, ids);
      return this;
    }

    BlockExpectation withLiveOutVariables(String... ids) {
      Collections.addAll(expectedLiveOutVariables, ids);
      return this;
    }

    BlockExpectation withGenVariables(String... ids) {
      Collections.addAll(expectedGenVariables, ids);
      return this;
    }

    BlockExpectation withKilledVariables(String... ids) {
      Collections.addAll(expectedKilledVariables, ids);
      return this;
    }

    boolean matchesBlock(CfgBlock block) {
      return collectionsEquals(expectedSuccessorIds, blockIds(block.successors()))
        && collectionsEquals(expectedPredecessorIds, blockIds(block.predecessors()));
    }

    private boolean collectionsEquals(Collection<?> collection1, Collection<?> collection2) {
      return collection1.size() == collection2.size() && collection1.containsAll(collection2);
    }
  }

  /**
   * The expected structure for each basic block is encoded in a function call
   * See {@link ControlFlowGraphTest} for details
   */
  private static class Parser {

    static ExpectedCfgStructure parse(Set<CfgBlock> blocks) {
      ExpectedCfgStructure result = new ExpectedCfgStructure();

      for (CfgBlock block : blocks) {
        if (block instanceof PythonCfgEndBlock) {
          result.createExpectation(block, "END");
          continue;
        }

        List<Tree> elements = block.elements();
        if (elements.isEmpty() || elements.stream().allMatch(element -> element.is(Tree.Kind.PARAMETER))) {
          continue;
        }

        CallExpression callExpression = getBlockFunctionCall(elements);
        if (callExpression == null) {
          throw new UnsupportedOperationException("CFG Block metadata must be the first statement in the block.");
        }

        Optional<CallExpression> otherCallWithSuccArg = callsWithSuccArg(elements.subList(1, elements.size())).findFirst();
        if (otherCallWithSuccArg.isPresent()) {
          throw new UnsupportedOperationException("Found block declaration which is not at the beginning of a block: " + getValue(otherCallWithSuccArg.get().callee()));
        }

        String id = getValue(callExpression.callee());
        if (id == null) {
          throw new UnsupportedOperationException("CFG Block metadata is not in expected format");
        }
        BlockExpectation expectation = result.createExpectation(block, id);
        for (Argument arg : callExpression.arguments()) {
          if (arg.is(Tree.Kind.REGULAR_ARGUMENT)) {
            RegularArgument argument = (RegularArgument) arg;
            Name name = argument.keywordArgument();
            if (name == null) {
              throw new UnsupportedOperationException("The arguments of block function call must be keyword arguments");
            }
            Expression expression = argument.expression();
            if (isNameWithValue(name, "succ")) {
              expectation.withSuccessorsIds(names(expression));
            } else if (isNameWithValue(name, "pred")) {
              expectation.withPredecessorIds(names(expression));
            } else if (isNameWithValue(name, "elem")) {
              expectation.withElementNumber(Integer.parseInt(getValue(expression)));
            } else if (isNameWithValue(name, "syntSucc")) {
              expectation.withSyntacticSuccessor(getValue(expression));
            } else if (isNameWithValue(name, "gen")) {
              expectation.withGenVariables(getVariableStrings(expression));
            } else if (isNameWithValue(name, "kill")) {
              expectation.withKilledVariables(getVariableStrings(expression));
            } else if (isNameWithValue(name, "liveIn")) {
              expectation.withLiveInVariables(getVariableStrings(expression));
            } else if (isNameWithValue(name, "liveOut")) {
              expectation.withLiveOutVariables(getVariableStrings(expression));
            } else if (isNameWithValue(name, "defIn")) {
              expectation.withDefInVariables(getVariableStrings(expression));
            } else if (isNameWithValue(name, "defOut")) {
              expectation.withDefOutVariables(getVariableStrings(expression));
            }
          }
        }
      }

      return result;
    }

    private static CallExpression getBlockFunctionCall(List<Tree> elements) {
      Tree firstElement = elements.get(0);

      Expression expression;
      if (firstElement instanceof ExpressionStatement expressionStatement) {
        expression = expressionStatement.expressions().get(0);
      } else if (firstElement instanceof Expression firstElementExpression) {
        expression = firstElementExpression;
      } else if (firstElement instanceof ForStatement forStatement) {
        expression = forStatement.expressions().get(0);
      } else if (firstElement instanceof ExceptClause exceptClause) {
        expression = exceptClause.exception();
      } else {
        return null;
      }

      if (!(expression instanceof CallExpression call)) {
        return null;
      }
      if (call.arguments().isEmpty()) {
        return null;
      }
      return call;
    }

    private static Stream<CallExpression> callsWithSuccArg(List<Tree> elements) {
      return elements.stream()
        .map(e -> e.is(Tree.Kind.EXPRESSION_STMT) ? ((ExpressionStatement) e).expressions().get(0) : e)
        .filter(e -> e.is(Tree.Kind.CALL_EXPR))
        .map(CallExpression.class::cast)
        .filter(c -> c.arguments().stream()
          .filter(arg -> arg.is(Tree.Kind.REGULAR_ARGUMENT))
          .map(RegularArgument.class::cast)
          .anyMatch(arg -> isNameWithValue(arg.keywordArgument(), "succ")));
    }

    private static String[] names(Tree tree) {
      return getStringList(tree).toArray(new String[] {});
    }

    private static String[] getVariableStrings(Tree tree) {
      return getStringList(tree).stream().map(s -> s.substring(1)).toList().toArray(new String[] {});
    }

    private static List<String> getStringList(Tree tree) {
      if (tree instanceof ListLiteral listLiteral) {
        return listLiteral.elements().expressions().stream().map(e -> getValue(e)).toList();
      } else {
        throw new UnsupportedOperationException("Expecting list literal, got '" + tree.toString() + "'");
      }
    }

    private static boolean isNameWithValue(@Nullable Tree tree, String s) {
      return tree != null &&
        tree.is(Tree.Kind.NAME) &&
        ((Name) tree).name().equalsIgnoreCase(s);
    }

    private static String getValue(Tree tree) {
      if (tree.is(Tree.Kind.NUMERIC_LITERAL)) {
        return ((NumericLiteral) tree).valueAsString();
      }
      if (tree.is(Tree.Kind.NAME)) {
        return ((Name) tree).name();
      }
      throw new IllegalArgumentException("Cannot get value from tree");
    }
  }
}
