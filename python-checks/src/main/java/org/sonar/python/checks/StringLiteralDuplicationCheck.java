/*
 * SonarQube Python Plugin
 * Copyright (C) SonarSource Sàrl
 * mailto:info AT sonarsource DOT com
 *
 * You can redistribute and/or modify this program under the terms of
 * the Sonar Source-Available License Version 1, as published by SonarSource Sàrl.
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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import org.sonar.check.Rule;
import org.sonar.check.RuleProperty;
import org.sonar.plugins.python.api.PythonVisitorCheck;
import org.sonar.plugins.python.api.quickfix.PythonQuickFix;
import org.sonar.plugins.python.api.symbols.Symbol;
import org.sonar.plugins.python.api.tree.Decorator;
import org.sonar.plugins.python.api.tree.ExpressionStatement;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.plugins.python.api.tree.FunctionDef;
import org.sonar.plugins.python.api.tree.Statement;
import org.sonar.plugins.python.api.tree.StatementList;
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.StringLiteral;
import org.sonar.plugins.python.api.tree.Token;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonar.plugins.python.api.tree.TypeAnnotation;
import org.sonar.python.checks.utils.Expressions;
import org.sonar.python.quickfix.TextEditUtils;
import org.sonar.python.tree.TreeUtils;

@Rule(key = "S1192")
public class StringLiteralDuplicationCheck extends PythonVisitorCheck {

  private static final Integer MINIMUM_LITERAL_LENGTH = 5;
  private static final int DEFAULT_THRESHOLD = 3;
  private static final Pattern BASIC_EXCLUSION_PATTERN = Pattern.compile("[_\\-a-zA-Z0-9]+");

  private static final Pattern FORMATTING_PATTERN = Pattern.compile("[0-9{} .\\-_%:dfrsymhYMHS<>]+");
  private static final Pattern COLOR_PATTERN = Pattern.compile("#[0-9a-fA-F]{6}");
  private static final String QUICK_FIX_MESSAGE = "Extract duplicated literal into constant '%s'";

  private static final String DEFAULT_CUSTOM_EXCLUSION_PATTERN = "";

  private Pattern customPattern = null;

  @RuleProperty(
    key = "threshold",
    description = "Number of times a literal must be duplicated to trigger an issue",
    defaultValue = "" + DEFAULT_THRESHOLD)
  public int threshold = DEFAULT_THRESHOLD;

  @RuleProperty(
    key = "exclusionRegex",
    description = "RegEx matching literals to exclude from triggering an issue",
    defaultValue = "")
  public String customExclusionRegex = DEFAULT_CUSTOM_EXCLUSION_PATTERN;

  private Map<String, List<StringLiteral>> literalsByValue = new HashMap<>();

  private boolean isCustomPatternInitialized = false;

  private Optional<Pattern> customExclusionPattern() {
    if (!isCustomPatternInitialized) {
      if (customExclusionRegex != null && !customExclusionRegex.isEmpty()) {
        try {
          customPattern = Pattern.compile(customExclusionRegex, Pattern.DOTALL);
        } catch (RuntimeException e) {
          throw new IllegalStateException("Unable to compile regular expression: " + customExclusionRegex, e);
        }
      }
      isCustomPatternInitialized = true;
    }
    return Optional.ofNullable(customPattern);
  }

  @Override
  public void visitFileInput(FileInput fileInput) {
    literalsByValue.clear();

    if (this.getContext().pythonFile().fileName().startsWith("test")) {
      return;
    }
    super.visitFileInput(fileInput);

    for (Map.Entry<String, List<StringLiteral>> entry : literalsByValue.entrySet()) {
      List<StringLiteral> occurrences = entry.getValue();
      int nbOfOccurrences = occurrences.size();
      if (nbOfOccurrences >= threshold) {
        StringLiteral first = occurrences.get(0);
        String message = String.format(
          "Define a constant instead of duplicating this literal %s %s times.",
          first.firstToken().value(),
          nbOfOccurrences);
        PreciseIssue issue = addIssue(first, message).withCost(nbOfOccurrences - 1);
        getQuickFix(fileInput, occurrences).ifPresent(issue::addQuickFix);
        occurrences.stream()
          .skip(1)
          .forEach(stringLiteral -> issue.secondary(stringLiteral, "Duplication"));
      }
    }
  }

  @Override
  public void visitExpressionStatement(ExpressionStatement expressionStatement) {
    // exclude docstrings
    if (!expressionStatement.expressions().get(0).is(Tree.Kind.STRING_LITERAL)) {
      super.visitExpressionStatement(expressionStatement);
    }
  }

  @Override
  public void visitStringLiteral(StringLiteral literal) {
    String value = Expressions.unescape(literal);
    boolean hasInterpolation = literal.stringElements().stream().anyMatch(StringElement::isInterpolated);
    boolean isExcluded = hasInterpolation
      || value.length() < MINIMUM_LITERAL_LENGTH
      || BASIC_EXCLUSION_PATTERN.matcher(value).matches()
      || FORMATTING_PATTERN.matcher(value).matches()
      || COLOR_PATTERN.matcher(value).matches()
      || matchesCustomExclusionPattern(value);
    if (!isExcluded) {
      String valueWithQuotes = TreeUtils.tokens(literal).stream().map(Token::value).collect(Collectors.joining());
      literalsByValue.computeIfAbsent(valueWithQuotes, key -> new ArrayList<>()).add(literal);
    }
  }

  private boolean matchesCustomExclusionPattern(String value) {
    return customExclusionPattern().map(p -> p.matcher(value).matches()).orElse(false);
  }

  @Override
  public void visitDecorator(Decorator decorator) {
    // Ignore literals in decorators
  }

  @Override
  public void visitTypeAnnotation(TypeAnnotation typeAnnotation) {
    // Ignore literals in type annotations
  }

  private static Optional<PythonQuickFix> getQuickFix(FileInput fileInput, List<StringLiteral> occurrences) {
    StatementList statementList = parentStatementList(occurrences.get(0));
    if (statementList == null || occurrences.stream().anyMatch(literal -> parentStatementList(literal) != statementList)) {
      return Optional.empty();
    }

    String literalSource = TreeUtils.treeToString(occurrences.get(0), false);
    if (literalSource == null) {
      return Optional.empty();
    }

    Statement insertionAnchor = insertionAnchor(statementList);
    if (insertionAnchor == null) {
      return Optional.empty();
    }

    String constantName = freshConstantName(fileInput, occurrences.get(0));
    return Optional.of(PythonQuickFix.newQuickFix(String.format(QUICK_FIX_MESSAGE, constantName))
      .addTextEdit(TextEditUtils.insertLineBefore(insertionAnchor, constantName + " = " + literalSource))
      .addTextEdit(occurrences.stream().map(literal -> TextEditUtils.replace(literal, constantName)).toList())
      .build());
  }

  private static StatementList parentStatementList(StringLiteral literal) {
    return TreeUtils.toOptionalInstanceOf(StatementList.class, TreeUtils.firstAncestorOfKind(literal, Tree.Kind.STATEMENT_LIST)).orElse(null);
  }

  private static Statement insertionAnchor(StatementList statementList) {
    if (statementList.statements().isEmpty()) {
      return null;
    }
    Statement firstStatement = statementList.statements().get(0);
    if (isDocstring(firstStatement) && statementList.statements().size() > 1) {
      return statementList.statements().get(1);
    }
    return firstStatement;
  }

  private static boolean isDocstring(Statement statement) {
    if (!statement.is(Tree.Kind.EXPRESSION_STMT)) {
      return false;
    }
    ExpressionStatement expressionStatement = (ExpressionStatement) statement;
    return !expressionStatement.expressions().isEmpty() && expressionStatement.expressions().get(0).is(Tree.Kind.STRING_LITERAL);
  }

  private static String freshConstantName(FileInput fileInput, StringLiteral literal) {
    String baseName = NamingConventionQuickFixUtils.toConstantCase(Expressions.unescape(literal));
    Set<String> namesInScope = namesInScope(fileInput, literal);

    String candidate = baseName;
    int suffix = 1;
    while (namesInScope.contains(candidate)) {
      candidate = baseName + "_" + suffix;
      suffix++;
    }
    return candidate;
  }

  private static Set<String> namesInScope(FileInput fileInput, StringLiteral literal) {
    FunctionDef functionDef = TreeUtils.toOptionalInstanceOf(FunctionDef.class, TreeUtils.firstAncestorOfKind(literal, Tree.Kind.FUNCDEF)).orElse(null);
    if (functionDef != null) {
      return functionDef.localVariables().stream().map(Symbol::name).collect(Collectors.toSet());
    }
    return fileInput.globalVariables().stream().map(Symbol::name).collect(Collectors.toSet());
  }
}
