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
package org.sonar.python.checks.regex;

import java.util.Collections;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.tree.CallExpression;
import org.sonarsource.analyzer.commons.regex.RegexParseResult;
import org.sonarsource.analyzer.commons.regex.ast.CharacterTree;
import org.sonarsource.analyzer.commons.regex.ast.RegexBaseVisitor;

@Rule(key = "S6537")
public class OctalEscapeCheck extends AbstractRegexCheck {

  public static final String MESSAGE = "Consider replacing this octal escape sequence with a Unicode or hexadecimal sequence instead.";

  @Override
  public void checkRegex(RegexParseResult regexParseResult, CallExpression regexFunctionCall) {
    new CharacterFinder().visit(regexParseResult);
  }

  private class CharacterFinder extends RegexBaseVisitor {
    @Override
    public void visitCharacter(CharacterTree tree) {
      if (tree.isEscapeSequence() && tree.getText().matches("\\\\{1,2}\\d+")) {
        addIssue(tree, MESSAGE, null, Collections.emptyList());
      }
      super.visitCharacter(tree);
    }
  }
}
