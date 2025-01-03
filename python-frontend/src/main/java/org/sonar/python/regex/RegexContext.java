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
package org.sonar.python.regex;

import javax.annotation.Nullable;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.tree.StringElement;
import org.sonar.plugins.python.api.tree.Tree;
import org.sonarsource.analyzer.commons.regex.RegexParseResult;
import org.sonarsource.analyzer.commons.regex.ast.FlagSet;
import org.sonarsource.analyzer.commons.regex.ast.RegexSyntaxElement;

public interface RegexContext {

  RegexParseResult regexForStringElement(StringElement stringElement, FlagSet flagSet);

  PythonCheck.PreciseIssue addIssue(Tree element, @Nullable String message);

  PythonCheck.PreciseIssue addIssue(RegexSyntaxElement element, @Nullable String message);
}
