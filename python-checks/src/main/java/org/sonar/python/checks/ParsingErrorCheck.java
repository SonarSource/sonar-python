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

import com.sonar.sslr.api.RecognitionException;
import org.sonar.check.Rule;
import org.sonar.plugins.python.api.IssueLocation;
import org.sonar.plugins.python.api.PythonCheck;
import org.sonar.plugins.python.api.PythonVisitorContext;

@Rule(key = "ParsingError")
public class ParsingErrorCheck implements PythonCheck {

  @Override
  public void scanFile(PythonVisitorContext context) {
    RecognitionException parsingException = context.parsingException();
    if (parsingException != null) {
      context.addIssue(new PreciseIssue(this, IssueLocation.atLineLevel(parsingException.getMessage(), parsingException.getLine(), context.pythonFile())));
    }
  }

}
