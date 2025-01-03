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
package org.sonar.plugins.python.api;

import com.sonar.sslr.api.AstNode;
import org.sonar.plugins.python.api.tree.FileInput;
import org.sonar.python.parser.PythonParser;
import org.sonar.python.tree.PythonTreeMaker;

/**
 * Use this class to parse Python module
 */
public class Parser {

  private PythonParser astParser = PythonParser.create();

  public FileInput parse(String code)  {
    AstNode astNode = astParser.parse(code);
    return new PythonTreeMaker().fileInput(astNode);
  }

}
