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
package org.sonar.plugins.python.api.tree;

import com.sonar.sslr.api.TokenType;
import java.util.List;
import org.sonar.plugins.python.api.PythonLine;

public interface Token extends Tree {

  String value();

  int line();

  int column();

  List<Trivia> trivia();

  TokenType type();

  PythonLine pythonLine();

  int pythonColumn();

  int includedEscapeChars();

  int valueLength();

  boolean isCompressed();

}
