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

import java.util.List;

/**
 * Mapping Pattern
 *
 * <pre>
 *   case {"x": "foo", "y": "bar"}:
 *     ...
 * </pre>
 *
 * See https://docs.python.org/3/reference/compound_stmts.html#mapping-patterns
 */
public interface MappingPattern extends Pattern {

  Token lCurlyBrace();

  List<Pattern> elements();

  List<Token> commas();

  Token rCurlyBrace();
}
