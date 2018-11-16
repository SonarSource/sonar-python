/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2018 SonarSource SA
 * mailto:info AT sonarsource DOT com
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3 of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */
package org.sonar.python;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.Set;
import java.util.stream.Collectors;

public enum PythonBuiltinFunctions {

  INSTANCE;

  private static final Set<String> BUILTINS = loadBuiltinNames(PythonBuiltinFunctions.class.getResourceAsStream("builtins.txt"));

  public static boolean contains(String name) {
    return BUILTINS.contains(name);
  }

  static Set<String> loadBuiltinNames(InputStream resourceStream) {
    try (BufferedReader reader = new BufferedReader(new InputStreamReader(resourceStream, StandardCharsets.UTF_8))) {
      return reader.lines()
        .map(String::trim)
        .filter(s -> !s.startsWith("#"))
        .filter(s -> !s.isEmpty())
        .collect(Collectors.toSet());
    } catch (IOException | UncheckedIOException e) {
      throw new IllegalStateException("Cannot read input stream", e);
    }

  }

}
