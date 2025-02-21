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
package org.sonar.plugins.python.dependency;

import com.fasterxml.jackson.annotation.JsonSetter;
import com.fasterxml.jackson.annotation.Nulls;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.dataformat.toml.TomlMapper;
import com.fasterxml.jackson.datatype.jdk8.Jdk8Module;
import java.io.IOException;
import java.io.Reader;
import java.util.List;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import org.sonar.plugins.python.dependency.model.Dependencies;
import org.sonar.plugins.python.dependency.model.Dependency;

public class PyProjectTomlParser {
  private static final Pattern IDENTIFIER_PATTERN = Pattern.compile("^[a-zA-Z0-9-_.]+");

  private PyProjectTomlParser() {
  }

  public static Dependencies parse(Reader reader) {
    try {
      PyProjectToml pyProjectToml = readPyProjectToml(reader);
      return convertToDependenciesModel(pyProjectToml);
    } catch (IOException e) {
      return new Dependencies(Set.of());
    }
  }

  private static PyProjectToml readPyProjectToml(Reader reader) throws IOException {
    return new TomlMapper()
      .registerModule(new Jdk8Module())
      .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)
      .readValue(reader, PyProjectToml.class);
  }

  private static Dependencies convertToDependenciesModel(PyProjectToml pyProjectToml) {
    if(pyProjectToml.project() == null) {
      return new Dependencies(Set.of());
    } else {
      var dependencies = pyProjectToml.project().dependencies().stream()
        .flatMap(PyProjectTomlParser::parseDependency)
        .collect(Collectors.toSet());
      return new Dependencies(dependencies);
    }
  }

  private static Stream<Dependency> parseDependency(String dependency) {
    Matcher matcher = IDENTIFIER_PATTERN.matcher(dependency);
    if(matcher.find()) {
      return Stream.of(new Dependency(matcher.group()));
    } else {
      return Stream.empty();
    }
  }


  private record PyProjectToml(@Nullable Project project) {}
  private record Project(@Nonnull @JsonSetter(nulls = Nulls.AS_EMPTY) List<String> dependencies) {}
}
