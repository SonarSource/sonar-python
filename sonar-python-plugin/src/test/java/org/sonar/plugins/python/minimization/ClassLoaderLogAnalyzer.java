/*
 * SonarQube Python Plugin
 * Copyright (C) 2011-2019 SonarSource SA
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
package org.sonar.plugins.python.minimization;

import com.google.common.io.Files;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

public class ClassLoaderLogAnalyzer {

  private static final List<Jar> JARS_TO_FILTER = new ArrayList<>(Arrays.asList(
    new Jar("com.jetbrains.pycharm", "extensions"),
    new Jar("org.jetbrains.intellij.deps", "trove4j"),
    new Jar("org.jetbrains.kotlin", "kotlin-stdlib"),
    new Jar("com.jetbrains.pycharm", "platform-impl"),
    new Jar("com.jetbrains.pycharm", "pycharm-pydev"),
    new Jar("com.jetbrains.pycharm", "util").resource("misc/registry.properties"),
    new Jar("com.jetbrains.pycharm", "resources_en").resource("com/jetbrains/python/PyBundle.properties"),
    new Jar("com.jetbrains.pycharm", "openapi"),
    new Jar("com.jetbrains.pycharm", "platform-api"),
    new Jar("com.jetbrains.pycharm", "pycharm"),
    new Jar("com.jetbrains.pycharm", "jps-model")
  ));

  public static void main(String[] args) throws IOException {
    File file = new File("sonar-python-plugin/target/class-logs.txt");
    String fileContent = Files.toString(file, StandardCharsets.UTF_8);
    List<Line> lines = Arrays.stream(fileContent.split("\\n"))
      .map(Line::parse)
      .filter(Objects::nonNull)
      .collect(Collectors.toList());

    Map<String, List<Line>> classesBySource = new TreeMap<>(lines.stream()
      .collect(Collectors.groupingBy(Line::source)));

    for (Jar jar : JARS_TO_FILTER) {
      List<Line> classes = classesBySource.entrySet().stream()
        .filter(e -> e.getKey().contains("/" + jar.groupId.replace('.', '/') + "/" + jar.artifactId + "/"))
        .map(Map.Entry::getValue)
        .findFirst()
        .orElse(Collections.emptyList());

      System.out.println(
        "<filter>\n" +
        "  <artifact>" + jar.groupId + ":" + jar.artifactId + "</artifact>\n" +
        "  <includes>");

      Map<String, List<Line>> classesByPackage = new TreeMap<>(classes.stream().collect(Collectors.groupingBy(Line::packageName)));
      for (Map.Entry<String, List<Line>> entry : classesByPackage.entrySet()) {
        List<Line> packageLines = entry.getValue();
        if (packageLines.size() > 50) {
          System.out.println("    <include>" + entry.getKey().replaceAll("\\.", "/") + "/*.class</include>");
        } else {
          packageLines.stream()
            .map(Line::classFileName)
            .sorted()
            // Kotlin class names may contain `$$`. This is not handled correctly by Maven Shade plugin, so we replace
            // `$$` with `??`
            .map(l -> "    <include>" + l.replaceAll("\\$\\$", "??") + "</include>")
            .forEach(System.out::println);
        }
      }

      jar.resources.stream()
        .map(l -> "    <include>" + l + "</include>")
        .forEach(System.out::println);

      System.out.println("  </includes>");
      System.out.println("</filter>");
    }

  }

  private static class Jar {

    private final String groupId;
    private final String artifactId;
    private final List<String> resources = new ArrayList<>();

    private Jar(String groupId, String artifactId) {
      this.groupId = groupId;
      this.artifactId = artifactId;
    }

    Jar resource(String resource) {
      resources.add(resource);
      return this;
    }
  }

  private static class Line {

    private static final Pattern PATTERN_JDK8 = Pattern.compile("\\[Loaded (.*) from (.*/.*)]");
    private static final Pattern PATTERN_JDK11 = Pattern.compile("\\[.*]\\[info]\\[class,load] (.*) source: (.*/.*)");

    String classFileName() {
      return fullClassName.replaceAll("\\.", "/") + ".class";
    }

    String fullClassName;
    String source;

    Line(String fullClassName, String source) {
      this.fullClassName = fullClassName;
      this.source = source;
    }

    static Line parse(String line) {
      Pattern pattern = line.startsWith("[Loaded ") ? PATTERN_JDK8 : PATTERN_JDK11;
      Matcher matcher = pattern.matcher(line);
      if (!matcher.matches()) {
        return null;
      }
      return new Line(matcher.group(1), matcher.group(2));
    }

    String source() {
      return source;
    }

    String packageName() {
      return fullClassName.substring(0, fullClassName.lastIndexOf('.'));
    }
  }

}
