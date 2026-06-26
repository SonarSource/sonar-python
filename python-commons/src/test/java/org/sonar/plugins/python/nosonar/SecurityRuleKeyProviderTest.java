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
package org.sonar.plugins.python.nosonar;

import java.io.IOException;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;
import java.util.Map;
import java.util.jar.JarEntry;
import java.util.jar.JarOutputStream;
import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.Test;

class SecurityRuleKeyProviderTest {

  @Test
  void test_loads_vulnerability_as_security() {
    var provider = new SecurityRuleKeyProvider();
    Assertions.assertThat(provider.isSecurityRule("TestVulnerability")).isTrue();
  }

  @Test
  void test_code_smell_is_not_security() {
    var provider = new SecurityRuleKeyProvider();
    Assertions.assertThat(provider.isSecurityRule("TestCodeSmell")).isFalse();
  }

  @Test
  void test_unknown_rule_is_not_security() {
    var provider = new SecurityRuleKeyProvider();
    Assertions.assertThat(provider.isSecurityRule("S9999")).isFalse();
    Assertions.assertThat(provider.isSecurityRule("NonExistentRule")).isFalse();
  }

  @Test
  void test_json_without_type_field_and_malformed_json_are_not_security() {
    var provider = new SecurityRuleKeyProvider();
    Assertions.assertThat(provider.isSecurityRule("TestNoType")).isFalse();
    Assertions.assertThat(provider.isSecurityRule("TestMalformedJson")).isFalse();
  }

  @Test
  void test_jar_url_with_null_entry_name_is_skipped() throws Exception {
    var jarPath = createTempJar(Map.of(
      "org/sonar/l10n/py/rules/python/", "",
      "org/sonar/l10n/py/rules/python/JarVuln.json", "{\"type\": \"VULNERABILITY\"}"
    ));
    try {
      // "jar:...!/" with nothing after the separator gives getEntryName() == null
      var noEntryUrl = new URL("jar:" + jarPath.toUri() + "!/");
      ClassLoader loader = new ClassLoader(null) {
        @Override
        public Enumeration<URL> getResources(String name) {
          return Collections.enumeration(List.of(noEntryUrl));
        }
      };
      var provider = new SecurityRuleKeyProvider(loader);
      Assertions.assertThat(provider.isSecurityRule("JarVuln")).isFalse();
    } finally {
      Files.deleteIfExists(jarPath);
    }
  }

  @Test
  void test_jar_without_directory_entry_yields_no_keys() throws Exception {
    // JARs without an explicit directory entry for the rule path are not discovered —
    // ClassLoader.getResources("dir/") requires a matching directory entry in the JAR.
    var jarPath = createTempJar(Map.of(
      "org/sonar/l10n/py/rules/python/JarVulnerability.json", "{\"type\": \"VULNERABILITY\"}"
    ));
    try {
      try (var jarClassLoader = new URLClassLoader(new URL[]{jarPath.toUri().toURL()}, null)) {
        var provider = new SecurityRuleKeyProvider(jarClassLoader);
        Assertions.assertThat(provider.isSecurityRule("JarVulnerability")).isFalse();
      }
    } finally {
      Files.deleteIfExists(jarPath);
    }
  }

  @Test
  void test_loads_from_jar_classpath() throws Exception {
    var jarPath = createTempJar(Map.of(
      "org/sonar/l10n/py/rules/python/", "",
      "org/sonar/l10n/py/rules/python/JarVulnerability.json", "{\"type\": \"VULNERABILITY\"}",
      "org/sonar/l10n/py/rules/python/JarCodeSmell.json", "{\"type\": \"CODE_SMELL\"}"
    ));
    try {
      try (var jarClassLoader = new URLClassLoader(new URL[]{jarPath.toUri().toURL()}, null)) {
        var provider = new SecurityRuleKeyProvider(jarClassLoader);
        var urls = jarClassLoader.getResources("org/sonar/l10n/py/rules/python/");
        Assertions.assertThat(urls.nextElement().toString()).startsWith("jar:file:");
        Assertions.assertThat(provider.isSecurityRule("JarVulnerability")).isTrue();
        Assertions.assertThat(provider.isSecurityRule("JarCodeSmell")).isFalse();
      }
    } finally {
      Files.deleteIfExists(jarPath);
    }
  }

  @Test
  void test_bad_jar_url_is_handled_gracefully() throws Exception {
    // A jar: URL pointing to a non-existent JAR exercises the catch block in scanDir
    var badJarUrl = new URL("jar:file:/nonexistent-path/missing.jar!/org/sonar/l10n/py/rules/python/");
    ClassLoader badLoader = new ClassLoader(null) {
      @Override
      public Enumeration<URL> getResources(String name) {
        return Collections.enumeration(List.of(badJarUrl));
      }
    };
    // Must not throw — error is logged at DEBUG and loading continues
    var provider = new SecurityRuleKeyProvider(badLoader);
    Assertions.assertThat(provider.isSecurityRule("anything")).isFalse();
  }

  @Test
  void test_file_url_pointing_to_non_directory_is_skipped() throws Exception {
    // A file: URL pointing to a regular file (not a directory) exercises the !isDirectory() early return
    String tmpdir = System.getenv("TMPDIR");
    Path tmpBase = (tmpdir != null) ? Path.of(tmpdir) : Path.of(System.getProperty("java.io.tmpdir"));
    var plainFile = Files.createTempFile(tmpBase, "not-a-dir", ".json");
    try {
      ClassLoader fileLoader = new ClassLoader(null) {
        @Override
        public Enumeration<URL> getResources(String name) throws IOException {
          return Collections.enumeration(List.of(plainFile.toUri().toURL()));
        }
      };
      var provider = new SecurityRuleKeyProvider(fileLoader);
      Assertions.assertThat(provider.isSecurityRule("anything")).isFalse();
    } finally {
      Files.deleteIfExists(plainFile);
    }
  }

  private static Path createTempJar(Map<String, String> entries) throws IOException {
    String tmpdir = System.getenv("TMPDIR");
    Path tmpBase = (tmpdir != null) ? Path.of(tmpdir) : Path.of(System.getProperty("java.io.tmpdir"));
    var jarPath = Files.createTempFile(tmpBase, "test-rules", ".jar");
    try (var jos = new JarOutputStream(Files.newOutputStream(jarPath))) {
      for (var entry : entries.entrySet()) {
        jos.putNextEntry(new JarEntry(entry.getKey()));
        jos.write(entry.getValue().getBytes(StandardCharsets.UTF_8));
        jos.closeEntry();
      }
    }
    return jarPath;
  }
}
