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

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.File;
import java.io.IOException;
import java.net.JarURLConnection;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Collections;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.sonar.api.scanner.ScannerSide;
import org.sonarsource.api.sonarlint.SonarLintSide;

@ScannerSide
@SonarLintSide
public class SecurityRuleKeyProvider {

  private static final Logger LOG = LoggerFactory.getLogger(SecurityRuleKeyProvider.class);

  private static final String JSON_EXTENSION = ".json";

  private static final String[] RULE_RESOURCE_DIRS = {
    "org/sonar/l10n/py/rules/python/",
    "com/sonar/l10n/py/rules/python/"
  };

  private final Set<String> securityRuleKeys;

  public SecurityRuleKeyProvider() {
    this(SecurityRuleKeyProvider.class.getClassLoader());
  }

  SecurityRuleKeyProvider(ClassLoader classLoader) {
    securityRuleKeys = Collections.unmodifiableSet(loadSecurityRuleKeys(classLoader));
  }

  SecurityRuleKeyProvider(Set<String> securityRuleKeys) {
    this.securityRuleKeys = Collections.unmodifiableSet(new HashSet<>(securityRuleKeys));
  }

  public boolean isSecurityRule(String ruleKey) {
    return securityRuleKeys.contains(ruleKey);
  }

  private static Set<String> loadSecurityRuleKeys(ClassLoader classLoader) {
    var keys = new HashSet<String>();
    for (var dir : RULE_RESOURCE_DIRS) {
      try {
        Enumeration<URL> resources = classLoader.getResources(dir);
        while (resources.hasMoreElements()) {
          scanDir(resources.nextElement(), keys);
        }
      } catch (IOException e) {
        LOG.warn("Failed to scan security rule keys from classpath path: {}", dir, e);
      }
    }
    LOG.debug("Loaded {} security rule key(s)", keys.size());
    return keys;
  }

  private static void scanDir(URL dirUrl, Set<String> keys) {
    try {
      if ("jar".equals(dirUrl.getProtocol())) {
        scanJar(dirUrl, keys);
      } else if ("file".equals(dirUrl.getProtocol())) {
        scanFileDir(dirUrl, keys);
      }
    } catch (IOException | URISyntaxException | RuntimeException e) {
      LOG.debug("Could not scan {} for security rule keys: {}", dirUrl, e.getMessage());
    }
  }

  private static void scanJar(URL jarDirUrl, Set<String> keys) throws IOException {
    var urlConnection = jarDirUrl.openConnection();
    if (!(urlConnection instanceof JarURLConnection connection)) {
      LOG.debug("Skipping jar URL {} — unexpected connection type: {}", jarDirUrl, urlConnection.getClass().getName());
      return;
    }
    var entryPrefix = connection.getEntryName();
    if (entryPrefix == null) {
      LOG.debug("Skipping jar URL {} — no entry name", jarDirUrl);
      return;
    }
    connection.setUseCaches(false);
    try (var jar = connection.getJarFile()) {
      var entries = jar.entries();
      while (entries.hasMoreElements()) {
        var entry = entries.nextElement();
        if (!entry.isDirectory() && entry.getName().startsWith(entryPrefix)) {
          var suffix = entry.getName().substring(entryPrefix.length());
          if (!suffix.contains("/") && suffix.endsWith(JSON_EXTENSION)) {
            try (var is = jar.getInputStream(entry)) {
              var json = new String(is.readAllBytes(), StandardCharsets.UTF_8);
              if (isSecurityRuleJson(json)) {
                keys.add(extractRuleKey(entry.getName()));
              }
            }
          }
        }
      }
    }
  }

  private static void scanFileDir(URL dirUrl, Set<String> keys) throws IOException, URISyntaxException {
    var dir = new File(dirUrl.toURI());
    if (!dir.isDirectory()) {
      return;
    }
    var jsonFiles = dir.listFiles((d, name) -> name.endsWith(JSON_EXTENSION));
    if (jsonFiles == null) {
      return;
    }
    for (var jsonFile : jsonFiles) {
      var json = Files.readString(jsonFile.toPath(), StandardCharsets.UTF_8);
      if (isSecurityRuleJson(json)) {
        keys.add(extractRuleKey(jsonFile.getName()));
      }
    }
  }

  private static boolean isSecurityRuleJson(String json) {
    try {
      JsonObject obj = JsonParser.parseString(json).getAsJsonObject();
      var typeElement = obj.get("type");
      if (typeElement == null) {
        return false;
      }
      var type = typeElement.getAsString();
      return "VULNERABILITY".equals(type);
    } catch (Exception e) {
      return false;
    }
  }

  private static String extractRuleKey(String pathOrName) {
    var lastSlash = pathOrName.lastIndexOf('/');
    var filename = lastSlash >= 0 ? pathOrName.substring(lastSlash + 1) : pathOrName;
    return filename.substring(0, filename.length() - JSON_EXTENSION.length());
  }
}
