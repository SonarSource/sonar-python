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
package org.sonar.plugins.python.caching;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import java.lang.reflect.Type;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.Set;
import javax.annotation.CheckForNull;
import org.sonar.plugins.python.api.DjangoViewInfo;
import org.sonar.plugins.python.api.caching.CacheContext;

public class DjangoCaching {

  static final String DJANGO_VIEWS_CACHE_KEY_PREFIX = "python:django_views:";

  private static final Gson GSON = new Gson();
  private static final Type MAP_TYPE = new TypeToken<Map<String, Set<String>>>() {}.getType();

  private final CacheContext cacheContext;

  public DjangoCaching(CacheContext cacheContext) {
    this.cacheContext = cacheContext;
  }

  public void writeDjangoViews(String fileKey, Map<String, DjangoViewInfo> views) {
    write(registrarCacheKey(fileKey), views);
  }

  @CheckForNull
  public Map<String, DjangoViewInfo> readDjangoViews(String fileKey) {
    return read(registrarCacheKey(fileKey));
  }

  public void copyFromPrevious(String fileKey) {
    copyIfPresent(registrarCacheKey(fileKey));
  }

  private void write(String cacheKey, Map<String, DjangoViewInfo> views) {
    if (views.isEmpty()) {
      return;
    }
    // Serialize as { "fqn": ["pattern1", "pattern2"], ... }
    // DjangoViewInfo is not directly serializable, so we convert to Map<String, Set<String>> first.
    Map<String, Set<String>> raw = new java.util.LinkedHashMap<>();
    views.forEach((fqn, info) -> raw.put(fqn, info.urlPatterns()));
    cacheContext.getWriteCache().write(cacheKey, GSON.toJson(raw).getBytes(StandardCharsets.UTF_8));
  }

  @CheckForNull
  private Map<String, DjangoViewInfo> read(String cacheKey) {
    byte[] bytes = cacheContext.getReadCache().readBytes(cacheKey);
    if (bytes == null) {
      return null;
    }
    Map<String, Set<String>> raw = GSON.fromJson(new String(bytes, StandardCharsets.UTF_8), MAP_TYPE);
    if (raw == null) {
      return null;
    }
    Map<String, DjangoViewInfo> result = new java.util.LinkedHashMap<>();
    raw.forEach((fqn, patterns) -> result.put(fqn, new DjangoViewInfo(patterns != null ? patterns : Set.of())));
    return result;
  }

  private void copyIfPresent(String cacheKey) {
    if (cacheContext.getReadCache().contains(cacheKey)) {
      cacheContext.getWriteCache().copyFromPrevious(cacheKey);
    }
  }

  private static String registrarCacheKey(String fileKey) {
    return DJANGO_VIEWS_CACHE_KEY_PREFIX + fileKey.replace('\\', '/');
  }
}
