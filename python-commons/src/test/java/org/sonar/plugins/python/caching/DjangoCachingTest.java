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

import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.sonar.plugins.python.api.DjangoViewInfo;
import org.sonar.python.caching.CacheContextImpl;
import org.sonar.python.caching.PythonReadCacheImpl;
import org.sonar.python.caching.PythonWriteCacheImpl;

import static org.assertj.core.api.Assertions.assertThat;

class DjangoCachingTest {

  private static DjangoCaching cachingWith(TestWriteCache write, TestReadCache read) {
    var ctx = new CacheContextImpl(true, new PythonWriteCacheImpl(write), new PythonReadCacheImpl(read));
    return new DjangoCaching(ctx);
  }

  @Test
  void write_and_read_round_trip() {
    var write = new TestWriteCache();
    var read = new TestReadCache();
    var caching = cachingWith(write, read);

    Map<String, DjangoViewInfo> views = Map.of(
      "myapp.views.article_detail", new DjangoViewInfo(Set.of("article/<int:pk>/")),
      "myapp.views.list_view", new DjangoViewInfo(Set.of()));

    caching.writeDjangoViews("myapp:urls.py", views);

    // Promote written bytes into the read cache for the round-trip
    var read2 = new TestReadCache().putAll(write);
    var caching2 = cachingWith(new TestWriteCache(), read2);

    Map<String, DjangoViewInfo> restored = caching2.readDjangoViews("myapp:urls.py");
    assertThat(restored).isNotNull()
    .containsKey("myapp.views.article_detail")
    .containsKey("myapp.views.list_view");
    assertThat(restored.get("myapp.views.article_detail").urlPatterns()).containsExactly("article/<int:pk>/");
    assertThat(restored.get("myapp.views.list_view").urlPatterns()).isEmpty();
  }

  @Test
  void write_skips_empty_map() {
    var write = new TestWriteCache();
    var caching = cachingWith(write, new TestReadCache());

    caching.writeDjangoViews("myapp:urls.py", Map.of());

    // Nothing written when the map is empty
    assertThat(write.getData()).isEmpty();
  }

  @Test
  void read_returns_null_when_key_absent() {
    var caching = cachingWith(new TestWriteCache(), new TestReadCache());

    assertThat(caching.readDjangoViews("nonexistent:file.py")).isNull();
  }

  @Test
  void patterns_with_delimiter_characters_survive_round_trip() {
    // re_path patterns routinely contain commas, semicolons, and pipes
    var write = new TestWriteCache();
    var caching = cachingWith(write, new TestReadCache());

    String tricky = "^comments/([0-9]{1,4});pipe|test/$";
    caching.writeDjangoViews("app:urls.py", Map.of(
      "app.views.comment", new DjangoViewInfo(Set.of(tricky))));

    var caching2 = cachingWith(new TestWriteCache(), new TestReadCache().putAll(write));
    Map<String, DjangoViewInfo> restored = caching2.readDjangoViews("app:urls.py");

    assertThat(restored).isNotNull();
    assertThat(restored.get("app.views.comment").urlPatterns()).containsExactly(tricky);
  }

  @Test
  void copy_from_previous_propagates_registrar_key() {
    var write1 = new TestWriteCache();
    var read1 = new TestReadCache();
    var caching1 = cachingWith(write1, read1);
    caching1.writeDjangoViews("app:urls.py", Map.of(
      "app.views.foo", new DjangoViewInfo(Set.of("foo/"))));

    // Simulate next analysis: previous write becomes the read cache
    var write2 = new TestWriteCache().bind(new TestReadCache().putAll(write1));
    var caching2 = cachingWith(write2, new TestReadCache().putAll(write1));
    caching2.copyFromPrevious("app:urls.py");

    // The key should have been copied into write2
    assertThat(write2.getData()).containsKey(DjangoCaching.DJANGO_VIEWS_CACHE_KEY_PREFIX + "app:urls.py");
  }

  @Test
  void copy_from_previous_is_no_op_when_key_absent() {
    var write = new TestWriteCache();
    var caching = cachingWith(write, new TestReadCache());

    // Should not throw; nothing to copy
    caching.copyFromPrevious("nonexistent:file.py");

    assertThat(write.getData()).isEmpty();
  }
}
