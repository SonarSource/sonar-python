package org.sonar.python.types;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.lang.reflect.Type;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

public class TypeContextReader {

  private final Gson gson;
  private final Type type;

  public static TypeContext fromJson(String json) {
    try (var reader = new StringReader(json)) {
      return new TypeContextReader().fromJson(reader);
    }
  }

  public TypeContextReader() {
    gson = new GsonBuilder()
      .registerTypeAdapter(PyTypeDetailedInfo.class, new PyTypeDetailedInfoDeserializer())
      .create();
    type = new TypeToken<Map<String, List<PyTypeInfo>>>() {
    }.getType();
  }

  public TypeContext fromJson(Path path) throws IOException {
    try (var reader = Files.newBufferedReader(path)) {
      return fromJson(reader);
    }
  }

  private TypeContext fromJson(Reader reader) {
    var files = gson.<Map<String, List<PyTypeInfo>>>fromJson(reader, type);
    return new TypeContext(files);
  }

}
