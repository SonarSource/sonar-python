package org.sonar.python.types;

import com.google.gson.JsonDeserializationContext;
import com.google.gson.JsonDeserializer;
import com.google.gson.JsonElement;
import com.google.gson.JsonParseException;
import java.lang.reflect.Type;

public class PyTypeDetailedInfoDeserializer implements JsonDeserializer<PyTypeDetailedInfo> {
  @Override
  public PyTypeDetailedInfo deserialize(JsonElement jsonElement, Type type, JsonDeserializationContext jsonDeserializationContext) throws JsonParseException {
    return new PyTypeDetailedInfo(jsonElement.getAsString());
  }
}
