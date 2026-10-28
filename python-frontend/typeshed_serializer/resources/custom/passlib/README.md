The Argon2 override preserves the parameter signature required by existing
security rules. The rest of Passlib is imported from the pinned historical
`types-passlib` distribution. There is deliberately no custom `hash.pyi`:
a partial `passlib.hash` module would shadow all of that distribution's public
hash-handler re-exports.

The Argon2 signature follows Typeshed revision
`b5c9d8d6770e09dff00648b79281702e111d4e01` and Passlib 1.7.4.
The upstream Typeshed license is included in this directory.
Regenerate descriptors with the supported serializer runner.
