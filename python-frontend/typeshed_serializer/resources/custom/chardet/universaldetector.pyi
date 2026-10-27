# chardet 7.x still exports this compatibility name through __all__, which the
# serializer's private-import filtering omits. Keep the package root upstream.
from chardet.detector import UniversalDetector as UniversalDetector
