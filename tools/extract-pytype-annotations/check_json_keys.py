import json
JSON_PATH = "/types/sources.json"
PATHS = ["biopython/Bio/Entrez/Parser.py",
         "buildbot-0.8.6p1/buildbot/changes/hgbuildbot.py",
         "buildbot-0.8.6p1/buildbot/status/build.py",
         "buildbot-0.8.6p1/buildbot/status/persistent_queue.py",
         "buildbot-0.8.6p1/buildbot/status/persistent_queue.py",
         "buildbot-0.8.6p1/buildbot/status/persistent_queue.py",
         "buildbot-0.8.6p1/buildbot/status/web/feeds.py",
         "buildbot-0.8.6p1/buildbot/status/web/waterfall.py",
         "buildbot-0.8.6p1/buildbot/status/words.py",
         "buildbot-0.8.6p1/buildbot/steps/python_twisted.py",
         "buildbot-0.8.6p1/buildbot/steps/shell.py",
         "django-2.2.3/django/core/files/locks.py",
         "django-2.2.3/django/core/files/locks.py",
         "django-cms-3.7.1/cms/utils/placeholder.py",
         "docker-compose-1.24.1/compose/cli/main.py",
         "docker-compose-1.24.1/compose/cli/main.py",
         "mypy-0.782/misc/actions_stubs.py",
         "mypy-0.782/test-data/stdlib-samples/3.2/test/test_set.py",
         "mypy-0.782/test-data/stdlib-samples/3.2/test/test_set.py",
         "mypy-0.782/test-data/stdlib-samples/3.2/test/test_set.py",
         "mypy-0.782/test-data/stdlib-samples/3.2/test/test_set.py",
         "mypy-0.782/test-data/stdlib-samples/3.2/test/test_set.py",
         "mypy-0.782/test-data/stdlib-samples/3.2/test/test_set.py",
         "mypy-0.782/test-data/stdlib-samples/3.2/test/test_set.py",
         "numpy-1.16.4/numpy/distutils/exec_command.py",
         "numpy-1.16.4/numpy/distutils/exec_command.py",
         "numpy-1.16.4/numpy/f2py/f2py2e.py",
         "pecos/examples/qp2q/models/indices.py",
         "S5706.py",
         "tensorflow/python/data/kernel_tests/interleave_test.py",
         "tensorflow/python/data/kernel_tests/interleave_test.py",
         "tensorflow/python/keras/engine/training_distributed.py",
         "tensorflow/python/keras/engine/training_distributed.py",
         "tensorflow/python/keras/layers/normalization.py",
         "tensorflow/python/keras/layers/wrappers.py",
         "tensorflow/python/ops/init_ops_v2.py",
         "tensorflow/python/ops/init_ops_v2.py",
         "tensorflow/python/ops/init_ops_v2.py",
         "tensorflow/python/ops/math_grad.py",
         "tensorflow/python/ops/math_grad.py",
         "tensorflow/python/training/saver.py",
         "tensorflow/tools/compatibility/tf_upgrade_v2.py",
         "twisted-12.1.0/doc/core/examples/longex.py",
         "twisted-12.1.0/doc/core/examples/pbbenchserver.py",
         "twisted-12.1.0/doc/core/examples/pbecho.py",
         "twisted-12.1.0/twisted/conch/test/test_checkers.py",
         "twisted-12.1.0/twisted/conch/test/test_checkers.py",
         "twisted-12.1.0/twisted/conch/test/test_checkers.py",
         "twisted-12.1.0/twisted/conch/test/test_checkers.py",
         "twisted-12.1.0/twisted/conch/test/test_checkers.py",
         "twisted-12.1.0/twisted/conch/test/test_checkers.py",
         "twisted-12.1.0/twisted/conch/test/test_checkers.py",
         "twisted-12.1.0/twisted/conch/test/test_checkers.py",
         "twisted-12.1.0/twisted/conch/test/test_checkers.py",
         "twisted-12.1.0/twisted/conch/test/test_checkers.py",
         "twisted-12.1.0/twisted/mail/imap4.py",
         "twisted-12.1.0/twisted/mail/imap4.py",
         "twisted-12.1.0/twisted/mail/imap4.py",
         "twisted-12.1.0/twisted/mail/imap4.py",
         "twisted-12.1.0/twisted/mail/imap4.py",
         "twisted-12.1.0/twisted/mail/imap4.py",
         "twisted-12.1.0/twisted/mail/imap4.py",
         "twisted-12.1.0/twisted/mail/imap4.py",
         "twisted-12.1.0/twisted/mail/imap4.py",
         "twisted-12.1.0/twisted/mail/imap4.py",
         "twisted-12.1.0/twisted/mail/relay.py",
         "twisted-12.1.0/twisted/mail/relay.py",
         "twisted-12.1.0/twisted/news/nntp.py",
         "twisted-12.1.0/twisted/news/nntp.py",
         "twisted-12.1.0/twisted/news/nntp.py",
         "twisted-12.1.0/twisted/news/nntp.py",
         "twisted-12.1.0/twisted/persisted/sob.py",
         "twisted-12.1.0/twisted/plugin.py",
         "twisted-12.1.0/twisted/protocols/ftp.py",
         "twisted-12.1.0/twisted/protocols/ftp.py",
         "twisted-12.1.0/twisted/python/test/test_versions.py",
         "twisted-12.1.0/twisted/python/test/test_versions.py",
         "twisted-12.1.0/twisted/python/test/test_versions.py",
         "twisted-12.1.0/twisted/python/test/test_versions.py",
         "twisted-12.1.0/twisted/python/test/test_versions.py",
         "twisted-12.1.0/twisted/python/test/test_versions.py",
         "twisted-12.1.0/twisted/python/test/test_versions.py",
         "twisted-12.1.0/twisted/python/test/test_versions.py",
         "twisted-12.1.0/twisted/python/test/test_versions.py",
         "twisted-12.1.0/twisted/python/test/test_versions.py",
         "twisted-12.1.0/twisted/python/test/test_versions.py",
         "twisted-12.1.0/twisted/python/test/test_versions.py",
         "twisted-12.1.0/twisted/test/test_paths.py",
         "twisted-12.1.0/twisted/test/test_paths.py",
         "twisted-12.1.0/twisted/web/distrib.py",
         "twisted-12.1.0/twisted/words/protocols/oscar.py",
         "twisted-12.1.0/twisted/words/protocols/oscar.py",
         ]


def json_to_dict(JSON_PATH):
    with open(JSON_PATH, 'r') as json_file:
        json_dict = json.load(json_file)
    return json_dict


def main():
    json_dict = json_to_dict(JSON_PATH)
    keys = json_dict.keys()
    checked = []
    for path in PATHS:
        if path not in checked and path not in keys:
            print("Path not in keys: " + path)
            checked.append(path)


if __name__ == "__main__":
    main()
