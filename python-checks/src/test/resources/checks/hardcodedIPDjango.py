from django.test import TestCase


class FailedLoginAttemptRepositoryTest(TestCase):
    def test_delete_removes_matching_record(self):
        self.repository.delete("8.8.8.8", "dave")
