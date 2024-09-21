import unittest
import doctest
import os
import glob


class TestDocStrings(unittest.TestCase):
    def test_docstrings(self):  # pragma: no cover
        root_dir_name = os.path.dirname(os.path.dirname(__file__))
        src_dir_name = os.path.join(root_dir_name, "src", "sigmaepsilon", "math")
        python_files = glob.glob(
            os.path.join(src_dir_name, "**", "*.py"), recursive=True
        )
        len_dir_path = len(src_dir_name)
        
        no_failed = 0

        for i in range(len(python_files)):
            file_path = python_files[i][len_dir_path + 1 :]
            doctest_results = doctest.testfile(file_path, package="sigmaepsilon.math")
            no_failed += doctest_results.failed
            
        self.assertEqual(no_failed, 0)


if __name__ == "__main__":
    unittest.main()
