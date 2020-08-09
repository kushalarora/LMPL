import pathlib
from allennlp.common.testing import AllenNlpTestCase, ModelTestCase as AllenNlpModelTestCase
from allennlp.common.util import import_module_and_submodules
ROOT = (pathlib.Path(__file__).parent / "..").resolve()  # pylint: disable=no-member

class LMPLTestCase(AllenNlpTestCase):
    PROJECT_ROOT = ROOT
    MODULE_ROOT = PROJECT_ROOT / "lmpl"
    TOOLS_ROOT = None  # just removing the reference from super class
    TESTS_ROOT = PROJECT_ROOT / "tests"
    FIXTURES_ROOT = PROJECT_ROOT / "test_fixtures"

class ModelTestCase(AllenNlpModelTestCase):
    PROJECT_ROOT = ROOT
    MODULE_ROOT = PROJECT_ROOT / "lmpl"
    TOOLS_ROOT = None  # just removing the reference from super class
    TESTS_ROOT = PROJECT_ROOT / "tests"
    FIXTURES_ROOT = PROJECT_ROOT / "test_fixtures"

    def set_up_model(self, param_file, dataset_file):
      import_module_and_submodules("lmpl")
      super().set_up_model(param_file, dataset_file)