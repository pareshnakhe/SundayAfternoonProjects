import unittest
import math
from model_revaluation import extract_main_ingredient


class TestStringMethods(unittest.TestCase):

    def test_invalid_character(self):
        result = extract_main_ingredient('ass$df')
        self.assertTrue(math.isnan(result))

        result = extract_main_ingredient('asdf, dfd - dfdf')
        self.assertEqual(result, 'asdf')

    def test_empty_string(self):
        result = extract_main_ingredient('')
        self.assertTrue(math.isnan(result))

        result = extract_main_ingredient(float('nan'))
        self.assertTrue(math.isnan(result))

    def test_diff_ingredient_formats(self):
        result = extract_main_ingredient('asddf, asasa - (56) - dsf')
        self.assertEqual(result, 'asddf')

        result = extract_main_ingredient('as, ddf #3 asasa')
        self.assertTrue(math.isnan(result))


if __name__ == '__main__':
    unittest.main()
