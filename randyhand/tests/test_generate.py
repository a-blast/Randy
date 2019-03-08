from unittest import TestCase

import randyhand

class TestGenerate(TestCase):
    """Test image generation"""
    def test_calculate_line_parameters(self):
        for _ in range(100000):
            char_height, space_between_lines = randyhand.calculate_line_parameters(320)
            self.assertLess(0, char_height)
            self.assertLess(0, space_between_lines)
            self.assertLess(space_between_lines, char_height)

