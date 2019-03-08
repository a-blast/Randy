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

    def test_get_user_word(self):
        test_text = ["Hey", "hi", "whatsup", "hello"]
        user_next_word = randyhand.get_next_word_function(test_text)
        test_text = test_text[:]
        test_text.reverse()
        for word in test_text:
            self.assertEqual(word, user_next_word())

    def text_get_random_word(self):
        random_next_word = randyhand.get_next_word_function(None)

        for _ in range(1000):
            word = random_next_word()
            self.assertIsInstance(word, str)
            self.assertGreater(len(word),0)
            for letter in word:
                self.assertNotEqual(" ", letter)

