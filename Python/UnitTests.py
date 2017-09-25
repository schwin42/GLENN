'''
Created on Sep 24, 2017

@author: schwin42
'''
import unittest

class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

class TestExperienceBuffer(unittest.TestCase):
	def test_sample(self):
		buffer = ExperienceBuffer()
		experience = []
		for i in range(10):
			experience.append(observation = [[i, i, i, i], i, i, [i, i, i, i], i])
		
		buffer.add(experience)
		exhaustive_sample = buffer.sample(10)
		
		for i in range(10):
			match_found = False
			for j in range(exhaustive_sample.length()):
				if buffer[i][0] == exhaustive_sample[j][0]:
					exhaustive_sample.remove(j)
					match_found = True
					break
				else:
					continue
				
			#Inner loop ended with no matching experience found, so fail
			self.assertTrue(match_found, "Experience match not found in sample")
		
		self.assertEqual(exhaustive_sample.length, 0, "Buffer contains more items than source experience")	
	

if __name__ == '__main__':
    unittest.main()