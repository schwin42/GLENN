'''
Created on Sep 24, 2017

@author: schwin42
'''
import unittest
import numpy as np
from Model import ExperienceBuffer

class TestExperienceBuffer(unittest.TestCase):
    def test_sample(self):
        buffer = ExperienceBuffer()
        experience = []
        for i in range(10):
            observation = [[i, i, i, i], i, i, [i, i, i, i], i]
            experience.append(observation)
        
        buffer.add(experience)
        exhaustive_sample = buffer.sample(10)
        
        for i in range(10):
            match_found = False
            for j in range(len(exhaustive_sample)):
                if experience[i][0] == exhaustive_sample[j][0]:
                    #print("exhaustive sample: " + str(exhaustive_sample))
                    #print("j = " + str(j))
                    #print("item b: " + str(exhaustive_sample[j]))
                    exhaustive_sample = np.delete(exhaustive_sample, j, 0)
                    #print("item a: " + str(exhaustive_sample[j]))
                    match_found = True
                    break
                else:
                    continue
                
            #Inner loop ended with no matching experience found, so fail
            self.assertTrue(match_found, "Experience match not found in sample")
            if not match_found: 
                return None
        
        self.assertEqual(len(exhaustive_sample), 0, "Sample was not successfully emptied with matches from experience")    
    

if __name__ == '__main__':
    unittest.main()