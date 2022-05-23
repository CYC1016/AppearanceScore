import unittest

from data import MyDataset


class MyTestCase(unittest.TestCase):
    def testTrainData(self):
        self.assertEqual(MyDataset("train").__len__(), 450)
    def testTestData(self):
        self.assertEqual(MyDataset("test").__len__(), 50)

if __name__ == '__main__':
    unittest.main()
