import unittest
from E46.e46b_b4_generalization import make_spec, CATEGORIES

class E46B4Tests(unittest.TestCase):
    def test_entity_disjoint_split_and_category_balance(self):
        spec=make_spec(123)
        self.assertEqual(len(spec["facts"]),256)
        train=[f for f in spec["facts"] if f["split"]=="train_entity"]
        test=[f for f in spec["facts"] if f["split"]=="unseen_entity"]
        self.assertEqual(len(train),192)
        self.assertEqual(len(test),64)
        self.assertFalse({f["entity"] for f in train}&{f["entity"] for f in test})
        for category, *_ in CATEGORIES:
            self.assertEqual(sum(f["category"]==category and f["split"]=="train_entity" for f in spec["facts"]),24)
            self.assertEqual(sum(f["category"]==category and f["split"]=="unseen_entity" for f in spec["facts"]),8)
    def test_queries_are_split_correctly(self):
        spec=make_spec(456)
        for f in spec["facts"]:
            if f["split"]=="train_entity":
                self.assertEqual(len(f["train_queries"]),2)
                self.assertEqual(len(f["validation_queries"]),1)
                self.assertEqual(len(f["test_queries"]),0)
            else:
                self.assertEqual(len(f["train_queries"]),0)
                self.assertEqual(len(f["validation_queries"]),0)
                self.assertEqual(len(f["test_queries"]),2)
    def test_deterministic_generation(self):
        self.assertEqual(make_spec(7),make_spec(7))

if __name__=="__main__": unittest.main()
