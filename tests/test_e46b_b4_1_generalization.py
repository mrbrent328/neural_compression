import unittest
import torch
from E46.e46b_b4_1_generalization import make_spec, CATEGORIES, score_queries
from E46.e46b_memory_adapter import NeuralMemoryAdapter

class E46B41Tests(unittest.TestCase):
    def test_randomized_stratified_split(self):
        spec=make_spec(123); facts=spec["facts"]
        train=[f for f in facts if f["split"]=="train_entity"]
        unseen=[f for f in facts if f["split"]=="unseen_entity"]
        self.assertEqual((len(facts),len(train),len(unseen)),(256,192,64))
        for category,*_ in CATEGORIES:
            self.assertEqual(sum(f["category"]==category and f["split"]=="train_entity" for f in facts),24)
            self.assertEqual(sum(f["category"]==category and f["split"]=="unseen_entity" for f in facts),8)
        nums=[int(f["entity"].split("-")[1]) for f in unseen]
        self.assertTrue(any(n<=24 for n in nums)); self.assertTrue(any(n>24 for n in nums))
    def test_four_factorized_query_conditions(self):
        for f in make_spec(456)["facts"]:
            if f["split"]=="train_entity":
                sizes=(len(f["train_queries"]),len(f["validation_queries"]),len(f["seen_novel_queries"]),len(f["unseen_familiar_queries"]),len(f["unseen_novel_queries"]))
                self.assertEqual(sizes,(2,1,2,0,0))
            else:
                sizes=(len(f["train_queries"]),len(f["validation_queries"]),len(f["seen_novel_queries"]),len(f["unseen_familiar_queries"]),len(f["unseen_novel_queries"]))
                self.assertEqual(sizes,(0,0,0,2,2))
    def test_seeded_split_is_deterministic_but_varies(self):
        self.assertEqual(make_spec(7),make_spec(7))
        a={f["entity"]:f["split"] for f in make_spec(7)["facts"]}
        b={f["entity"]:f["split"] for f in make_spec(8)["facts"]}
        self.assertNotEqual(a,b)
    def test_margin_uses_best_incorrect_candidate(self):
        facts=[{"capture":torch.tensor([1.,0.])},{"capture":torch.tensor([0.,1.])},{"capture":torch.tensor([-1.,0.])}]
        adapter=NeuralMemoryAdapter(2,2,2)
        with torch.no_grad():
            for layer in (adapter.address_writer[0],adapter.query_encoder[0]):
                layer.weight.copy_(torch.eye(2)); layer.bias.zero_()
            for norm in (adapter.address_writer[1],adapter.query_encoder[1]):
                norm.weight.fill_(1.); norm.bias.zero_()
        result=score_queries(adapter,facts,[{"vector":torch.tensor([1.,0.]),"target":0,"category":"x"}])
        self.assertAlmostEqual(result["mean_margin"],2.0,places=5)
if __name__=="__main__": unittest.main()
