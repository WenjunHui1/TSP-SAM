from dataloaders.video_list import test_dataset, EvalDataset

def test_dataloader(args):   
    test_loader = test_dataset(dataset=args.dataset,
                              testsize=args.testsize)
    print('Test with %d image pairs' % len(test_loader))
    return test_loader 