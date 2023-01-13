import pytest
import numpy as np
from nanodet.data.dataset import CocoDataset, build_dataset
import torch
from nanodet.data.collate import naive_collate
import pdb

def test_cocodataset():
    cfg = dict(
        name="CocoMultiTaskDataset",
        img_path="./tests/data",
        ann_path="./tests/data/dummy_coco.json",
        seg_path="./tests/data",
        input_size=[320, 320],  # [w,h]
        keep_ratio=True,
        use_instance_mask=False,
        pipeline=dict(normalize=[[103.53, 116.28, 123.675], [57.375, 57.12, 58.395]]),
    )
    dataset = build_dataset(cfg, "train")
    assert isinstance(dataset, CocoDataset)

    for i, data in enumerate(dataset):
        assert data["img"].shape == (3, 320, 285)
        print(np.unique(data["gt_seg"]))
 

    with pytest.raises(AssertionError):
        build_dataset(cfg, "2333")

def test_multitask_aug():
    cfg = dict(
        name="CocoMultiTaskDataset",
        img_path="./tests/data",
        ann_path="./tests/data/dummy_coco.json",
        seg_path="./tests/data",
        input_size=[320, 320],  # [w,h]
        keep_ratio=False,
        use_instance_mask=False,
        pipeline=dict(
            normalize=[[103.53, 116.28, 123.675], [57.375, 57.12, 58.395]],
            scale=[0.6, 1.4],
            stretch= [[0.8, 1.2], [0.8, 1.2]],
            rotation=0,
            shear=0,
            translate=0.2,
            flip=0.5,
            brightness=0.2,
            contrast=[0.6,1.4],
            saturation=[0.5,1.2],
            perspective=0.0),
        
    )
    dataset = build_dataset(cfg, "train")
    for i, data in enumerate(dataset):
        assert data["img"].shape[1]==data["gt_seg"].shape[0]
        assert data["img"].shape[2]==data["gt_seg"].shape[1]
    dataset.get_train_data()

def test_train():
    cfg = dict(
        name="CocoMultiTaskDataset",
        img_path="/datasets/coco/images/train2017",
        ann_path="/datasets/coco/annotations/instances_train2017.json",
        seg_path="/datasets/coco_stuff_164k_nanodetscnn/annotations/train2017",
        input_size=[320, 320],  # [w,h]
        keep_ratio=False,
        use_instance_mask=False,
        pipeline=dict(
            normalize=[[103.53, 116.28, 123.675], [57.375, 57.12, 58.395]],
            # scale=[0.6, 1.4],
            # stretch= [[0.8, 1.2], [0.8, 1.2]],
            # rotation=0,
            # shear=0,
            # translate=0.2,
            # flip=0.5,
            # brightness=0.2,
            # contrast=[0.6,1.4],
            # saturation=[0.5,1.2],
            # perspective=0.0
            ),
        
    )
    dataset = build_dataset(cfg, "train")
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=32,
        pin_memory=True,
        collate_fn=naive_collate,
        drop_last=True,
    )
    # print(dir(train_dataloader))
    for i,batch in enumerate(train_dataloader):
        if i>=1:
            break
        else:
            for seg_map in batch['gt_seg']:
                print(np.unique(seg_map))
    #pdb.set_trace()
    return train_dataloader
    

def test_multi_scale():
    cfg = dict(
        name="CocoDataset",
        img_path="./tests/data",
        ann_path="./tests/data/dummy_coco.json",
        input_size=[320, 320],  # [w,h]
        multi_scale=[1.5, 1.5],
        keep_ratio=True,
        use_instance_mask=True,
        pipeline=dict(normalize=[[103.53, 116.28, 123.675], [57.375, 57.12, 58.395]]),
    )
    dataset = build_dataset(cfg, "train")

    for i, data in enumerate(dataset):
        assert data["img"].shape == (3, 480, 427)
        for mask in data["gt_masks"]:
            assert mask.shape == (480, 427)
