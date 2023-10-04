from transformers import (Mask2FormerForUniversalSegmentation,
                          SegformerForSemanticSegmentation)

id2label = {
    0: 'Road',
    1: 'Sidewalk',
    2: 'Construction',
    3: 'Fence',
    4: 'Pole',
    5: 'Traffic Light',
    6: 'Traffic Sign',
    7: 'Nature',
    8: 'Sky',
    9: 'Person',
    10: 'Rider',
    11: 'Car',
    12: 'Unknown'
}

label2id = {v: k for k, v in id2label.items()}

SegFormer = SegformerForSemanticSegmentation.from_pretrained(
            'nvidia/segformer-b5-finetuned-cityscapes-1024-1024',
            return_dict=False,
            num_labels=len(id2label),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

Mask2Former = Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-large-cityscapes-semantic",
    id2label=id2label,
    ignore_mismatched_sizes=True
)
