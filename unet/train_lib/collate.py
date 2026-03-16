from torch.utils.data.dataloader import default_collate


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    first = batch[0]
    if isinstance(first, (tuple, list)) and len(first) == 3:
        images = default_collate([b[0] for b in batch])
        masks = default_collate([b[1] for b in batch])
        metas = [b[2] for b in batch]
        return images, masks, metas
    return default_collate(batch)
