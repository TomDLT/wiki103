import torch
from wiki103.softmax import AdaptiveLogSoftmaxWithLoss


def test_softmax():
    # 32 bit precision works
    adaptive16b = AdaptiveLogSoftmaxWithLoss(5, 20, [5]).to(dtype=torch.float32, device='cuda')
    float16b = torch.randn([5, 5], dtype=torch.float32, device='cuda')
    _ = adaptive16b(float16b, torch.tensor([1, 2, 3, 4, 5], device='cuda'))

    # 16 bit precision works
    adaptive16b = AdaptiveLogSoftmaxWithLoss(5, 20, [5]).to(dtype=torch.float16, device='cuda')
    float16b = torch.randn([5, 5], dtype=torch.float16, device='cuda')
    _ = adaptive16b(float16b, torch.tensor([1, 2, 3, 4, 5], device='cuda'))

    # 16 bit with amp works
    with torch.cuda.amp.autocast():
        adaptive16b = AdaptiveLogSoftmaxWithLoss(5, 20, [5]).to(dtype=torch.float16, device='cuda')
        float16b = torch.randn([5, 5], dtype=torch.float16, device='cuda')
        _ = adaptive16b(float16b, torch.tensor([1, 2, 3, 4, 5], device='cuda'))
