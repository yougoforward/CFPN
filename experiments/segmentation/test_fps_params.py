import time

import torch
import encoding

from encoding.nn import BatchNorm
from .option import Options


if __name__ == "__main__":
    args = Options().parse()
    model = encoding.models.get_segmentation_model(args.model, dataset = args.dataset,
                                       backbone = args.backbone, dilated = args.dilated,
                                       lateral = args.lateral, jpu = args.jpu, aux = args.aux,
                                       se_loss = args.se_loss, norm_layer = BatchNorm)
    print(args.model)
    num_parameters = sum([l.nelement() for l in model.pretrained.parameters()])
    print(num_parameters)
    num_parameters = sum([l.nelement() for l in model.head.parameters()])
    print(num_parameters)

    model.cuda()
    model.eval()
    x = torch.Tensor(1, 3, 520, 520).cuda()

    N = 100
    count=0
    with torch.no_grad():
        for _ in range(N):
            out = model(x)

        result = []
        # st = time.time()
        for _ in range(10):
            st = time.time()
            for _ in range(N):
                # if count == 2*N:
                #     st = time.time()
                # if count == 8*N:
                #     result.append(6*N/(time.time()-st))
                out = model(x)
                # count+=1
            result.append(N/(time.time()-st))
        # result.append(10*N/(time.time()-st))

        import numpy as np
        print(np.mean(result), np.std(result))
