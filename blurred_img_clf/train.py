from src.dataset import ImageFolderDataset
from src.models import BICModel
from src.utils import train, test, plot_images, train_test_split
import yaml
import torch
from torch.nn import CrossEntropyLoss
import os

def main():
    with open("./args.yml", "r") as fp:
        args = yaml.safe_load(fp)

    if not os.path.isdir('./results'):
        os.mkdir('./results')
    if not os.path.isdir('./checkpoints'):
        os.mkdir('./checkpoints')


    device = 'cuda'
    show_sample = True
    cls_labels = os.listdir(args["dataset_dirname"])
    
    dataset = ImageFolderDataset(args["dataset_dirname"], device=device)

    train_dataloader, test_dataloader = train_test_split(dataset, 0.9, args["batch_size"])

    model = BICModel().to(device).requires_grad_(True)
    model = torch.nn.DataParallel(model)

    loss_fn = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=0.1)

    loss_history = {"fit": [], "val": []}

    for epoch in range(args['num_epochs']):
        train_loss = train(epoch, model, loss_fn, train_dataloader, optimizer, device)
        test_loss = test(model, loss_fn, test_dataloader, device)

        loss_history["fit"].append(train_loss)
        loss_history["val"].append(test_loss)

        if show_sample and not epoch%10:
            sample_loader = torch.utils.data.DataLoader(
                dataset, batch_size=args['batch_size'], shuffle=True, num_workers=0
            )

            data_iter = iter(sample_loader)
            x, y = next(data_iter)
            x, y = x.to(device), y.to(device)
            out = model(x)

            x = x.cpu().numpy()[:5].transpose([0, 2, 3, 1])
            y = y.cpu().numpy()[:5]
            out = out.cpu().detach().numpy()[:5]

            plot_images(x, out, y, epoch, cls_labels)

            torch.save(
                {
                    "loss_history": loss_history,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                f"checkpoints/checkpoint_{epoch}.pkl",
            )
        
if __name__ == "__main__":
    main()