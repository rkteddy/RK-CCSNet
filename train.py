import argparse
import os
import warnings
warnings.filterwarnings("ignore")

from models.rkccsnet import *
from models.csnet import *
from loss import *
import torch.optim as optim
from data_processor import *
from trainer import *


def main():
    global args
    args = parser.parse_args()
    setup_seed(1)

    # Create save directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        torch.backends.cudnn.benchmark = True

    if args.model == 'rkccsnet':
        model = RKCCSNet(sensing_rate=args.sensing_rate)
    elif args.model == 'csnet':
        model = CSNet(sensing_rate=args.sensing_rate)

    model = model.cuda()
    criterion = loss_fn
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60, 90, 120, 150, 180], gamma=0.25, last_epoch=-1)
    train_loader, valid_loader = data_loader(args)

    print('\nModel: %s\n'
          'Sensing Rate: %.2f\n'
          'Epoch: %d\n'
          'Initial LR: %f\n'
          % (args.model, args.sensing_rate, args.epochs, args.lr))

    print('Start training')
    for epoch in range(args.epochs):
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        loss = train(train_loader, model, criterion, optimizer, epoch)
        scheduler.step()
        psnr, ssim = valid(valid_loader, model, criterion)
        print("\nTotal Loss: %f" % loss)
        print("PSNR: %f" % psnr)
        print("SSIM: %f" % ssim)
        torch.save(model.state_dict(), os.path.join(args.save_dir, args.model+'.pth'))
    print('Trained finished.')
    print('Model saved in %s' % (os.path.join(args.save_dir, args.model+'.pth')))


if __name__ == '__main__':
    torch.cuda.set_device(1)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='rkccsnet',
                        choices=['csnet', 'rkccsnet'],
                        help='choose model to train')
    parser.add_argument('--sensing-rate', type=float, default=0.50000,
                        choices=[0.50000, 0.25000, 0.12500, 0.06250, 0.03125],
                        help='set sensing rate')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=4, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--block-size', default=32, type=int,
                        metavar='N', help='block size (default: 32)')
    parser.add_argument('--image-size', default=96, type=int,
                        metavar='N', help='image size used for training (default: 96)')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--save-dir', dest='save_dir',
                        help='The directory used to save the trained models',
                        default='save_temp', type=str)

    main()
