import datetime
import sys
from construct import len_

import matplotlib.pyplot as plt
import torch
import wandb
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as transforms
from torchvision.models import alexnet
import torch.nn.functional as F
from datasets.dataset import RopeActionDataset, RopeDataset
from nets.inverse_dynamics import InverseDynamics
from nets.infogan import Q, T, D, G, Normal, Uniform
from torch.optim import Adam
from utils import weights_init
from torch import nn
import os


def init_net(net: torch.nn.Module, device, weights_init=weights_init):
    net = net.to(device).cuda()
    net.apply(weights_init)
    for param in net.parameters():
        param.requires_grad = True
    return net


class CausalInfoganTrainer:
    def __init__(self, **kwargs):
        self.params = kwargs
        self.batch_size = kwargs["batch_size"]
        self.epochs = kwargs["epochs"]
        self.paths = kwargs["paths"]
        self.out_dir = kwargs["out_dir"]

        self.s_dim = kwargs["s_dim"]
        self.z_dim = kwargs["z_dim"]
        self.ch_dim = kwargs["channel_dim"]

        self.info_w = kwargs["max_info_w"]
        self.trans_w = kwargs["trans_reg_w"]

        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.prior = Uniform(torch.Size([self.batch_size, self.s_dim])).to(self.device)
        self.noise = Normal(torch.Size([self.batch_size, self.z_dim])).to(self.device)

        self.G = init_net(
            G(s_dim=self.s_dim, z_dim=2, channel_dim=self.ch_dim), self.device
        )
        self.D = init_net(D(self.ch_dim), self.device)
        self.Q = init_net(Q(self.s_dim, self.ch_dim), self.device)
        self.T = init_net(T(s_dim=self.s_dim), self.device)

        self.lr = kwargs["lr"]
        self.betas = kwargs["betas"]
        self.optimD = Adam(self.D.parameters(), self.lr, betas=self.betas)
        self.optimG = Adam(
            [
                {"params": self.G.parameters()},
                {"params": self.Q.parameters()},
                {"params": self.T.parameters()},
            ],
            self.lr,
            betas=self.betas,
        )

        self.criterion = nn.BCEWithLogitsLoss()  # for generator and discriminator

        self.REAL = 1.0
        self.FAKE = 0.0

        trans = transforms.Compose(
            [
                transforms.CenterCrop((320, 320)),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                lambda x: x.mean(dim=0)[None, :, :],  # Grayscale
                lambda x: (x - x.min()) / x.max(),  # 0-1
            ]
        )

        datasets = ConcatDataset(
            [RopeDataset(path, transforms=trans) for path in self.paths]
        )
        print(len(datasets))
        self.train_loader = DataLoader(
            datasets, self.batch_size, num_workers=3, shuffle=True
        )

    def train(self):
        wandb.init(project="rope_manipulation", config=self.params)
        label = torch.ones(torch.Size([self.batch_size])).cuda()
        for epoch in range(self.epochs):
            self.G.train()
            self.D.train()
            self.Q.train()
            self.T.train()
            for batch_idx, (
                obs,
                next_obs,
            ) in enumerate(self.train_loader):

                o_real, o_real_next = obs.to(self.device), next_obs.to(self.device)

                # Disc - real
                self.optimD.zero_grad()
                real_probs = self.D(o_real, o_real_next)
                label.fill_(self.REAL)
                real_D_loss = self.criterion(real_probs.view(-1), label.view(-1))
                real_D_loss.backward()

                # Disc - fake
                s = self.prior().to(self.device)
                s_next = self.T(s).to(self.device)
                z = self.noise().to(self.device)

                o_fake, o_fake_next = self.G(s, s_next, z)
                fake_G_loss = self.D(o_fake.detach(), o_fake_next.detach()).view(-1)
                label.fill_(self.FAKE)
                fake_D_loss = self.criterion(fake_G_loss, label)
                fake_D_loss.backward()
                D_loss = fake_D_loss + real_D_loss
                self.optimD.step()

                # Gen
                fake_G_loss = self.D(o_fake, o_fake_next).view(-1)
                label.fill_(self.REAL)
                G_loss = self.criterion(fake_G_loss, label)

                # Maximal mutual inf
                crossen_loss = -self.Q.log_post_prob(o_fake, s).mean(0)
                crossen_next_loss = -self.Q.log_post_prob(o_fake_next, s_next).mean(0)
                ent_next_loss = -self.T.log_post_prob(s, s_next).mean(0)
                ent_loss = -self.prior.log_prob(s).mean(0)
                Q_loss = crossen_loss - ent_loss + crossen_next_loss - ent_next_loss

                T_loss = (self.T.ret_var(s) ** 2).sum(1).mean(0)

                self.optimG.zero_grad()
                (G_loss + self.info_w * Q_loss + self.trans_w * T_loss).backward()
                self.optimG.step()

                if batch_idx % 100 == 0:
                    wandb.log(
                        {
                            "D_loss": D_loss.item(),
                            "G_loss": G_loss.item(),
                            "Q_loss": Q_loss.item(),
                            "T_loss": T_loss.item(),
                            "crossen_loss": crossen_loss.item(),
                            "crossen_next_loss": crossen_loss.item(),
                            "ent_loss": ent_loss.item(),
                            "ent_next_loss": ent_next_loss.item(),
                            "real_D_loss": real_D_loss.mean(),
                            "fake_D_loss": fake_D_loss.mean(),
                            "fake_G_loss": fake_G_loss.mean(),
                        }
                    )
                    print(
                        "\n#######################"
                        "\nEpoch/Iter:%d/%d; "
                        "\nDloss: %.3f; "
                        "\nGloss: %.3f; "
                        "\nQloss: %.3f; "
                        "\nT_loss: %.3f; "
                        "\nEnt: %.3f, %.3f; "
                        "\nCross Ent: %.3f, %.3f; "
                        "\nD(x): %.3f; "
                        "\nD(G(z)): b %.3f, a %.3f;"
                        % (
                            epoch,
                            batch_idx,
                            D_loss.item(),
                            G_loss.item(),
                            Q_loss.item(),
                            T_loss.item(),
                            ent_loss.item(),
                            ent_next_loss.item(),
                            crossen_loss.item(),
                            crossen_next_loss.item(),
                            real_D_loss.data.mean(),
                            fake_D_loss.data.mean(),
                            fake_G_loss.data.mean(),
                        )
                    )

            if epoch % 5 == 0:
                folder = datetime.datetime.today().strftime("%Y-%m-%d-%H:%M:%S")
                if not os.path.exists("%s/%s" % (self.out_dir, folder)):
                    os.makedirs("%s/%s" % (self.out_dir, folder))
                for i in [self.G, self.D, self.Q, self.T]:
                    torch.save(
                        i.state_dict(),
                        os.path.join(
                            self.out_dir,
                            "%s" % folder,
                            "%s_%d"
                            % (
                                i.__class__.__name__,
                                epoch,
                            ),
                        ),
                    )

    def display_sample(self):
        o, o_next = next(iter(self.train_loader))
        o, o_next = o.permute(0, 2, 3, 1), o_next.permute(0, 2, 3, 1)
        print(o[0].numpy().squeeze(axis=2).shape)
        print(o_next[0].numpy().squeeze(axis=2).shape)
        plt.imsave("test_batch_1.png", o[0].numpy().squeeze(axis=2))
        plt.imsave("test_batch_2.png", o_next[0].numpy().squeeze(axis=2))


class InverseDynamicsTrainer:
    def __init__(self, **kwargs):
        self.params = kwargs
        self.batch_size = kwargs["batch_size"]
        self.lr = kwargs["lr"]
        self.betas = kwargs["betas"]
        self.epochs = kwargs["epochs"]
        self.train_paths = kwargs["train_paths"]
        self.test_paths = kwargs["test_paths"]
        self.out_dir = kwargs["out_dir"]

        self.p_dim = kwargs["rope_site_dim"]
        self.th_dim = kwargs["angle_dim"]
        self.len_dim = kwargs["length_dim"]

        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

        self.model = InverseDynamics(self.p_dim, self.th_dim, self.len_dim)

        self.model = self.model.to(self.device).cuda()

        self.model.p_fc.apply(weights_init)
        self.model.th_fc.apply(weights_init)
        self.model.len_fc.apply(weights_init)
        for param in self.model.len_fc.parameters():
            param.requires_grad = True
        trans = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                lambda x: x.mean(dim=0)[None, :, :].repeat(3, 1, 1),  # Grayscale
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                lambda x: (x - x.min()) / x.max(),  # 0-1
            ]
        )

        train_dataset = ConcatDataset(
            [
                RopeActionDataset(path, transforms=trans, **kwargs)
                for path in self.train_paths
            ]
        )
        test_dataset = ConcatDataset(
            [
                RopeActionDataset(path, transforms=trans, **kwargs)
                for path in self.test_paths
            ]
        )
        print(len(train_dataset))
        print(len(test_dataset))
        self.train_loader = DataLoader(
            train_dataset, self.batch_size, num_workers=3, shuffle=True
        )
        self.test_loader = DataLoader(
            test_dataset, self.batch_size, num_workers=3, shuffle=True
        )
        self.criterion = lambda x, label: -torch.sum(
            torch.log_softmax(x, dim=-1) * label
        )

        self.optimizer = Adam(self.model.parameters(), self.lr, betas=self.betas)
        wandb.init(group="Inverse Dynamics")

    def train_test(self):
        for epoch in range(self.epochs):
            print(f"Epoch: {epoch}")
            self.train()
            self.test()
            if epoch % 5 == 0:
                folder = datetime.datetime.today().strftime("%Y-%m-%d-%H:%M:%S")
                if not os.path.exists("%s/%s" % (self.out_dir, folder)):
                    os.makedirs("%s/%s" % (self.out_dir, folder))
                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        self.out_dir,
                        "%s" % folder,
                        "%s_%d"
                        % (
                            self.model.__class__.__name__,
                            epoch,
                        ),
                    ),
                )

    def train(self):
        self.model.train()
        for batch_idx, (o, o_next, (p_label, th_label, len_label)) in enumerate(
            self.train_loader
        ):
            self.model.zero_grad()
            o, o_next = o.to(self.device), o_next.to(self.device)
            p_label, th_label, len_label = (
                p_label.to(self.device),
                th_label.to(self.device),
                len_label.to(self.device),
            )
            p_pred, th_pred, len_pred = self.model(o, o_next)

            p_loss = self.criterion(p_pred.float(), p_label)
            th_loss = self.criterion(th_pred.float(), th_label)
            len_loss = self.criterion(len_pred.float(), len_label)
            loss = p_loss + th_loss + len_loss
            loss.requires_grad = True
            loss.backward()
            self.optimizer.step()
            if batch_idx % 20 == 0:
                wandb.log(
                    {
                        "batch_idx": batch_idx,
                        "p_loss": p_loss.item(),
                        "th_loss": th_loss.item(),
                        "len_loss": len_loss.item(),
                        "loss": loss.item(),
                    }
                )
                print(
                    "\n#######################"
                    "\nbatch_idx: %.3f; "
                    "\np_loss: %.3f; "
                    "\nth_loss: %.3f; "
                    "\nlen_loss: %.3f; "
                    "\nloss: %.3f; "
                    % (
                        batch_idx,
                        p_loss.item(),
                        th_loss.item(),
                        len_loss.item(),
                        loss.item(),
                    )
                )

    def test(self):
        self.model.eval()
        for batch_idx, (o, o_next, (p_label, th_label, len_label)) in enumerate(
            self.train_loader
        ):
            o, o_next = o.to(self.device), o_next.to(self.device)
            p_label, th_label, len_label = (
                p_label.to(self.device),
                th_label.to(self.device),
                len_label.to(self.device),
            )
            p_pred, th_pred, len_pred = self.model(o, o_next)
            p_loss = self.criterion(p_pred.float(), p_label)
            th_loss = self.criterion(th_pred.float(), th_label)
            len_loss = self.criterion(len_pred.float(), len_label)
            loss = p_loss + th_loss + len_loss

            if batch_idx % 20 == 0:
                wandb.log(
                    {
                        "batch_idx": batch_idx,
                        "p_loss": p_loss.item(),
                        "th_loss": th_loss.item(),
                        "len_loss": len_loss.item(),
                        "loss": loss.item(),
                    }
                )
                print(
                    "\n#######################"
                    "\nbatch_idx: %.3f; "
                    "\np_loss: %.3f; "
                    "\nth_loss: %.3f; "
                    "\nlen_loss: %.3f; "
                    "\nloss: %.3f; "
                    % (
                        batch_idx,
                        p_loss.item(),
                        th_loss.item(),
                        len_loss.item(),
                        loss.item(),
                    )
                )

    def display_sample(self):
        o, o_next, (rope_site, angle, len) = next(iter(self.train_loader))
        o, o_next = o.permute(0, 2, 3, 1), o_next.permute(0, 2, 3, 1)
        plt.imsave("test_batch_1.png", o[0].numpy())
        plt.imsave("test_batch_2.png", o_next[0].numpy())
