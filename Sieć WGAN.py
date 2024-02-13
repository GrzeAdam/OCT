import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from WGAN import Discriminator, Generator, initialize_weights
from utils import gradient_penalty
from torchvision.utils import save_image

device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 256
CHANNELS_IMG = 1
Z_DIM = 128
NUM_EPOCHS = 100
FEATURES_CRITIC = 64
FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

gen_img_dir = "Gen_WGAN"
os.makedirs(gen_img_dir, exist_ok=True)


transforms = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]
)


dataset = datasets.ImageFolder(root="oct_crop", transform=transforms)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
critic = Discriminator(CHANNELS_IMG, FEATURES_CRITIC).to(device)
initialize_weights(gen)
initialize_weights(critic)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0,0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0,0.9))

fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
writer_real = SummaryWriter(f"logsWGAN/real")
writer_fake = SummaryWriter(f"logsWGAN/fake")
step = 0

gen.train()
critic.train()

checkpoint_dir = "WGAN_checkpoint"
critic_checkpoint = "critic.pth.tar"
generator_checkpoint = "generator.pth.tar"

def save_checkpoint(model, optimizer, epoch, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }
    torch.save(checkpoint, os.path.join(checkpoint_dir, filename))

def load_checkpoint(model, optimizer, filename, checkpoint_dir):
    filepath = os.path.join(checkpoint_dir, filename)
    if not os.path.isfile(filepath):
        print(f"No checkpoint found at '{filepath}'")
        return None

    try:
        checkpoint = torch.load(filepath, map_location='cpu')  # or 'cuda' if you're using a GPU
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"Checkpoint loaded from '{filename}' (epoch {checkpoint['epoch']})")
        return checkpoint["epoch"]
    except RuntimeError as e:
        print(f"Error loading checkpoint '{filename}': {e}")
        return None


print("Before loading generator checkpoint")
print("Before loading generator checkpoint")
if os.path.exists(os.path.join(checkpoint_dir, generator_checkpoint)):
    start_epoch = load_checkpoint(gen, opt_gen, generator_checkpoint, checkpoint_dir)
    if start_epoch is not None:
        start_epoch += 1
    print(f"Generator checkpoint loaded, starting at epoch {start_epoch}")
else:
    start_epoch = 0
    print("No generator checkpoint found, starting from beginning")

print("Before loading critic checkpoint")
if os.path.exists(os.path.join(checkpoint_dir, critic_checkpoint)):
    last_epoch = load_checkpoint(critic, opt_critic, critic_checkpoint, checkpoint_dir)
    print("Critic checkpoint loaded")
else:
    print("No critic checkpoint found")


for epoch in range(NUM_EPOCHS):
    for batch_idx, (data, _) in enumerate(tqdm(loader)):
        data = data.to(device)
        cur_batch_size = data.shape[0]

        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn(cur_batch_size, Z_DIM, 1, 1).to(device)
            fake = gen(noise)
            critic_real = critic(data).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            gp = gradient_penalty(critic, data, fake, device=device)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP*gp
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()

        gen_fake = critic(fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx % 10 == 0 and batch_idx > 0:
            gen.eval()
            critic.eval()
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \
                  Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}"
            )

            with torch.no_grad():
                fake = gen(noise)
                img_grid_real = torchvision.utils.make_grid(
                    data[:16], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:16], normalize=True
                )

                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)

            save_checkpoint(gen, opt_gen, epoch, filename=f"generator.pth.tar")
            save_checkpoint(critic, opt_critic, epoch, filename=f"critic.pth.tar")
            step += 1
            gen.train()
            critic.train()
    if epoch % 2 == 0:
        with torch.no_grad():
            fixed_noise = torch.randn(32, Z_DIM, 1, 1).to(device)
            fake = gen(fixed_noise)
            img_grid = torchvision.utils.make_grid(fake, normalize=True)
            save_image(img_grid, os.path.join(gen_img_dir, f"epoch_{epoch+1}.jpg"))
            
