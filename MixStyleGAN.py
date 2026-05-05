"""
MixStyleGAN - Image Style Transfer using CycleGAN
Transforms images to Van Gogh or Picasso styles
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os
from PIL import Image
import numpy as np
from pathlib import Path


# ==================== DATASET ====================

class StyleDataset(Dataset):
    """Dataset for loading content and style images"""

    def __init__(self, content_dir, style_dir, transform=None):
        self.content_paths = list(Path(content_dir).glob("*.jpg")) + \
                            list(Path(content_dir).glob("*.png"))
        self.style_paths = list(Path(style_dir).glob("*.jpg")) + \
                          list(Path(style_dir).glob("*.png"))
        self.transform = transform

    def __len__(self):
        return max(len(self.content_paths), len(self.style_paths))

    def __getitem__(self, idx):
        content_img = Image.open(self.content_paths[idx % len(self.content_paths)]).convert('RGB')
        style_img = Image.open(self.style_paths[idx % len(self.style_paths)]).convert('RGB')

        if self.transform:
            content_img = self.transform(content_img)
            style_img = self.transform(style_img)

        return content_img, style_img


# ==================== GENERATOR ====================

class ResidualBlock(nn.Module):
    """Residual block for generator"""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """Generator network for style transfer"""

    def __init__(self, input_channels=3, features=64, num_residuals=9):
        super().__init__()

        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, features, 7, padding=3),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
        )

        # Downsampling
        self.downsample = nn.Sequential(
            nn.Conv2d(features, features * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(features * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features * 2, features * 4, 3, stride=2, padding=1),
            nn.InstanceNorm2d(features * 4),
            nn.ReLU(inplace=True),
        )

        # Residual blocks
        self.residuals = nn.Sequential(
            *[ResidualBlock(features * 4) for _ in range(num_residuals)]
        )

        # Upsampling
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(features * 4, features * 2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(features * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(features * 2, features, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
        )

        # Output convolution
        self.output = nn.Conv2d(features, input_channels, 7, padding=3)

    def forward(self, x):
        x = self.initial(x)
        x = self.downsample(x)
        x = self.residuals(x)
        x = self.upsample(x)
        return torch.tanh(self.output(x))


# ==================== DISCRIMINATOR ====================

class Discriminator(nn.Module):
    """PatchGAN discriminator"""

    def __init__(self, input_channels=3, features=[64, 128, 256, 512]):
        super().__init__()

        layers = [
            nn.Conv2d(input_channels, features[0], 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        for feature in features[:-1]:
            layers.extend([
                nn.Conv2d(feature, feature * 2, 4, stride=2, padding=1),
                nn.InstanceNorm2d(feature * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ])

        layers.append(nn.Conv2d(features[-1], 1, 4, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ==================== LOSSES ====================

class CycleLoss(nn.Module):
    """Cycle consistency loss"""

    def __init__(self, lambda_cycle=10.0):
        super().__init__()
        self.lambda_cycle = lambda_cycle
        self.mse = nn.MSELoss()

    def forward(self, original, reconstructed):
        return self.mse(original, reconstructed) * self.lambda_cycle


class IdentityLoss(nn.Module):
    """Identity loss to preserve color composition"""

    def __init__(self, lambda_identity=5.0):
        super().__init__()
        self.lambda_identity = lambda_identity
        self.mse = nn.MSELoss()

    def forward(self, input_img, output_img):
        return self.mse(input_img, output_img) * self.lambda_identity


class AdversarialLoss(nn.Module):
    """Adversarial loss for GAN training"""

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, prediction, target):
        return self.mse(prediction, target)


# ==================== TRAINER ====================

class MixStyleGANTrainer:
    """Trainer for MixStyleGAN - Style Transfer"""

    def __init__(
        self,
        content_dir="content_images",
        style_dir="style_images",
        save_dir="generated_images",
        checkpoint_dir="checkpoints",
        image_size=256,
        batch_size=1,
        learning_rate=0.0002,
        num_epochs=100,
        lambda_cycle=10.0,
        lambda_identity=5.0,
        device=None
    ):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Create directories
        for dir_path in [content_dir, style_dir, save_dir, checkpoint_dir]:
            Path(dir_path).mkdir(exist_ok=True)

        self.save_dir = save_dir
        self.checkpoint_dir = checkpoint_dir

        # Data transformation
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        # Initialize models
        self.G_content_to_style = Generator().to(self.device)
        self.G_style_to_content = Generator().to(self.device)
        self.D_content = Discriminator().to(self.device)
        self.D_style = Discriminator().to(self.device)

        # Initialize losses
        self.cycle_loss = CycleLoss(lambda_cycle).to(self.device)
        self.identity_loss = IdentityLoss(lambda_identity).to(self.device)
        self.adversarial_loss = AdversarialLoss().to(self.device)

        # Initialize optimizers
        self.opt_G = optim.Adam(
            list(self.G_content_to_style.parameters()) +
            list(self.G_style_to_content.parameters()),
            lr=learning_rate, betas=(0.5, 0.999)
        )
        self.opt_D = optim.Adam(
            list(self.D_content.parameters()) +
            list(self.D_style.parameters()),
            lr=learning_rate, betas=(0.5, 0.999)
        )

        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def load_data(self, content_dir, style_dir):
        """Load training data"""
        dataset = StyleDataset(content_dir, style_dir, self.transform)
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
        print(f"Loaded {len(dataset)} training samples")

    def train_step(self, content_imgs, style_imgs):
        """Single training step"""
        content_imgs = content_imgs.to(self.device)
        style_imgs = style_imgs.to(self.device)

        # ==================== TRAIN DISCRIMINATORS ====================
        self.opt_D.zero_grad()

        # Generate fake images
        fake_style = self.G_content_to_style(content_imgs)
        fake_content = self.G_style_to_content(style_imgs)

        # Discriminator predictions
        pred_real_content = self.D_content(content_imgs)
        pred_fake_content = self.D_content(fake_content.detach())
        pred_real_style = self.D_style(style_imgs)
        pred_fake_style = self.D_style(fake_style.detach())

        # Calculate discriminator losses
        real_target = torch.ones_like(pred_real_content)
        fake_target = torch.zeros_like(pred_fake_content)

        d_loss_content = self.adversarial_loss(pred_real_content, real_target) + \
                        self.adversarial_loss(pred_fake_content, fake_target)
        d_loss_style = self.adversarial_loss(pred_real_style, real_target) + \
                      self.adversarial_loss(pred_fake_style, fake_target)

        d_loss = (d_loss_content + d_loss_style) / 2
        d_loss.backward()
        self.opt_D.step()

        # ==================== TRAIN GENERATORS ====================
        self.opt_G.zero_grad()

        # Forward pass
        fake_style = self.G_content_to_style(content_imgs)
        fake_content = self.G_style_to_content(style_imgs)

        # Cycle consistency
        reconstructed_content = self.G_style_to_content(fake_style)
        reconstructed_style = self.G_content_to_style(fake_content)

        # Identity mapping
        identity_content = self.G_content_to_style(style_imgs)
        identity_style = self.G_style_to_content(content_imgs)

        # Generator losses
        # Adversarial loss - fool the discriminator
        pred_fake_style = self.D_style(fake_style)
        pred_fake_content = self.D_content(fake_content)
        g_adv_loss = self.adversarial_loss(pred_fake_style, real_target) + \
                    self.adversarial_loss(pred_fake_content, real_target)

        # Cycle consistency loss
        g_cycle_loss = self.cycle_loss(content_imgs, reconstructed_content) + \
                      self.cycle_loss(style_imgs, reconstructed_style)

        # Identity loss
        g_identity_loss = self.identity_loss(style_imgs, identity_content) + \
                         self.identity_loss(content_imgs, identity_style)

        # Total generator loss
        g_loss = g_adv_loss + g_cycle_loss + g_identity_loss
        g_loss.backward()
        self.opt_G.step()

        return {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item(),
            'g_adv_loss': g_adv_loss.item(),
            'g_cycle_loss': g_cycle_loss.item(),
            'g_identity_loss': g_identity_loss.item()
        }

    def train(self, content_dir="content_images", style_dir="style_images"):
        """Train the model"""
        self.load_data(content_dir, style_dir)

        print(f"Starting training for {self.num_epochs} epochs...")

        for epoch in range(self.num_epochs):
            epoch_losses = {'d_loss': 0, 'g_loss': 0, 'g_cycle_loss': 0}

            for batch_idx, (content_imgs, style_imgs) in enumerate(self.dataloader):
                losses = self.train_step(content_imgs, style_imgs)

                for key in epoch_losses:
                    epoch_losses[key] += losses[key]

            # Average losses
            for key in epoch_losses:
                epoch_losses[key] /= len(self.dataloader)

            print(f"Epoch [{epoch+1}/{self.num_epochs}] - "
                  f"D Loss: {epoch_losses['d_loss']:.4f}, "
                  f"G Loss: {epoch_losses['g_loss']:.4f}, "
                  f"Cycle Loss: {epoch_losses['g_cycle_loss']:.4f}")

            # Save samples every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_samples(content_imgs[:1], style_imgs[:1], epoch + 1)
                self.save_checkpoint(epoch + 1)

        print("Training completed!")

    def save_samples(self, content_img, style_img, epoch):
        """Save sample generated images"""
        self.G_content_to_style.eval()
        self.G_style_to_content.eval()

        with torch.no_grad():
            fake_style = self.G_content_to_style(content_img.to(self.device))
            fake_content = self.G_style_to_content(style_img.to(self.device))

            # Denormalize
            content_img = (content_img + 1) / 2
            style_img = (style_img + 1) / 2
            fake_style = (fake_style + 1) / 2
            fake_content = (fake_content + 1) / 2

            # Save
            save_image(content_img, f"{self.save_dir}/content_{epoch}.png")
            save_image(style_img, f"{self.save_dir}/style_{epoch}.png")
            save_image(fake_style, f"{self.save_dir}/fake_style_{epoch}.png")
            save_image(fake_content, f"{self.save_dir}/fake_content_{epoch}.png")

        self.G_content_to_style.train()
        self.G_style_to_content.train()

    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        checkpoint = {
            'G_content_to_style': self.G_content_to_style.state_dict(),
            'G_style_to_content': self.G_style_to_content.state_dict(),
            'D_content': self.D_content.state_dict(),
            'D_style': self.D_style.state_dict(),
            'opt_G': self.opt_G.state_dict(),
            'opt_D': self.opt_D.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint, f"{self.checkpoint_dir}/checkpoint_{epoch}.pth")
        print(f"Saved checkpoint at epoch {epoch}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.G_content_to_style.load_state_dict(checkpoint['G_content_to_style'])
        self.G_style_to_content.load_state_dict(checkpoint['G_style_to_content'])
        self.D_content.load_state_dict(checkpoint['D_content'])
        self.D_style.load_state_dict(checkpoint['D_style'])
        print(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint['epoch']


# ==================== STYLE TRANSFER INFERENCE ====================

class StyleTransfer:
    """Apply style transfer to images"""

    def __init__(self, checkpoint_path, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.generator = Generator().to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['G_content_to_style'])
        self.generator.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        self.denormalize = transforms.Compose([
            transforms.Normalize([0, 0, 0], [2, 2, 2]),
            transforms.Normalize([-0.5, -0.5, -0.5], [1, 1, 1]),
        ])

    def transform(self, image_path, output_path=None, resize_size=256):
        """Transform an image to the target style"""
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size

        # Apply transform
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Generate styled image
        with torch.no_grad():
            styled_tensor = self.generator(image_tensor)

        # Denormalize and convert back to PIL
        styled_tensor = self.denormalize(styled_tensor.squeeze(0).cpu())
        styled_image = transforms.ToPILImage()(styled_tensor.clip(0, 1))

        # Resize back to original size
        styled_image = styled_image.resize(original_size, Image.LANCZOS)

        # Save if output path provided
        if output_path:
            styled_image.save(output_path)
            print(f"Saved styled image to {output_path}")

        return styled_image


# ==================== MIXED STYLE TRANSFER (SLIDER) ====================

class MixedStyleTransfer:
    """Blend between multiple style transfers with slider control"""

    def __init__(self, checkpoints, device=None):
        """
        Args:
            checkpoints: dict mapping style names to checkpoint paths
                        e.g., {'vangogh': 'vangogh.pth', 'picasso': 'picasso.pth'}
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.generators = {}
        for style_name, checkpoint_path in checkpoints.items():
            generator = Generator().to(self.device)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            generator.load_state_dict(checkpoint['G_content_to_style'])
            generator.eval()
            self.generators[style_name] = generator

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def transform(self, image_path, style_weights, output_path=None):
        """
        Transform image with blended styles.

        Args:
            image_path: Path to input image
            style_weights: dict mapping style names to weights
                          e.g., {'vangogh': 0.7, 'picasso': 0.3}
            output_path: Path to save output
        """
        image = Image.open(image_path).convert('RGB')
        original_size = image.size

        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Generate styled versions and blend
        blended_output = torch.zeros_like(image_tensor)

        with torch.no_grad():
            for style_name, weight in style_weights.items():
                if style_name in self.generators:
                    styled = self.generators[style_name](image_tensor)
                    blended_output += weight * styled

            # Normalize weights
            total_weight = sum(style_weights.values())
            if total_weight > 0:
                blended_output = blended_output / total_weight

        # Denormalize
        blended_output = (blended_output.squeeze(0).cpu() + 1) / 2
        blended_image = transforms.ToPILImage()(blended_output.clip(0, 1))
        blended_image = blended_image.resize(original_size, Image.LANCZOS)

        if output_path:
            blended_image.save(output_path)
            print(f"Saved blended style image to {output_path}")

        return blended_image


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    # Example: Train Van Gogh style transfer
    print("=" * 50)
    print("MixStyleGAN - Image Style Transfer")
    print("=" * 50)

    # Create trainer
    trainer = MixStyleGANTrainer(
        content_dir="content_images",
        style_dir="style_images",
        save_dir="generated_images",
        checkpoint_dir="checkpoints",
        image_size=256,
        batch_size=1,
        learning_rate=0.0002,
        num_epochs=100,
        lambda_cycle=10.0,
        lambda_identity=5.0,
    )

    # Train the model
    trainer.train()

    # After training, use style transfer
    style_transfer = StyleTransfer("checkpoints/checkpoint_100.pth")
    styled_image = style_transfer.transform("my_photo.jpg", "my_photo_vangogh.jpg")

    print("\nUsage:")
    print("1. Place content images in 'content_images/' folder")
    print("2. Place Van Gogh paintings in 'style_images/' folder")
    print("3. Uncomment trainer.train() to start training")
    print("4. Use StyleTransfer class to apply style to new images")
