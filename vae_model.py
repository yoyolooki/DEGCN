import torch
from torch import nn


# VAE model
class VAE(nn.Module):
    def __init__(self, num_hidden, omics1_dim, omics2_dim, omics3_dim, a=0.4, b=0.3, c=0.3):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.num_hidden = num_hidden
        self.mu_omics1 = nn.Linear(self.num_hidden, self.num_hidden)
        self.mu_omics2 = nn.Linear(self.num_hidden, self.num_hidden)
        self.mu_omics3 = nn.Linear(self.num_hidden, self.num_hidden)
        self.log_var_omics1 = nn.Linear(self.num_hidden, self.num_hidden)
        self.log_var_omics2 = nn.Linear(self.num_hidden, self.num_hidden)
        self.log_var_omics3 = nn.Linear(self.num_hidden, self.num_hidden)

        # Set the number of hidden units

        # Define the encoder part of the autoencoder
        self.encoder_omics1 = nn.Sequential(
            nn.Linear(omics1_dim, num_hidden),
            nn.BatchNorm1d(num_hidden),
            nn.Sigmoid(),
        )
        self.encoder_omics2 = nn.Sequential(
            nn.Linear(omics2_dim, num_hidden),
            nn.BatchNorm1d(num_hidden),
            nn.Sigmoid(),
        )
        self.encoder_omics3 = nn.Sequential(
            nn.Linear(omics3_dim, num_hidden),
            nn.BatchNorm1d(num_hidden),
            nn.Sigmoid(),
        )

        # Define the decoder part of the autoencoder
        self.decoder_omics1 = nn.Sequential(
            nn.Linear(self.num_hidden, omics1_dim),
        )
        self.decoder_omics2 = nn.Sequential(
            nn.Linear(self.num_hidden, omics2_dim),
        )
        self.decoder_omics3 = nn.Sequential(
            nn.Linear(self.num_hidden, omics3_dim),
        )

    def reparameterize(self, mu, log_var):
        # Compute the standard deviation from the log variance
        std = torch.exp(0.5 * log_var)
        # Generate random noise using the same shape as std
        eps = torch.randn_like(std)
        # Return the reparameterized sample
        return mu + eps * std

    def forward(self, omics1, omics2, omics3):
        encoded_omics1 = self.encoder_omics1(omics1)
        encoded_omics2 = self.encoder_omics2(omics2)
        encoded_omics3 = self.encoder_omics3(omics3)

        latent_data = torch.mul(encoded_omics1, self.a) + torch.mul(encoded_omics2, self.b) + torch.mul(
            encoded_omics3, self.c)

        mu_omics1 = self.mu_omics1(encoded_omics1)
        mu_omics2 = self.mu_omics2(encoded_omics2)
        mu_omics3 = self.mu_omics3(encoded_omics3)
        log_var_omics1 = self.log_var_omics1(encoded_omics1)
        log_var_omics2 = self.log_var_omics2(encoded_omics2)
        log_var_omics3 = self.log_var_omics3(encoded_omics3)
        # Reparameterize the latent variable
        z_omics1 = self.reparameterize(mu_omics1, log_var_omics1)
        z_omics2 = self.reparameterize(mu_omics2, log_var_omics2)
        z_omics3 = self.reparameterize(mu_omics3, log_var_omics3)

        # test
        latent_data_z = torch.mul(z_omics1, self.a) + torch.mul(z_omics2, self.b) + torch.mul(
            z_omics3, self.c)

        # Pass the latent variable through the decoder
        decoded_omics1 = self.decoder_omics1(z_omics1)
        decoded_omics2 = self.decoder_omics2(z_omics2)
        decoded_omics3 = self.decoder_omics3(z_omics3)
        # Return the encoded output, decoded output, mean, and log variance
        return latent_data, latent_data_z, decoded_omics1, decoded_omics2, decoded_omics3, mu_omics1, mu_omics2, mu_omics3, log_var_omics1, log_var_omics2, log_var_omics3


# CVAE model
class ConditionalVAE(VAE):
    # VAE implementation from the article linked above
    def __init__(self, num_hidden, num_classes, omics1_dim, omics2_dim, omics3_dim):
        super().__init__(num_hidden, omics1_dim, omics2_dim, omics3_dim)
        # Add a linear layer for the class label
        self.label_projector_omics1 = nn.Sequential(
            nn.Linear(num_classes, self.num_hidden),
            nn.ReLU(),
        )
        self.label_projector_omics2 = nn.Sequential(
            nn.Linear(num_classes, self.num_hidden),
            nn.ReLU(),
        )
        self.label_projector_omics3 = nn.Sequential(
            nn.Linear(num_classes, self.num_hidden),
            nn.ReLU(),
        )

    def condition_on_label_omics1(self, z1, y):
        projected_label = self.label_projector_omics1(y.float())
        return z1 + projected_label

    def condition_on_label_omics2(self, z2, y):
        projected_label = self.label_projector_omics2(y.float())
        return z2 + projected_label

    def condition_on_label_omics3(self, z3, y):
        projected_label = self.label_projector_omics3(y.float())
        return z3 + projected_label

    def forward(self, omics1, omics2, omics3, y):
        encoded_omics1 = self.encoder_omics1(omics1)
        encoded_omics2 = self.encoder_omics2(omics2)
        encoded_omics3 = self.encoder_omics3(omics3)

        mu_omics1 = self.mu_omics1(encoded_omics1)
        mu_omics2 = self.mu_omics2(encoded_omics2)
        mu_omics3 = self.mu_omics3(encoded_omics3)
        log_var_omics1 = self.log_var_omics1(encoded_omics1)
        log_var_omics2 = self.log_var_omics2(encoded_omics2)
        log_var_omics3 = self.log_var_omics3(encoded_omics3)
        # Reparameterize the latent variable
        z_omics1 = self.reparameterize(mu_omics1, log_var_omics1)
        z_omics2 = self.reparameterize(mu_omics2, log_var_omics2)
        z_omics3 = self.reparameterize(mu_omics3, log_var_omics3)
        # Pass the latent variable through the decoder
        decoded_omics1 = self.decoder_omics1(self.condition_on_label_omics1(z_omics1, y))
        decoded_omics2 = self.decoder_omics2(self.condition_on_label_omics2(z_omics2, y))
        decoded_omics3 = self.decoder_omics3(self.condition_on_label_omics3(z_omics3, y))
        # Return the encoded output, decoded output, mean, and log variance
        return decoded_omics1, decoded_omics2, decoded_omics3, mu_omics1, mu_omics2, mu_omics3, log_var_omics1, log_var_omics2, log_var_omics3

    def cvae_sample(self, num_samples, y, device=torch.device('cuda')):
        with torch.no_grad():
            # Generate random noise
            z1 = torch.randn(num_samples, self.num_hidden).to(device)
            z2 = torch.randn(num_samples, self.num_hidden).to(device)
            z3 = torch.randn(num_samples, self.num_hidden).to(device)
            # Pass the noise through the decoder to generate samples
            samples1 = self.decoder_omics1(self.condition_on_label_omics1(z1, y))
            samples2 = self.decoder_omics2(self.condition_on_label_omics2(z2, y))
            samples3 = self.decoder_omics3(self.condition_on_label_omics3(z3, y))
        # Return the generated samples
        return samples1, samples2, samples3
