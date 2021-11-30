# VAE
This is a Variational Autoencoder Implementation trained on MNIST
The dimension of the latent variable is chosen to be 3 for visualization purposes. Higher dimensional latent variables will likely get better performance.
This was implemented for my 16-811 project on Variational Inference and Information Theory.

A part of the testing is visualizing images generated along a line in latent space. In the saved model that I trained, this represented the path between the digits 3 and 7 which crossed regions of 8 and 9 as well. With a different random initialization the embedding of digits in latent space will most likely be different and the start/end points of the line may need to change to get a meaningful visualization (this is why I compute the centers of the embeddings for each label). But it should look cool either way.
