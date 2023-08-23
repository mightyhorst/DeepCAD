# 3.2. Autoencoder for CAD Models
We now introduce an autoencoder network that leverages
our representation of CAD commands. Figure 3 illustrates
its structure, and more details are provided in Sec. C of
supplementary document. 

> Figure 3. Our network architecture. 
```
[C1 ... Cn]
    |
    |
[ Embedding ]
    |
    |
[ Transformer Encoder ]
    |
    |
[Average Pooling]
    |
    |
[Latent vector z]
    |
    |
[ Transformer Decoder D]
    |
    |
[Linear]
    |
    |
[C_hat_1 ... C_hat_n]

```
The input CAD model, represented as a command sequence `M = {Ci} Nc i=1` is first projected to an embedding space and then fed to the encoder `E` resulting in a latent vector `z`. The decoder `D` takes learned constant embeddings as input, and also attends to the latent vector `z`. It then outputs the predicted command sequence `M_hat = {C_hat_i} Nc i=1`

Once trained, the decoder part of the network will serve naturally as a CAD generative model.
Our autoencoder is based on the Transformer network, inspired by its success for processing sequential data [40, 13, 28]. Our autoencoder takes as input a CAD command sequence `M = [C1, · · · , CNc]`, where `Nc` is a fixed number (recall Sec. 3.1.2). 

First, each command `Ci` is projected separately onto a continuous embedding space of dimension
`d_E = 256`. Then, all the embeddings are put together to feed into an encoder `E`, which in turn outputs a latent vector `z ∈ R^256`. The decoder takes the latent vector `z` as input,
and outputs a generated CAD command sequence `M_hat`

