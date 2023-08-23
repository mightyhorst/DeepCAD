# 2. Related work
Parametric shape inference. Advance in deep learning
has enabled neural network models that analyze geometric
data and infer parametric shapes. 

<mark>ParSeNet</mark> [38] decomposes
a 3D point cloud into a set of parametric surface patches.
PIE-NET [43] extracts parametric boundary curves from
3D point clouds. UV-Net [19] and BrepNet [24] focus on
encoding a parametric model’s boundary curves and surfaces.
Li et al. [25] trained a neural network on synthetic data to
convert 2D user sketches into CAD operations. Recently,
Xu et al. [51] applied neural-guided search to infer CAD
modeling sequence from parametric solid shapes.
Generative models of 3D shapes. Recent years have also
witnessed increasing research interests on deep generative
models for 3D shapes. Most existing methods generate 3D
shapes in discrete forms, such as voxelized shapes [49, 16,
27, 26], point clouds [6, 52, 53, 8, 30], polygon meshes [17,
42, 31], and implicit signed distance fields [12, 33, 29, 50,
11]. The resulting shapes may still suffer from noise, lack
sharp geometric features, and are not directly user editable.
Therefore, more recent works have sought neural network models that generate 3D shape as a series of geometric
operations. CSGNet [37] infers a sequence of Constructive Solid Geometry (CSG) operations based on voxelized
shape input; and UCSG-Net [21] further advances the inference with no supervision from ground truth CSG trees.

3D shapes using their proposed domain specific languages
(DSLs) [39, 41, 30, 20]. For example, Jones et al. [20] proposed ShapeAssembly, a DSL that constructs 3D shapes by
structuring cuboid proxies in a hierarchical and symmetrical fashion, and this structure can be generated through a
variational autoencoder.


In contrast to all these works, our autoencoder network
outputs CAD models specified as a sequence of CAD operations. CAD models have become the standard shape representation in almost every sectors of industrial production.


Thus, the output from our network can be readily imported
into any CAD tools [1, 2, 3] for user editing. It can also
be directly converted into other shape formats such as point
clouds and polygon meshes. To our knowledge, this is the
first generative model directly producing CAD designs.
Transformer-based models. 

Technically, our work is related to the Transformer network [40], which was introduced as an attention-based building block for many natural
language processing tasks [13]. The success of the Transformer network has also inspired its use in image processing
tasks [34, 9, 14] and for other types of data [31, 10, 44].

Concurrent works [47, 32, 15] on constrained CAD sketches
generation also rely on Transformer network.
Also related to our work is DeepSVG [10], a Transformer based network for the generation of Scalable Vector Graphic
(SVG) images. SVG images are described by a collection
of parametric primitives (such as lines and curves). Apart
from limited in 2D, those primitives are grouped with no specific order or dependence. In contrast, CAD commands are
described in 3D; they can be interdependent (e.g., through
CSG boolean operations) and must follow a specific order.
We therefore seek a new way to encode CAD commands and
their sequential order in a Transformer-based autoencoder

# Summary
**Context:**
This section discusses related work in the field of computer-aided design (CAD) models and generative models for 3D shapes. It highlights various approaches in deep learning and machine learning for understanding and generating parametric shapes and 3D models. The section also mentions the relevance of Transformer-based models, especially in the context of generating CAD designs.

1. **Summary in Simple Language:**
   This section talks about other research that's similar to the paper's topic. It mentions different ways that computer programs have been taught to understand and make 3D shapes. Some methods help the computer figure out shapes based on points in space or lines. Others focus on the sequence of steps needed to create a shape, which is important for CAD designs used in industries. The paper is unique because it teaches computers to create CAD designs directly, which is different from what most other methods do. It also mentions the use of a powerful technique called the Transformer, often used for understanding languages and now applied to creating 3D shapes.

2. **Relevant Deep Learning, Machine Learning, and Math Topics:**
   - Parametric Shape Inference
   - Generative Models for 3D Shapes
   - Deep Learning Models (e.g., Neural Networks)
   - Transformer Networks
   - Attention Mechanisms
   - Autoencoders
   - Scalable Vector Graphics (SVG)
   - Constructive Solid Geometry (CSG)

3. **Simple Python Code Using PyTorch:**

   Here's a simplified Python code snippet using PyTorch to create a basic Transformer-based model for a simple sequence-to-sequence task. This example does not directly implement the paper's approach but demonstrates the use of a Transformer-like architecture for sequence generation, which is relevant to the paper's discussion.

   ```python
   import torch
   import torch.nn as nn

   # Define a simple Transformer-like model for sequence-to-sequence tasks
   class Transformer(nn.Module):
       def __init__(self, input_dim, output_dim, n_heads, n_layers):
           super(Transformer, self).__init__()
           self.embedding = nn.Embedding(input_dim, 512)
           self.transformer = nn.Transformer(512, n_heads, n_layers)
           self.fc = nn.Linear(512, output_dim)

       def forward(self, src):
           embedded = self.embedding(src)
           output = self.transformer(embedded)
           output = self.fc(output)
           return output

   # Initialize the model
   input_dim = 100  # Example input dimension
   output_dim = 50  # Example output dimension
   n_heads = 4      # Number of attention heads
   n_layers = 2     # Number of transformer layers

   model = Transformer(input_dim, output_dim, n_heads, n_layers)

   # Generate a sequence (input sequence length = 10)
   input_sequence = torch.randint(0, input_dim, (10,))

   # Forward pass
   output_sequence = model(input_sequence.unsqueeze(0))

   print(output_sequence.shape)  # Example: torch.Size([1, 10, 50])
   ```

   This code demonstrates a basic Transformer-like model using PyTorch for sequence-to-sequence tasks. The paper's approach uses a Transformer-based autoencoder for CAD commands, which is more complex and tailored to CAD data, but this simple example illustrates the concept of using Transformers for sequence generation.

