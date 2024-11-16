Understanding Multimodal LLMs

In this article, i aim to explain how multimodal LLMs work and summarize the recent research papers.


Use cases for multimodal LLMs
What are the multimodal LLMs?

Multimodal LLMs are LLMs that accept different types of inputs. One typical example is summarizing an image when we provide it as input.

Another example is extracting information from a PDF and putting it in Latex.

2. Common approaches to building the multimodal LLMs:-

There are two methods:-

Method A:- Unified Embedding Decoder Architecture Approach
Method B:- Cross-modality Attention Architecture approach

Unified Embedding-Decoder Architecture — in this method, images are converted into the same tokens with the same embedding size as the original text tokens, allowing the LLM to process text and image input tokens together after concatenation.

Cross-modality Attention Architecture approach - It employs a cross-attention mechanism to integrate image and text embeddings directly into the attention layer.

2.1 Method A: Unified Embedding Decoder Architecture
Let's begin with Unified Embedding Decoder Architecture


This method encodes the image as embeddings, the same as text.

Text input is tokenized and passed through the embedding layer for typical text-only LLM.

2.1.1 Understanding Image Encoders
Image embeddings are generated using Image encoders.


What happens inside the image encoder? It is first divided into chunks and then converted into embeddings using the vision transformer.


2.1.2 The role of the linear projection module
It is to flatten and project the image patches to 256 and then 756 dimensions.


import torch


class PatchProjectionLayer(torch.nn.Module):

    def __init__(self, patch_size, num_channels, embedding_dim):
        super().__init__()
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embedding_dim = embedding_dim
        self.projection = torch.nn.Linear(
            patch_size * patch_size * num_channels, embedding_dim
        )

    def forward(self, x):

        batch_size, num_patches, channels, height, width = x.size()
        x = x.view(batch_size, num_patches, -1)  # Flatten each patch
        x = self.projection(x)  # Project each flattened patch
        return x


# Example Usage:
batch_size = 1
num_patches = 9  # Total patches per image
patch_size = 16  # 16x16 pixels per patch
num_channels = 3  # RGB image
embedding_dim = 768  # Size of the embedding vector

projection_layer = PatchProjectionLayer(patch_size, num_channels, embedding_dim)

patches = torch.rand(
    batch_size, num_patches, num_channels, patch_size, patch_size
)

projected_embeddings = projection_layer(patches)
print(projected_embeddings.shape)

# This prints
# torch.Size([1, 9, 768])
2.1.3 Image vs text tokenization

As you can see in the figure above, I included an additional projector module that follows the image encoder. This projector is usually just another linear projection layer similar to the one explained earlier. The purpose is to project the image encoder outputs into a dimension that matches the dimensions of the embedded text tokens, as illustrated in the figure below. (As we will see later, the projector is sometimes also called an adapter, adaptor, or connector.)


Now that the image patch embeddings have the same embedding dimension as the text token embeddings, we can concatenate them as input to the LLM, as shown in the figure at the beginning of this section. Below is the same figure again for easier reference.


By the way, the image encoder we discussed in this section is usually a pretrained vision transformer. A popular choice is CLIP or OpenCLIP.

However, there are also versions of Method A that operate directly on patches, such as Fuyu, which is shown in the figure below.


As illustrated in the figure above, Fuyu passes the input patches directly into a linear projection (or embedding layer) to learn its own image patch embeddings rather than relying on an additional pretrained image encoder like other models and methods do. This greatly simplifies the architecture and training setup.

2.2 Method B: Cross-Modality Attention Architecture
Now that we have discussed the unified embedding decoder architecture approach to building multimodal LLMs and understand the basic concept behind image encoding, let’s talk about an alternative way of implementing multimodal LLMs via cross-attention, as summarized in the figure below.


We connect the input patches in the multi-head attention layer via a cross-attention mechanism.


In multimodal LLM, the encoder is an image encoder instead of a text encoder, but the same idea applies.

How does cross-attention work? Let’s look at a conceptual drawing of what happens inside the regular self-attention mechanism.


In the figure above, x is the input, and Wq is a weight matrix used to generate the queries (Q). Similarly, K stands for keys, and V stands for values. A represents the attention scores matrix, and Z is the inputs (x) transformed into the output context vectors.

In cross-attention, in contrast to self-attention, we have two different input sources, as illustrated in the following figure.


In the case of the original transformer architecture in the Attention Is All You Need paper, the two inputs x1 and x2 correspond to the sequence returned by the encoder module on the left (x2) and the input sequence being processed by the decoder part on the right (x1). In the context of a multimodal LLM, x2 is the output of an image encoder. (Note that the queries usually come from the decoder, and the keys and values typically come from the encoder.)

Note that in cross-attention, the two input sequences x1 and x2 can have different numbers of elements. However, their embedding dimensions must match. If we set x1 = x2, this is equivalent to self-attention.

3. Unified decoder and cross-attention model training
Let's talk about we deal with 3 components in model training, below picture talks about it:-


It has two phases:-

Pre-training
Instruction Finetuning
It generally starts with pre-trained, instruction-finetuned text-only LLM as the base model.

The following are the key points:-

Pre-training
- For the image encoder, CLIP is commonly used and often remains unchanged during the entire training process.
- It is also usual to keep the LLM part frozen phase, focusing only on training the projector — a linear layer or a small multi-layer perception.
Instruction Finetuning
- Given the projector’s limited learning capacity, usually comprising just one or two layers, the LLM is often unfrozen during multimodal instruction finetuning (stage 2) to allow for more comprehensive updates. However, note that in the cross-attention-based models (Method B), the cross-attention layers are unfrozen throughout the entire training process.
It mostly depends upon the trade-offs which approach to be taken

Method A: Unified Embedding Decoder Architecture — easier to implement since it doesn’t require any modifications to the LLM architecture itself.
Method B: Cross-modality Attention Architecture — considered more computationally efficient because it doesn’t overload the input context with additional image tokens, introducing them later in the cross-attention layers. Additionally, this approach maintains the text-only performance of the original LLM if the LLM parameters are kept frozen during training. NVIDIA’s NVLM paper has it.
4. Recent multimodal models and methods
The conclusion section at the end has an overview that compares the methods used in these papers.

4.1 The Llama 3 Herd of Models
The multimodal Llama 3.2 models, which come in an 11-billion and 90-billion parameter version, are image-text models that use the previously described cross-attention-based approach, illustrated in the figure below.


Llama 3.2 uses the cross-attention-based approach. We usually freeze the image encoder and only update the LLM parameters during pretraining.

Here, the researchers almost take the opposite approach: they update the image encoder but do not update the language model’s parameters. They write that this is intentional and done to preserve the text-only capabilities so that the 11B and 90B multimodal models can be used as drop-in replacements for the Llama 3.1 8B and 70B text-only model on text tasks.

The training itself is done in multiple iterations, starting with the Llama 3.1 text models. After adding the image encoder and projection (here called “adapter”) layers, they pretrain the model on image-text data. Then, similar to the Llama 3 model text-only training they follow up with instruction and preference fine-tuning.

Instead of adopting a pretrained model such as CLIP as an image encoder, the researchers used a vision transformer that they pretrained from scratch. Specifically, they adopted the ViT-H/14 variant (630 million parameters) of the classic vision transformer architecture. They then pretrained the ViT on a dataset of 2.5 billion image-text pairs over five epochs; this was done before connecting the image encoder to the LLM. (The image encoder takes 224×224 resolution images and divides them into a 14×14 grid of patches, with each patch sized at 16×16 pixels.)

As the cross-attention layers add a substantial amount of parameters, they are only added in every fourth transformer block. (For the 8B model, this adds 3B parameters, and for the 70B model, this adds 20 billion parameters.)

4.2 Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Multimodal Models
The Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Multimodal Models paper (September 25, 2024) is notable because it promises to open source not only the model weights but also the dataset and source code similar to the language-only OLMo LLM.

If you are wondering why the paper title has two names, Molmo refers to the model (Multimodal Open Language Model), and PixMo (Pixels for Molmo) is the dataset.


The figure above illustrates that the image encoder employs an off-the-shelf vision transformer, specifically CLIP. The term “connector” here refers to a “projector” that aligns image features with the language model.

Molmo streamlines the training process by avoiding multiple pretraining stages. Instead, it chooses a simple pipeline that updates all parameters in a unified approach, including those of the base LLM, the connector, and the image encoder.

The Molmo team offers several options for the base LLM:

OLMo-7B-1024 (a fully open model backbone),
OLMoE-1B-7B (a mixture-of-experts architecture; the most efficient model)
Qwen2 7B (an open-weight model that performs better than OLMo-7B-1024)
Qwen2 72B (an open-weight model and the best-performing model)
4.3 NVLM: Open Frontier-Class Multimodal LLMs
NVIDIA’s NVLM: Open Frontier-Class Multimodal LLMs paper (September 17, 2024) is particularly interesting because, rather than focusing on a single approach, it explores both methods:

Method A, the Unified Embedding Decoder Architecture (“decoder-only architecture,” NVLM-D), and
Method B, the Cross-Modality Attention Architecture (“cross-attention-based architecture,” NVLM-X).
Additionally, they develop a hybrid approach (NVLM-H) and provide an apples-to-apples comparison of all three methods.


In short, the research team find that:

NVLM-X demonstrates superior computational efficiency for high-resolution images.
NVLM-D achieves higher accuracy in OCR-related tasks.
NVLM-H combines the advantages of both methods.
Like Molmo and other approaches, they begin with a text-only LLM rather than pretraining a multimodal model from scratch (as this generally performs better). Additionally, they use an instruction-tuned LLM instead of a base LLM. Specifically, the backbone LLM is Qwen2–72B-Instruct (to my knowledge, Molmo used the Qwen2–72B base model).

While training all LLM parameters in the NVLM-D approach, they found that for NVLM-X, freezing the original LLM parameters and training only the cross-attention layers during both pretraining and instruction finetuning works well.

Instead of using a typical CLIP model for the image encoder, they use InternViT-6B, which remains frozen throughout all stages.

The projector is a multilayer perceptron rather than a single linear layer.

4.4 Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution
The previous two papers and models, Molmo and NVLM, were based on Qwen2–72B LLM. In this paper, the Qwen research team itself announces a multimodal LLM, Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution (October 3rd, 2024).

At the core of this work is their so-called “Naive Dynamic Resolution” mechanism (the term “naive” is intentional and not a typo for “native,” though “native” could also be fitting). This mechanism allows the model to handle images of varying resolutions without simple downsampling, enabling the input of images in their original resolution.


The native resolution input is implemented via a modified ViT by removing the original absolute position embeddings and introducing 2D-RoPE.

They used a classic vision encoder with 675M parameters and LLM backbones of varying sizes, as shown in the table below.


The training itself consists of 3 stages: (1) pretraining only the image encoder, (2) unfreezing all parameters (including LLM), and (3) freezing the image encoder and instruction-finetuning only the LLM.

4.5 Pixtral 12B
Pixtral 12B (September 17, 2024), which uses the Method A: Unified Embedding Decoder Architecture approach, is Mistral AI's first multimodal model. Unfortunately, no technical paper or report is available, but the Mistral team shared a few interesting tidbits in their blog post.

Interestingly, they chose not to use a pretrained image encoder, instead training one with 400 million parameters from scratch. For the LLM backbone, they used the 12-billion-parameter Mistral NeMo model.

Similar to Qwen2-VL, Pixtral also supports variable image sizes natively, as illustrated in the figure below.


4.6 MM1.5: Methods, Analysis & Insights from Multimodal LLM Fine-tuning
The MM1.5: Methods, Analysis & Insights from Multimodal LLM Fine-tuning paper (September 30, 2024) provides practical tips and introduces a mixture-of-experts multimodal model alongside a dense model similar to Molmo. The models span a wide size range, from 1 billion to 30 billion parameters.

The models described in this paper focuse on Method A, a Unified Embedding Transformer Architecture, which structures inputs effectively for multimodal learning.

In addition, the paper has a series of interesting ablation studies looking into data mixtures and the effects of using coordinate tokens.


4.7 Aria: An Open Multimodal Native Mixture-of-Experts Model
The Aria: An Open Multimodal Native Mixture-of-Experts Model paper (October 8, 2024) introduces another mixture-of-experts model approach, similar to one of the variants in the Molmo and MM1.5 lineups.

The Aria model has 24.9 billion parameters, with 3.5 billion parameters allocated per text token. The image encoder (SigLIP) has 438-million-parameters.

This model is based on a cross-attention approach with the following overall training procedure:

Training the LLM backbone entirely from scratch.
Pretraining both the LLM backbone and the vision encoder.
4.8 Baichuan-Omni
The Baichuan-Omni Technical Report (October 11, 2024) introduces Baichuan-Omni, a 7-billion-parameter multimodal LLM based on Method A: the Unified Embedding Decoder Architecture approach, as shown in the figure below.

The training process for Baichuan-Omni involves a three-stage approach:


Projector training: Initially, only the projector is trained, while both the vision encoder and the language model (LLM) remain frozen.
Vision encoder training: Next, the vision encoder is unfrozen and trained, with the LLM still frozen.
Full model training: Finally, the LLM is unfrozen, allowing the entire model to be trained end-to-end.
The model utilizes the SigLIP vision encoder and incorporates the AnyRes module to handle high-resolution images through down-sampling techniques.

While the report does not explicitly specify the LLM backbone, it is likely based on the Baichuan 7B LLM, given the model’s parameter size and the naming convention.

4.9 Emu3: Next-Token Prediction is All You Need
The Emu3: Next-Token Prediction is All You Need paper (September 27, 2024) based on a transformer-based decoder architecture. Although it’s not a multimodal LLM in the classic sense (i.e., models focused on image understanding rather than generation), Emu3 is super interesting as it demonstrates that it’s possible to use transformer decoders for image generation, which is a task typically dominated by diffusion methods. (However, note that there have been other similar approaches before, such as Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation.)


The researchers trained Emu3 from scratch and then used Direct Preference Optimization (DPO) to align the model with human preferences.

The architecture includes a vision tokenizer inspired by SBER-MoVQGAN. The core LLM architecture is based on Llama 2, yet it is trained entirely from scratch.

4.10 Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation
We previously focused on multimodal LLMs for image understanding and just saw one example for image generation with Emu 3 above. The Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation paper (October 17, 2024) introduces a framework that unifies multimodal understanding and generation tasks within a single LLM backbone.

A key feature of Janus is the decoupling of visual encoding pathways to address the distinct requirements of understanding and generation tasks. The researchers argue that image understanding tasks require high-dimensional semantic representations, while generation tasks require detailed local information and global consistency in images. By separating these pathways, Janus effectively manages these differing needs.

The model employs the SigLIP vision encoder, similar to that used in Baichuan-Omni, for processing visual inputs. For image generation, it utilizes a Vector Quantized (VQ) tokenizer to handle the generation process. The base LLM in Janus is the DeepSeek-LLM with 1.3 billion parameters.


The training process for the model in this image follows three stages, as shown in the figure below.


In Stage I, only the projector layers and image output layer are trained while the LLM, understanding, and generation encoders remain frozen. In Stage II, the LLM backbone and text output layer are unfrozen, allowing for unified pretraining across understanding and generation tasks. Finally, in Stage III, the entire model, including the SigLIP image encoder, is unfrozen for supervised fine-tuning, enabling the model to fully integrate and refine its multimodal capabilities.

Conclusion
Comparing LLMs and Multimodal LLMs is difficult as public benchmark data may be included in the training.

Architectures are very different, so it becomes more difficult to generate comparable results. The NVIDIA team has developed NVLM in different flavors, which allowed for a comparison between the decoder-only and cross-attention approaches. Below is a summary of all the different kinds of architecture.
