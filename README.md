# NLP Project - ImagiNarrate: Building a Narrative with Images and Generated Captions

Git Repository for NLP Project of Group 46.

## Abstract

In this paper, we introduce a new natural language processing (NLP) approach to solve the
problem of visual storytelling that utilizes image features to generate captions and subsequently
develop a coherent story line for the images. By incorporating image features in
the caption generation process, our proposed approach aims to provide a more relevant and
informative description of the images that can be used to build a cohesive and engaging narrative.
We evaluate our model in comparison to the AREL model (Wang et al., 2018) used
for story generation on the basis of traditional metrics like Meteor and Bleu as well as human
evaluation of the generated stories.

## Acknowledgements & References
* [VIST evaluation code](https://github.com/lichengunc/vist_eval)
```
@InProceedings{xwang-2018-AREL,
  author = 	"Wang, Xin and Chen, Wenhu and Wang, Yuan-Fang and Wang, William Yang",
  title = 	"No Metrics Are Perfect: Adversarial Reward Learning for Visual Storytelling",
  booktitle = 	"Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  year = 	"2018",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"899--909",
  location = 	"Melbourne, Australia",
  url = 	"http://aclweb.org/anthology/P18-1083"
  git = "https://github.com/eric-xw/AREL.git"
}
```
Downloaded the preprocessed ResNet-152 features [here](https://vist-arel.s3.amazonaws.com/resnet_features.zip) for Image Embeddings.