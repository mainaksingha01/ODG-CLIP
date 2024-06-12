# Unknown Prompt, the only Lacuna: Unveiling CLIP’s Potential for Open Domain Generalization

Official repository of ODG-CLIP, one of the first works in Open Domain Generalization using pre-trained vision-language model (VLM) [CLIP](https://arxiv.org/abs/2103.00020) to focus on the completely unlablled real-world open samples.

[![arXiv](https://img.shields.io/badge/arXiv-Paper-brightgreen)](https://arxiv.org/pdf/2404.00710)
[![poster](https://img.shields.io/badge/Poster-yellow)](https://github.com/mainaksingha01/ODG-CLIP/blob/master/docs/odgclip-poster.pdf)
[![ppt](https://img.shields.io/badge/PPT-orange)](https://github.com/mainaksingha01/ODG-CLIP/blob/master/docs/odgclip-ppt.pptx)
[![video](https://img.shields.io/badge/Video-darkred)](https://www.youtube.com/watch?v=AWUtIgpo6oM)

## Abstract
<img src="https://github.com/mainaksingha01/ODG-CLIP/blob/master/images/teaser.png" width="1000">

We delve into Open Domain Generalization (ODG), marked by domain and category shifts between training’s labeled source and testing’s unlabeled target domains. Existing solutions to ODG face limitations due to constrained generalizations of traditional CNN backbones and
errors in detecting target open samples in the absence of prior knowledge. Addressing these pitfalls, we introduce ODG-CLIP, harnessing the semantic prowess of the vision-language model, CLIP. Our framework brings forth three primary innovations: Firstly, distinct from prevailing paradigms, we conceptualize ODG as a multi-class classification challenge encompassing both known and novel categories. Central to our approach is modeling a unique prompt tailored for detecting unknown class samples, and to train this, we employ a readily accessible stable diffusion model, elegantly generating proxy images for the open class. Secondly, aiming for domain-tailored classification (prompt) weights while ensuring a balance of precision and simplicity, we devise a novel visual style-centric prompt learning mechanism. Finally, we infuse images with class-discriminative knowledge derived from the prompt space to augment the fidelity of CLIP’s visual embeddings. We introduce a novel objective to safeguard the continuity of this infused semantic intel across domains, especially for the shared classes. Through rigorous testing on diverse datasets, covering closed and open-set DG contexts, ODG-CLIP demonstrates clear supremacy, consistently outpacing peers with performance boosts between 8%-16%.

## Architecture

<img src="https://github.com/mainaksingha01/ODG-CLIP/blob/master/images/architecture.png" width="800">

## Code

 - First of all, clone the awesome toolbox of [dassl](https://github.com/KaiyangZhou/Dassl.pytorch/tree/master/dassl) inside this repo.
 - Install `Stable Diffusion` using `diffusers` following the versions in `requirements.txt`.
 - Main model `odgclip.py` is in the `trainer` folder.
 - Train files are given separately for each of the datasets. Testing is also done in each of the epochs while training, and the best train model is saved accordingly. The results will be saved in a txt file.
 - To train the PACS dataset,
 
 ```
$ python pacs.py
```

- Change the source domains and target domain orders accordingly . For examples, domains = ['art_painting', 'cartoon', 'photo', 'sketch']. First three are the source domains and the last one is the target domain.
- Results will be saved in `results` folder.
- Training models of each epoch will be saved in `train_models` folder.
- The training model that performed best while testing will be saved in `test_models` folder.


## Bibtex

Please cite the paper if you use our work . Thanks.

```
@article{singha2024unknown,
  title={Unknown Prompt, the only Lacuna: Unveiling CLIP's Potential for Open Domain Generalization},
  author={Singha, Mainak and Jha, Ankit and Bose, Shirsha and Nair, Ashwin and Abdar, Moloud and Banerjee, Biplab},
  journal={arXiv preprint arXiv:2404.00710},
  year={2024}
}
```

