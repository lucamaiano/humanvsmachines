
# Test Data
[Download](https://drive.google.com/drive/folders/1A-JOqRficR9Jr9ySjT_M02xmvLuxT1Ym?usp=sharing)

# Pretrained models
[Download](https://drive.google.com/drive/folders/11v3K3GcJbNE-YV2u9OKXDIwwgBDVh4Gp?usp=sharing)


## Dataset (from the old repo)
| Real (FFHQ)   | Stable Diffusion (ours) | GAN    | GAN2   | GAN3  |
| ------------- |:-----------------------:|:------:|:------:|:-----:|
|![alt text](https://github.com/LucaCorvittoblob/main/readme_images/real.png)|![alt text](https://github.com/LucaCorvittoblob/main/readme_images/fake.png)| ![alt text](https://github.com/LucaCorvittoblob/main/readme_images/gan.png)| ![alt text](https://github.com/LucaCorvittoblob/main/readme_images/gan2.png)| ![alt text](https://github.com/LucaCorvittoblob/main/readme_images/gan3.png)|

The fake generated dataset that we propose is available at the drive folder: [Stable Diffusion fakes](https://drive.google.com/drive/folders/10-n9jY3USb5O_2bh4yUpo1IRPWxe1RIA). The dataset was created using the prompts inside the [`prompts.txt`](prompts.txt) file. Each image's name is structured in this way:
```
<#prompt>-<#process>
```
where `<#prompt>` is the associated number of the prompt in the [`prompts.txt`](prompts.txt) file, numbered from 0, and `<#process>` is the number representing the order in which the image was generated starting from 0 (each of these images is generated using a different seed). For example, the file named `0-0` is the first generated image from the first prompt in the prompts file, while the one named `248-70` is the 71-st generated image from the 249-th prompt in the file:
| Generated Image | Prompt |
| --------------- |:-------:|
| ...| ... |
|![alt text](https://github.com/LucaCorvittoblob/main/readme_images/4-54.png)| headshot portrait of a nigerian man, real life, realistic background, 50mm, Facebook, Instagram, shot on iPhone, HD, HDR color, 4k, natural lighting, photography |
| ...| ... |
|![alt text](https://github.com/LucaCorvittoblob/main/readme_images/248-70.png)| headshot portrait of an old woman with braids blonde hair, real life, shot on iPhone, realistic background, HD, HDR color, 4k, natural lighting, photography, Facebook, Instagram, Pexels, Flickr, Unsplash, 50mm, 85mm, #wow, AMAZING, epic details, epic, beautiful face, fantastic, cinematic, dramatic lighting|
| ...| ... |

However, new images can be generated using the code [`main.py`](main.py). In order to generate images from different prompts then the [`prompts.txt`](prompts.txt) file must be updated.

The other datasets used in this project for detection and classification purpose were taken from external resources. They are:
* [FFHQ dataset](https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq) composed by real faces images;
* [StyleGAN dataset](https://iplab.dmi.unict.it/deepfakechallenge/training/1-STYLEGAN.zip) made available for the [Deepfake challenge](https://iplab.dmi.unict.it/deepfakechallenge/#[object%20Object]);
* [StyleGAN2 dataset](https://www.kaggle.com/datasets/bwandowando/all-these-people-dont-exist) composed by the images generated from the famous website [This Person Does Not Exist](https://thispersondoesnotexist.com/);
* [StyleGAN3 dataset](https://nvlabs-fi-cdn.nvidia.com/stylegan3/images/) made available directly from NVIDIA.


