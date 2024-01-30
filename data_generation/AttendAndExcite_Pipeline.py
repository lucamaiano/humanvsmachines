import torch
from diffusers import StableDiffusionAttendAndExcitePipeline
import os
from torch import autocast
"""
Script for generating images using the Stable Diffusion Attend and Excite model with specific features, 
and utilizing negative prompts for image generation. Images are saved to specified directories.


Usage:
1. Make sure to have the required dependencies installed.
2. Set your Hugging Face model token in the YOUR_TOKEN variable.
3. Adjust other parameters such as image dimensions, batch size, and model identifier.
4. Specify the directories for training, testing, and evaluation images.
5. Run the script.

Parameters:
- start_index (int): Starting index for processing prompts. Default is 0.

Functions:
- roundplus(a, b): Rounds 'a' up to the nearest multiple of 'b'.
- makedir(path): Creates a directory if it doesn't exist.

Main Function:
- main(start_index): Main function to generate and save images for training, testing, and evaluation sets.
  - Loads prompts from 'prompts.txt'.
  - Initializes the StableDiffusionAttendAndExcitePipeline.
  - Rounds up the number of images to be generated for each set based on the batch size.
  - Processes prompts and generates images for training, testing, and evaluation sets, saving them in the specified directories.

Note: Commented-out code for saving images in the training and testing sets is provided for reference but is currently disabled.
Uncomment the chunk of code you need for either generating train,eval or test.
"""


def roundplus(a, b):
    return b * ((a + b - 1) // b)


def makedir(path):
    os.makedirs(path, exist_ok=True)


def main(start_index=0):
    YOUR_TOKEN = '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'  # Insert your actual token
    num_inference_steps=50
    height=512
    width=512
    guidance_scale = 7
    num_train = 80 
    num_test = 10
    num_eval = 10
    high_end_gpu = True
    batch_size = 1



    with open('prompts.txt') as f:
        lines = f.readlines()

    prompts = []
    for i in lines:
        prompts.append(i.strip())


    if high_end_gpu:
        model_identifier = "CompVis/stable-diffusion-v1-4"  # Use the correct model identifier
    else:
        model_identifier = "CompVis/stable-diffusion-v1-4"  # Use the correct model identifier


    pipe = StableDiffusionAttendAndExcitePipeline.from_pretrained(model_identifier, torch_dtype=torch.float16, use_auth_token=YOUR_TOKEN)
    pipe = pipe.to("cuda")


    num_train = roundplus(num_train, batch_size)
    num_test = roundplus(num_test, batch_size)
    num_eval = roundplus(num_eval, batch_size)


    train_dir = 'train'  # Specify the directory paths
    test_dir = 'test'
    eval_dir = 'eval'

    #makedir('train')
    #makedir('test')
    #makedir('eval')


    train_image_count = 0
    test_image_count = 0
    eval_image_count = 0

    
    negative_prompt = ["disfigured, ugly, bad, immature, cartoon, anime, 3d, painting, b&w"]  # Define negative_prompt as a list of strings

    


    for i, prompt in enumerate(prompts[start_index:], start=start_index):
        prompt = prompt.strip()  # Remove any leading/trailing whitespace
        token_indices = [1,2,3,4,5,6]


#########################################################
##############   Saving images in train  ################
#########################################################
        
        """
        
        inc = 0
        for i, prompt in enumerate(prompts[start_index:], start=start_index):
            prompt = [prompt] * batch_size
            inc=0
            for j in range(num_train):
                if high_end_gpu:
                    images = pipe(prompt = prompt, num_inference_steps = num_inference_steps,guidance_scale=guidance_scale, height=height, width=width, token_indices = token_indices,negative_prompt=negative_prompt,).images
                else:
                    with autocast("cuda"):
                        images = pipe(prompt = prompt, num_inference_steps = num_inference_steps,guidance_scale=guidance_scale, height=height, width=width, token_indices = token_indices,negative_prompt=negative_prompt,).images
                for image in images:
                    image.save(os.path.join(train_dir, "{}-{}.png".format(i, inc)))  # Save images in the train directory
                    inc+=1
                    train_image_count += 1 
                    if train_image_count >= 7000:
                        break
                if train_image_count >= 7000:
                    break
        """
    

        
#########################################################
##############   Saving images in test   ################
#########################################################

        
        """
                for i, prompt in enumerate(prompts[start_index:], start=start_index):
            prompt = [prompt] * batch_size
            inc=0
            for j in range(num_test):
                if high_end_gpu:
                    images = pipe(prompt = prompt, num_inference_steps = num_inference_steps,guidance_scale=guidance_scale, height=height, width=width, token_indices = token_indices,negative_prompt=negative_prompt,).images            
                else:
                    with autocast("cuda"):
                        images = pipe(prompt = prompt, num_inference_steps = num_inference_steps,guidance_scale=guidance_scale, height=height, width=width, token_indices = token_indices,negative_prompt=negative_prompt,).images
                for image in images:
                    image.save(os.path.join(test_dir, "{}-{}.png".format(i, inc)))  # Save images in the test directory
                    inc+=1
                    test_image_count += 1 
                    if test_image_count >= 590:
                        break
                if test_image_count >= 590:
                    break
        """
    
    
        
    
#########################################################
##############   Saving images in eval   ################
#########################################################
    
    
    for i, prompt in enumerate(prompts[start_index:], start=start_index):
        prompt = [prompt] * batch_size
        inc=0
        for j in range(num_eval):
            if high_end_gpu:
                images = pipe(prompt = prompt, num_inference_steps = num_inference_steps,guidance_scale=guidance_scale, height=height, width=width, token_indices = token_indices,negative_prompt=negative_prompt,).images
            else:
                with autocast("cuda"):
                    images = pipe(prompt = prompt, num_inference_steps = num_inference_steps,guidance_scale=guidance_scale, height=height, width=width, token_indices = token_indices,negative_prompt=negative_prompt,).images
                
            for image in images:
                image.save(os.path.join(eval_dir, "{}-{}.png".format(i, inc)))  # Save images in the eval directory
                inc+=1
                eval_image_count += 1 
                if eval_image_count >= 269:
                    break
            if eval_image_count >= 269:
                break




if __name__ == "__main__":
    start_index = 42# change this value to wherever you left off -1
    main(start_index)


