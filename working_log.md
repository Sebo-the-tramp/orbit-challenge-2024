# Working Log

So the most interesting things are happening in the /module/protonet_video_post file. There we have canny Edge detector, that could be easily substituted by the backbone, we are going to use. Because anyway the mac to personalize might be even less if we share the backbone.di

### 03/04/2024
So today we were able to do the training/testing on the full dataset, use the follwing code to get enough RAM on A6000, the limit for the 4090 is 44GB:

```bash 
    salloc --x11 -N1 -n12 -w $1 --mem=100G
```

We are trying to replace the backbone:
- mobilevitv2 - 0.75
- phinetBIG 

Let's see. 

Also refactoring of the github repository. It needs to be cleaned.

- Added IFresV4, the log for saving the images that are worth for the support vectors

## Metamadness Hackathon 

### 05/04/2024 - 1st day hackathon

Finally we got some initial results:

this is with FEAT and 50 episodes per users. 3% lower than the orbit-winner-baseline and it might due to some reasons:
- less objects per user in the testing
- different testing settings
- different amount of episodes in training

-> we are trying to bridge the gap!

| Name                     |  Average Frame Accuracy  | Confidence Interval   |
|--------------------------|--------------------------|-----------------------|
| EfficientNet - 50 episodes - big dataset (BASELINE)   | 0.6638479368843097       | ±0.05643283613519739   |
| Phinet - 50 episodes - big dataset   | 0.6509896065430318      |  ±0.05423007214999352 |
| Phinet - 50 episodes - better dataset (non-legit)  | 0.6803234484591006      |  ±0.05423007214999352 |
| Phinet - 500 episodes - better dataset (LEGIT)  | 0.6847902479362096      |  ±0.06035767933311224 |
| Phinet - 50 episodes - better dataset (LEGIT)  | 0.6761672757795106      |  ±0.060917889965372646 |
| Vit_S - 50 episodes - big dataset | - crashes -     |  - crashes - |
| Vit_S - 50 episodes - better dataset | - crashes -     |  - crashes - |

**the crashes are due to the limited memory of the GPU**


### 06/04/2024 - 2nd day hackathon

Today I am trying to see if the vit-small can be used
Create a legit clean testing dataset
- take away the filtering and background 
- also tonight launch efficientnet 500 to see what's wrong in the baseline


I found out out to make the test work by freeing the memory each time after each user, add:

```python
    # crazy thing
    torch.cuda.empty_cache()
```

**you need to add this snippet just after ```model.reset()``` in the file protonet_video_post.py and protonet

NOTE!
we also removed the object of the train user that didn't contain any clean object, as it was useless to have. We were able to do this because the training is not that strict


### 07/04/2024 - 3rd day hackathon 

After 30km in 3.23 minutes here I am trying to make other experiments on the legit database
I got the results from 500-episodes-legit 
Now I want to see how big of a difference there is between 500 episodes and 50.

pytorch_lightning.loops.epoch.evaluation_epoch_loop -> this is what I should modify lol and I don't know where to start from

-> https://lightning.ai/docs/pytorch/stable/common/lightning_module.html saviour


I started a huuuuge work of changing the code. THis is badass cool but suuuper difficoult
Something is cooking in the pot, the testing is going, but I don't know what is happening.

Also the problem is conneted to the clip lenght, with 2 it works, with 1 it does not. Start there tomorrow

### 08/04/2024 - 4th day hackathon 

Trying to solve the previous day's bug and trying to finish some coding


-> I think yesterday I did overengineered everything, today it seems it is working the testing, but I am a bit doubtious on the availability of the GPU, i think that not all images are loaded in, now I will investigate.

On 5 episodes per user I get:

```The average frame accuracy across all 300 testing videos  = 0.618925481```

It seems it is correct... the memory requirements is small as it seems.
Let's calculate for each user the total number of images.

#### Trying to find the bug

P901 analysis 
11 objects

    - 1. 3 -> 3 x 200
    - 2. 2 -> 2 x 200
    - 3. 2 -> 2 x 200
    - 4. 2 -> 2 x 200
    - 5. 1 -> 1 x 200
    - 6. 5 -> 5 x 200
    - 7. 7 -> 7 x 200
    - 8. 2 -> 2 x 200
    - 9. 2 -> 2 x 200
    - 10.1 -> 1 x 200
    - 11.2 -> 2 x 200

    total videos -> 3

 -> 3+2+2+2+1+5+7+2+2+1+2 -> the number of videos per person are correct

 The problem I see is that I always get the same accuracy. Even if that should be random. I don't like I will investigate.

 It changes if I change the number of episodes per user... 

 ```The average frame accuracy across all 300 testing videos  = 0.6167484619213235, confidence_interval = 0.040658351955932405```

 Moreover some images are very tricky to differentiate: eg ```/home/zhumakhanova/Desktop/cv703_project/orbit_benchmark_224/dataset_legit/orbit_benchmark_224/test/P204/airpods/clutter/P204--airpods--clutter--8xtbt38Q53Ug5GhxqdlXPJA9nMVqGKw9jc2FG_-Opwg/P204--airpods--clutter--8xtbt38Q53Ug5GhxqdlXPJA9nMVqGKw9jc2FG_-Opwg-00004.jpg```

 I also have to change the json file to be submittable

 Ok what is doing, is only saving 1 of the episodes in the file, in 204 there are 90 occurrences of ```p204```, 1 for initial dict and the other 89 for each clutter video. There should be 3 x 89

 #### RANDOM SEED
 I see why the result is always the same:

 by seeding the same number at the beginnig, and assuming the order of things doesn't change, the samplers will always return the same things. We therefore might want to change the seed ahahahahahah. Finetunining the seed ofc

## Overall contributions

- show that a lightweight convnet can substitute efficientnet and still give reasonable results by halving the number of parameters
- show that a clever sampling strategy and cleaning of the dataset improves the accuracy -> (sobstitute the canny edge detector with phinet IFnet ) (for training purposes we could even say that we used a lookup table to counteract for low computational power)
- demonstrate that trough some tricks it is possible to train the protonet on a 4090.
- demonstrate that a vit might give more accuracy
- if we have phinet self supervised, make the comparison.



